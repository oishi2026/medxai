"""Train the CheXpert multi-label ResNet-50 backbone.

Designed for free-tier survival: per-epoch checkpoints to persistent storage,
clean --resume, and a --smoke flag to verify the whole loop on a few batches
before spending GPU hours.

Example (Kaggle):
    python -m medxai.backbones.train_chest \
        --chest_train /kaggle/input/medxai-splits-v1/chest_train.csv \
        --chest_val   /kaggle/input/medxai-splits-v1/chest_val.csv \
        --image_root  /kaggle/input/datasets/ashery/chexpert \
        --out_dir     /kaggle/working/outputs/chest \
        --loss bce --smoke        # drop --smoke for the real run
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from medxai.backbones.losses import build_loss
from medxai.backbones.model import (
    ResNet50MultiLabel,
    freeze_until,
    set_frozen_bn_eval,
)
from medxai.data.dataset import ChestDataset, compute_pos_weight
from medxai.data.splits import CHEXLOCALIZE_PATHOLOGIES
from medxai.utils.determinism import make_generator, seed_worker, set_determinism
from medxai.utils.metrics import macro_auroc


def _load_frozen(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, label_cols):
    model.eval()
    all_t, all_p, losses = [], [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
            losses.append(loss_fn(logits, y).item())
        all_p.append(torch.sigmoid(logits).float().cpu().numpy())
        all_t.append(y.cpu().numpy())
    t, p = np.concatenate(all_t), np.concatenate(all_p)
    macro, per_class = macro_auroc(t, p, label_cols)
    return float(np.mean(losses)), macro, per_class


def save_ckpt(path, model, opt, sched, scaler, epoch, best, cfg):
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict() if sched else None,
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best": best,
            "cfg": cfg,
        },
        path,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chest_train", required=True)
    ap.add_argument("--chest_val", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--strip_prefix", default="",
                    help="leading manifest-path segment absent on disk, "
                         "e.g. 'CheXpert-v1.0-small/'")
    ap.add_argument("--out_dir", default="/kaggle/working/outputs/chest")
    ap.add_argument("--frozen", default="conf/frozen.yaml")
    ap.add_argument("--loss", choices=["bce", "focal"], default="bce")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3,
                    help="dropout before the classifier head")
    ap.add_argument("--freeze_until", default="layer2",
                    help="freeze stages up to this one; 'none' to train all. "
                         "layer3 stays trainable for the GNN arm.")
    ap.add_argument("--strong_aug", action="store_true", default=True,
                    help="translation + mild contrast jitter (anatomy-safe)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--resolution", type=int, default=None, help="override frozen")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="few batches, 1 epoch")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default=None,
                    help="override W&B project (else config.yaml wandb.project, else 'medxai')")
    args = ap.parse_args()

    cfg = _load_frozen(args.frozen)
    seed = cfg["seeds"][0]
    set_determinism(seed)
    resolution = args.resolution or cfg["input_resolution"]["chest"]
    label_cols = CHEXLOCALIZE_PATHOLOGIES
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # data
    g = make_generator(seed)
    train_ds = ChestDataset(args.chest_train, args.image_root, label_cols,
                            resolution, train=True, strip_prefix=args.strip_prefix,
                            strong_aug=args.strong_aug)
    val_ds = ChestDataset(args.chest_val, args.image_root, label_cols,
                          resolution, train=False, strip_prefix=args.strip_prefix)
    dl_kw = dict(num_workers=args.num_workers, pin_memory=True,
                 worker_init_fn=seed_worker, generator=g)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          drop_last=True, **dl_kw)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **dl_kw)

    # model / loss / optim
    model = ResNet50MultiLabel(num_classes=len(label_cols), pretrained=True,
                               dropout=args.dropout).to(device)
    freeze_until(model, args.freeze_until)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_train/1e6:.1f}M / {n_total/1e6:.1f}M "
          f"(frozen up to '{args.freeze_until}')")
    pos_weight = compute_pos_weight(args.chest_train, label_cols).to(device)
    loss_fn = build_loss(args.loss, pos_weight=pos_weight).to(device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    start_epoch, best = 0, -1.0
    last_path = os.path.join(args.out_dir, "last.ckpt")
    best_path = os.path.join(args.out_dir, "best.ckpt")
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        if ck["sched"]: sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print(f"Resumed from epoch {start_epoch} (best macro-AUROC {best:.4f})")

    run = None
    if args.wandb:
        import wandb
        project = args.wandb_project or cfg.get("wandb", {}).get("project", "medxai")
        run = wandb.init(project=project, name=f"chest_{args.loss}_r{resolution}",
                         config={**vars(args), "resolution": resolution, "seed": seed})

    epochs_no_improve = 0
    for epoch in range(start_epoch, 1 if args.smoke else args.epochs):
        model.train()
        set_frozen_bn_eval(model, args.freeze_until)  # frozen BN stays in eval
        running = 0.0
        for step, (x, y) in enumerate(train_dl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            if args.smoke and step >= 5:
                break
        sched.step()

        val_loss, macro, per_class = evaluate(model, val_dl, loss_fn, device, label_cols)
        train_loss = running / (step + 1)
        print(f"epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"| macroAUROC {macro:.4f}")
        if run:
            run.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                     "val_macro_auroc": macro,
                     **{f"auroc/{k}": v for k, v in per_class.items()}})

        save_ckpt(last_path, model, opt, sched, scaler, epoch, best, vars(args))
        if macro > best:
            best = macro
            epochs_no_improve = 0
            save_ckpt(best_path, model, opt, sched, scaler, epoch, best, vars(args))
            with open(os.path.join(args.out_dir, "best_metrics.json"), "w") as f:
                json.dump({"epoch": epoch, "macro_auroc": macro,
                           "per_class": per_class}, f, indent=2)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience and not args.smoke:
                print(f"Early stop at epoch {epoch} (no improve {args.patience})")
                break

    if run:
        run.finish()
    print(f"Done. Best macro-AUROC {best:.4f}. Checkpoints in {args.out_dir}")
    if args.smoke:
        print("SMOKE OK")


if __name__ == "__main__":
    main()
