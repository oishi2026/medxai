"""Train the region-graph GNN on cached graphs.

Mirrors the CNN backbone trainer (same 10-class multi-label task, same weighted
BCE / focal option, same macro-AUROC checkpoint selection) so the two paradigms
are compared on equal footing. Free-tier essentials included: AMP, checkpoint +
resume, W&B, and --smoke.

Example (Kaggle):
    python -m medxai.graph.train_gnn \
        --graph_dir /kaggle/input/medxai-chest-graphs-v1 \
        --out_dir   /kaggle/working/outputs/gnn \
        --conv gat --smoke
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from medxai.backbones.losses import build_loss
from medxai.data.splits import CHEXLOCALIZE_PATHOLOGIES
from medxai.graph.dataset import CachedGraphDataset
from medxai.graph.gnn import RegionGNN
from medxai.utils.determinism import set_determinism
from medxai.utils.metrics import macro_auroc


def _pos_weight_from_graphs(ds, num_classes, clamp_max=10.0):
    ys = torch.cat([ds[i].y for i in range(len(ds))], dim=0)  # (N, C)
    pos = ys.sum(0).clamp(min=1)
    neg = ys.shape[0] - pos
    return (neg / pos).clamp(max=clamp_max)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, label_cols):
    model.eval()
    tp, pr, losses = [], [], []
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(batch)
            losses.append(loss_fn(logits, batch.y).item())
        pr.append(torch.sigmoid(logits).float().cpu().numpy())
        tp.append(batch.y.cpu().numpy())
    t, p = np.concatenate(tp), np.concatenate(pr)
    macro, per_class = macro_auroc(t, p, label_cols)
    return float(np.mean(losses)), macro, per_class


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_dir", required=True,
                    help="dir containing train/ and val/ graph caches")
    ap.add_argument("--out_dir", default="/kaggle/working/outputs/gnn")
    ap.add_argument("--frozen", default="conf/frozen.yaml")
    ap.add_argument("--conv", choices=["gat", "sage"], default="gat")
    ap.add_argument("--loss", choices=["bce", "focal"], default="bce")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.frozen))
    seed = cfg["seeds"][0]
    set_determinism(seed)
    label_cols = CHEXLOCALIZE_PATHOLOGIES
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    from torch_geometric.loader import DataLoader as GeoLoader
    train_ds = CachedGraphDataset(os.path.join(args.graph_dir, "train"))
    val_ds = CachedGraphDataset(os.path.join(args.graph_dir, "val"))
    train_dl = GeoLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = GeoLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    in_dim = train_ds[0].x.shape[1]
    print(f"graphs: {len(train_ds)} train / {len(val_ds)} val | in_dim {in_dim}")

    model = RegionGNN(in_dim=in_dim, hidden=args.hidden, num_classes=len(label_cols),
                      num_layers=args.num_layers, conv=args.conv, heads=args.heads,
                      dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.conv.upper()} GNN | {n_params/1e6:.2f}M params")

    pos_weight = _pos_weight_from_graphs(train_ds, len(label_cols)).to(device)
    loss_fn = build_loss(args.loss, pos_weight=pos_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    start_epoch, best = 0, -1.0
    last_path = os.path.join(args.out_dir, "last.ckpt")
    best_path = os.path.join(args.out_dir, "best.ckpt")
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        if ck.get("sched"): sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print(f"Resumed epoch {start_epoch} (best {best:.4f})")

    run = None
    if args.wandb:
        import wandb
        project = args.wandb_project or cfg.get("wandb", {}).get("project", "medxai")
        run = wandb.init(project=project, name=f"gnn_{args.conv}_{args.loss}",
                         config={**vars(args), "seed": seed, "params_M": n_params/1e6})

    no_improve = 0
    for epoch in range(start_epoch, 1 if args.smoke else args.epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_dl):
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(batch)
                loss = loss_fn(logits, batch.y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
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

        ck = {"model": model.state_dict(), "opt": opt.state_dict(),
              "sched": sched.state_dict(), "scaler": scaler.state_dict(),
              "epoch": epoch, "best": best, "cfg": vars(args)}
        torch.save(ck, last_path)
        if macro > best:
            best = macro; no_improve = 0
            ck["best"] = best; torch.save(ck, best_path)
            with open(os.path.join(args.out_dir, "best_metrics.json"), "w") as f:
                json.dump({"epoch": epoch, "macro_auroc": macro,
                           "per_class": per_class, "conv": args.conv}, f, indent=2)
        else:
            no_improve += 1
            if no_improve >= args.patience and not args.smoke:
                print(f"Early stop at epoch {epoch}")
                break

    if run:
        run.finish()
    print(f"Done. Best macro-AUROC {best:.4f}. Checkpoints in {args.out_dir}")
    if args.smoke:
        print("SMOKE OK")


if __name__ == "__main__":
    main()
