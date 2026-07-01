"""Build reproducible train/val(/test) manifests for both arms.

Chest (CheXpert-small): carve train/val out of CheXpert `train.csv`; CheXlocalize
stays untouched as the held-out localization/test set.
Derm (ISIC 2020): carve train/val/test (no external localization set exists).

Run on Kaggle after the loop is green, e.g.:

    python -m medxai.data.build_splits \
        --chexpert_csv /kaggle/input/<chexpert>/train.csv \
        --isic_csv     /kaggle/input/<isic>/train.csv \
        --out_dir      /kaggle/working/splits

Seed and uncertainty policy are read from conf/frozen.yaml so the binding
decisions live in one place.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from medxai.data.splits import (
    CHEXLOCALIZE_PATHOLOGIES,
    apply_uncertainty_policy,
    assert_no_group_leakage,
    extract_patient_id_chexpert,
    make_grouped_splits,
    prevalence_drift,
    prevalence_table,
    write_manifest,
)


def _report_fidelity(arm: str, drift: pd.Series, soft: float, hard: float) -> None:
    """Print per-class drift; warn above soft tol; hard-fail above hard tol."""
    print(f"[{arm}] per-class prevalence drift (kept splits vs overall):")
    for label, v in drift.items():
        flag = "FAIL" if v > hard else ("warn" if v > soft else "ok")
        print(f"    {label:<32s} {v:.4f}  {flag}")
    worst, cls = float(drift.max()), str(drift.idxmax())
    if worst > hard:
        raise AssertionError(
            f"[{arm}] drift {worst:.3f} on '{cls}' exceeds hard tol {hard} — "
            f"investigate (possible leakage or label bug)."
        )
    if worst > soft:
        print(
            f"[{arm}] NOTE: max drift {worst:.3f} on '{cls}' is above soft tol "
            f"{soft} but below hard tol {hard}. Acceptable for model-selection "
            f"splits — common classes drift most under grouped stratification."
        )


def _load_frozen(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    pol = cfg["decisions"]["chexpert_uncertain_policy"]
    if pol == "TODO":
        raise SystemExit(
            "Set decisions.chexpert_uncertain_policy in conf/frozen.yaml "
            "(recommended: u_zeros) before building splits."
        )
    return cfg


def build_chest(args, cfg) -> dict:
    df = pd.read_csv(args.chexpert_csv)
    if args.frontal_only and "Frontal/Lateral" in df.columns:
        df = df[df["Frontal/Lateral"] == "Frontal"].copy()

    label_cols = CHEXLOCALIZE_PATHOLOGIES
    df = apply_uncertainty_policy(
        df, label_cols, cfg["decisions"]["chexpert_uncertain_policy"]
    )
    path_col = "Path" if "Path" in df.columns else df.columns[0]
    df["patient_id"] = df[path_col].map(extract_patient_id_chexpert)

    # one pass: train / val / drop  (drop = subsample remainder)
    f = args.chest_subset_frac
    v = args.val_frac
    df = make_grouped_splits(
        df,
        group_col="patient_id",
        label_cols=label_cols,
        split_names=["train", "val", "drop"],
        proportions=[f * (1 - v), f * v, 1 - f],
        seed=cfg["seeds"][0],
    )
    kept = df[df["split"] != "drop"].copy()

    assert_no_group_leakage(kept, "patient_id")
    _report_fidelity("chest", prevalence_drift(kept, label_cols),
                     soft=args.tol, hard=args.hard_tol)

    sort_keys = [path_col]
    manifests = {}
    for name in ["train", "val"]:
        sub = kept[kept["split"] == name]
        p, sha = write_manifest(sub, args.out_dir, f"chest_{name}", sort_keys)
        manifests[f"chest_{name}"] = {"path": str(p), "sha256": sha, "n": len(sub)}
    return {
        "manifests": manifests,
        "prevalence": prevalence_table(kept, label_cols).round(4).to_dict(),
        "label_cols": label_cols,
        "note": "localization/test = CheXlocalize (CheXpert val/test), held out",
    }


def build_derm(args, cfg) -> dict:
    df = pd.read_csv(args.isic_csv)
    if "patient_id" not in df.columns or "target" not in df.columns:
        raise SystemExit("ISIC csv must have 'patient_id' and 'target' columns.")

    label_cols = ["target"]
    f = args.derm_subset_frac
    df = make_grouped_splits(
        df,
        group_col="patient_id",
        label_cols=label_cols,
        split_names=["train", "val", "test", "drop"],
        proportions=[f * 0.70, f * 0.15, f * 0.15, 1 - f],
        seed=cfg["seeds"][0],
    )
    kept = df[df["split"] != "drop"].copy()

    assert_no_group_leakage(kept, "patient_id")
    _report_fidelity("derm", prevalence_drift(kept, label_cols),
                     soft=args.tol, hard=args.hard_tol)

    id_col = "image_name" if "image_name" in df.columns else df.columns[0]
    manifests = {}
    for name in ["train", "val", "test"]:
        sub = kept[kept["split"] == name]
        p, sha = write_manifest(sub, args.out_dir, f"derm_{name}", [id_col])
        manifests[f"derm_{name}"] = {"path": str(p), "sha256": sha, "n": len(sub)}
    return {
        "manifests": manifests,
        "prevalence": prevalence_table(kept, label_cols).round(4).to_dict(),
        "label_cols": label_cols,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chexpert_csv", required=True)
    ap.add_argument("--isic_csv", required=True)
    ap.add_argument("--out_dir", default="/kaggle/working/splits")
    ap.add_argument("--frozen", default="conf/frozen.yaml")
    ap.add_argument("--chest_subset_frac", type=float, default=0.15)
    ap.add_argument("--derm_subset_frac", type=float, default=0.20)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--frontal_only", action="store_true", default=True)
    ap.add_argument("--tol", type=float, default=0.05,
                    help="soft per-class drift tol: warn above this")
    ap.add_argument("--hard_tol", type=float, default=0.10,
                    help="hard per-class drift tol: fail above this")
    args = ap.parse_args()

    cfg = _load_frozen(args.frozen)
    meta = {
        "date": str(date.today()),
        "seed": cfg["seeds"][0],
        "uncertainty_policy": cfg["decisions"]["chexpert_uncertain_policy"],
        "chest_subset_frac": args.chest_subset_frac,
        "derm_subset_frac": args.derm_subset_frac,
        "chest": build_chest(args, cfg),
        "derm": build_derm(args, cfg),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(json.dumps(meta, indent=2, default=str))
    print("\nSplits written to", out)


if __name__ == "__main__":
    main()
