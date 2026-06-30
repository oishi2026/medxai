"""Leakage-free, reproducible, stratified split generation.

Design choices that matter for a Q1 benchmark and are easy to get wrong:

1. PATIENT GROUPING. A patient with several images must land entirely in one
   split. We aggregate labels to patient level, stratify *patients*, then expand
   back to images. This makes leakage structurally impossible.

2. MULTI-LABEL STRATIFICATION. Chest images carry several pathologies at once;
   naive stratification breaks. We use the Sechidis et al. (2011) iterative
   stratification algorithm so every pathology's prevalence survives subsampling.
   Implemented here directly (no external dependency) so the install can't break
   it and the result is fully reproducible.

3. REPRODUCIBLE MANIFESTS. Rows are sorted deterministically and hashed
   (SHA-256), so "re-run from the seed -> identical checksum" is a real gate.

The same functions serve both arms: chest is genuinely multi-label; ISIC is the
single-label special case (label matrix with one column).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# The 10 pathologies with CheXlocalize segmentations (so every scored class is
# both trainable and localizable). Override via config if you want a subset.
CHEXLOCALIZE_PATHOLOGIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pneumothorax",
    "Support Devices",
]

# CheXpert paper found U-Ones best for these; default per-class policy.
DEFAULT_PER_CLASS_UONES = {"Atelectasis", "Edema"}

_PATIENT_RE = re.compile(r"(patient\d+)")


# --------------------------------------------------------------------------- #
# CheXpert helpers
# --------------------------------------------------------------------------- #
def extract_patient_id_chexpert(path: str) -> str:
    """Pull 'patientXXXXX' out of a CheXpert image path."""
    m = _PATIENT_RE.search(str(path))
    if not m:
        raise ValueError(f"No patient id found in path: {path!r}")
    return m.group(1)


def apply_uncertainty_policy(
    df: pd.DataFrame,
    label_cols: Sequence[str],
    policy: str,
    per_class_uones: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Map CheXpert labels to {0,1}. Blank/NaN -> 0; -1 (uncertain) per policy."""
    if policy not in {"u_zeros", "u_ones", "per_class"}:
        raise ValueError(f"Unknown uncertainty policy: {policy!r}")
    per_class_uones = set(per_class_uones or DEFAULT_PER_CLASS_UONES)
    out = df.copy()
    for c in label_cols:
        col = out[c].fillna(0.0)
        if policy == "u_zeros":
            col = col.replace(-1.0, 0.0)
        elif policy == "u_ones":
            col = col.replace(-1.0, 1.0)
        else:  # per_class
            col = col.replace(-1.0, 1.0 if c in per_class_uones else 0.0)
        out[c] = col.astype(int)
    return out


# --------------------------------------------------------------------------- #
# Iterative stratification (Sechidis, Tsoumakas, Vlahavas 2011)
# --------------------------------------------------------------------------- #
def iterative_stratification(
    Y: np.ndarray, proportions: Sequence[float], seed: int = 0
) -> np.ndarray:
    """Assign each row of binary label matrix Y to one of len(proportions) splits.

    Greedily balances per-label counts across splits. Deterministic given seed
    (the seed only breaks exact ties). Returns an int array of split indices.
    """
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, dtype=int)
    n, L = Y.shape
    props = np.asarray(proportions, dtype=float)
    props = props / props.sum()

    c = props * n                              # desired remaining per split
    label_totals = Y.sum(axis=0).astype(float)
    cj = np.outer(props, label_totals)         # desired per split, per label (k,L)
    label_remaining = Y.sum(axis=0).astype(int)
    samples_of_label = [np.where(Y[:, j] == 1)[0] for j in range(L)]

    assignment = np.full(n, -1, dtype=int)
    assigned = np.zeros(n, dtype=bool)

    while True:
        pos = np.where(label_remaining > 0)[0]
        if len(pos) == 0:
            break
        # label with fewest remaining positives; tie -> smallest total
        counts = label_remaining[pos]
        cand = pos[counts == counts.min()]
        lbl = int(cand[np.argmin(label_totals[cand])])

        for i in samples_of_label[lbl]:
            if assigned[i]:
                continue
            col = cj[:, lbl]
            sc = np.where(col == col.max())[0]            # most-desired split(s)
            if len(sc) > 1:                                # tie -> largest overall
                cvals = c[sc]
                sc = sc[cvals == cvals.max()]
            split = int(sc[0]) if len(sc) == 1 else int(rng.choice(sc))

            assignment[i] = split
            assigned[i] = True
            cj[split] -= Y[i]
            c[split] -= 1
            for j in np.where(Y[i] == 1)[0]:
                label_remaining[j] -= 1

    # rows with no positive label (e.g. all-negative / "No Finding") -> by count
    for i in np.where(~assigned)[0]:
        split = int(np.argmax(c))
        assignment[i] = split
        c[split] -= 1
    return assignment


# --------------------------------------------------------------------------- #
# Grouped splitting
# --------------------------------------------------------------------------- #
def aggregate_to_patient(
    df: pd.DataFrame, group_col: str, label_cols: Sequence[str]
) -> pd.DataFrame:
    """Patient-level label = max over their images (present if any image positive)."""
    return df.groupby(group_col)[list(label_cols)].max()


def make_grouped_splits(
    df: pd.DataFrame,
    group_col: str,
    label_cols: Sequence[str],
    split_names: Sequence[str],
    proportions: Sequence[float],
    seed: int,
    split_col: str = "split",
) -> pd.DataFrame:
    """Stratified, patient-grouped split. A 'drop' name in split_names is the
    discard bucket used to subsample (it is filtered out by the caller)."""
    patient_labels = aggregate_to_patient(df, group_col, label_cols)
    assign = iterative_stratification(patient_labels.values, proportions, seed)
    mapping = {
        pid: split_names[a]
        for pid, a in zip(patient_labels.index.to_numpy(), assign)
    }
    out = df.copy()
    out[split_col] = out[group_col].map(mapping)
    return out


# --------------------------------------------------------------------------- #
# Verification + manifests
# --------------------------------------------------------------------------- #
def assert_no_group_leakage(
    df: pd.DataFrame, group_col: str, split_col: str = "split"
) -> None:
    counts = df.groupby(group_col)[split_col].nunique()
    leaked = counts[counts > 1]
    assert leaked.empty, f"{len(leaked)} {group_col}(s) leak across splits"


def prevalence_table(
    df: pd.DataFrame, label_cols: Sequence[str], split_col: str = "split"
) -> pd.DataFrame:
    return df.groupby(split_col)[list(label_cols)].mean()


def prevalence_drift(
    df: pd.DataFrame,
    label_cols: Sequence[str],
    split_col: str = "split",
    ignore_splits: Sequence[str] = ("drop",),
) -> pd.Series:
    """Per-label max |split prevalence - overall prevalence|, descending.

    Note: splits are stratified at the *group* (patient) level but prevalence is
    measured at the *image* level, so common classes drift slightly more — this
    is inherent to leakage-free grouped splitting, not a fault.
    """
    kept = df[~df[split_col].isin(ignore_splits)]
    overall = kept[list(label_cols)].mean()
    per_split = kept.groupby(split_col)[list(label_cols)].mean()
    return (per_split - overall).abs().max(axis=0).sort_values(ascending=False)


def assert_prevalence_fidelity(
    df: pd.DataFrame,
    label_cols: Sequence[str],
    split_col: str = "split",
    tol: float = 0.05,
    ignore_splits: Sequence[str] = ("drop",),
) -> None:
    """Each kept split's per-label prevalence within `tol` of the overall."""
    kept = df[~df[split_col].isin(ignore_splits)]
    overall = kept[list(label_cols)].mean()
    per_split = kept.groupby(split_col)[list(label_cols)].mean()
    diff = (per_split - overall).abs()
    worst = diff.max().max()
    assert worst <= tol, (
        f"Prevalence drift {worst:.3f} exceeds tol {tol} "
        f"(class: {diff.max().idxmax()})"
    )


def sha256_of_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    df: pd.DataFrame,
    out_dir: str | Path,
    name: str,
    sort_keys: Sequence[str],
) -> tuple[Path, str]:
    """Write a split manifest with deterministic row order, return (path, sha256)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    df_sorted = df.sort_values(list(sort_keys)).reset_index(drop=True)
    df_sorted.to_csv(path, index=False, lineterminator="\n")
    return path, sha256_of_file(path)
