"""Validate the split logic on synthetic data before it touches real datasets."""
import numpy as np
import pandas as pd

from medxai.data.splits import (
    apply_uncertainty_policy,
    assert_no_group_leakage,
    assert_prevalence_fidelity,
    iterative_stratification,
    make_grouped_splits,
    write_manifest,
)


def _synthetic_multilabel(n_patients=600, n_labels=10, seed=0):
    """Patients with 1-4 images each; correlated, imbalanced multi-labels."""
    rng = np.random.default_rng(seed)
    base_rates = rng.uniform(0.03, 0.30, size=n_labels)
    rows = []
    for p in range(n_patients):
        pid = f"patient{p:05d}"
        patient_pos = (rng.random(n_labels) < base_rates).astype(int)
        for k in range(rng.integers(1, 5)):
            img = patient_pos.copy()
            # small per-image noise, but patient signal dominates (=> grouping matters)
            flip = rng.random(n_labels) < 0.05
            img = np.where(flip, 1 - img, img)
            rows.append([f"{pid}/img{k}", pid, *img])
    cols = ["path", "patient_id"] + [f"L{j}" for j in range(n_labels)]
    return pd.DataFrame(rows, columns=cols)


def test_subsetting_integrity():
    """15% subset: no leakage and the kept fraction is close to requested.
    (Per-class fidelity of a tiny val split is checked separately at scale.)"""
    df = _synthetic_multilabel()
    labels = [c for c in df.columns if c.startswith("L")]
    out = make_grouped_splits(
        df, "patient_id", labels,
        split_names=["train", "val", "drop"],
        proportions=[0.15 * 0.85, 0.15 * 0.15, 0.85],
        seed=0,
    )
    assert_no_group_leakage(out, "patient_id")
    frac = (out["split"] != "drop").mean()
    assert 0.10 < frac < 0.20, frac


def test_stratification_fidelity_at_scale():
    """At realistic patient counts, per-label prevalence is preserved AND the
    iterative stratification beats a random grouped split."""
    df = _synthetic_multilabel(n_patients=3000)
    labels = [c for c in df.columns if c.startswith("L")]
    strat = make_grouped_splits(
        df, "patient_id", labels,
        split_names=["train", "val", "test"], proportions=[0.7, 0.15, 0.15], seed=0,
    )
    assert_no_group_leakage(strat, "patient_id")
    assert_prevalence_fidelity(strat, labels, tol=0.03)

    def max_drift(out):
        ov = out[labels].mean()
        return (out.groupby("split")[labels].mean() - ov).abs().max().max()

    rng = np.random.default_rng(0)
    pids = df["patient_id"].drop_duplicates().to_numpy()
    rng.shuffle(pids)
    n = len(pids)
    m = {p: ("train" if i < 0.7 * n else "val" if i < 0.85 * n else "test")
         for i, p in enumerate(pids)}
    rand = df.assign(split=df["patient_id"].map(m))
    assert max_drift(strat) < max_drift(rand)


def test_reproducible_checksum(tmp_path):
    df = _synthetic_multilabel()
    labels = [c for c in df.columns if c.startswith("L")]
    kw = dict(
        group_col="patient_id", label_cols=labels,
        split_names=["train", "val"], proportions=[0.8, 0.2], seed=7,
    )
    a = make_grouped_splits(df, **kw)
    b = make_grouped_splits(df, **kw)
    _, sha_a = write_manifest(a, tmp_path / "a", "m", ["path"])
    _, sha_b = write_manifest(b, tmp_path / "b", "m", ["path"])
    assert sha_a == sha_b                                # deterministic


def test_uncertainty_policies():
    df = pd.DataFrame({"Atelectasis": [1.0, -1.0, 0.0, np.nan],
                       "Cardiomegaly": [-1.0, 0.0, 1.0, np.nan]})
    cols = ["Atelectasis", "Cardiomegaly"]
    z = apply_uncertainty_policy(df, cols, "u_zeros")
    assert z["Atelectasis"].tolist() == [1, 0, 0, 0]
    o = apply_uncertainty_policy(df, cols, "u_ones")
    assert o["Cardiomegaly"].tolist() == [1, 0, 1, 0]
    pc = apply_uncertainty_policy(df, cols, "per_class")  # Atelectasis is U-Ones
    assert pc["Atelectasis"].tolist() == [1, 1, 0, 0]
    assert pc["Cardiomegaly"].tolist() == [0, 0, 1, 0]


def test_single_label_binary_split():
    """ISIC-style: one rare label, grouped by patient."""
    rng = np.random.default_rng(1)
    rows = []
    for p in range(400):
        pid = f"p{p:04d}"
        pos = int(rng.random() < 0.05)
        for _ in range(rng.integers(1, 6)):
            rows.append([f"{pid}_i{_}", pid, pos])
    df = pd.DataFrame(rows, columns=["image_name", "patient_id", "target"])
    out = make_grouped_splits(
        df, "patient_id", ["target"],
        split_names=["train", "val", "test"], proportions=[0.7, 0.15, 0.15], seed=3,
    )
    assert_no_group_leakage(out, "patient_id")
    # every split should contain at least some positives despite rarity
    pos_per_split = out.groupby("split")["target"].sum()
    assert (pos_per_split > 0).all(), pos_per_split.to_dict()
