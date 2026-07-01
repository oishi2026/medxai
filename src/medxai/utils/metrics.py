"""Metrics. Macro-AUROC is the model-selection metric for chest: averaging over
classes makes it insensitive to the small val-split prevalence drift we observed,
so checkpoint selection isn't skewed by a couple of points of class imbalance.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def macro_auroc(targets: np.ndarray, probs: np.ndarray, label_cols: Sequence[str]):
    """Returns (macro_auroc, {class: auroc}). Classes with one label present in
    the batch get NaN and are excluded from the macro average."""
    per_class = {}
    for i, c in enumerate(label_cols):
        y = targets[:, i]
        if y.min() == y.max():           # only one class present -> undefined
            per_class[c] = float("nan")
        else:
            per_class[c] = float(roc_auc_score(y, probs[:, i]))
    valid = [v for v in per_class.values() if not math.isnan(v)]
    macro = float(np.mean(valid)) if valid else float("nan")
    return macro, per_class


def macro_auprc(targets: np.ndarray, probs: np.ndarray, label_cols: Sequence[str]):
    """Average precision, the better summary for very rare classes (e.g. derm)."""
    per_class = {}
    for i, c in enumerate(label_cols):
        y = targets[:, i]
        if y.max() == 0:
            per_class[c] = float("nan")
        else:
            per_class[c] = float(average_precision_score(y, probs[:, i]))
    valid = [v for v in per_class.values() if not math.isnan(v)]
    macro = float(np.mean(valid)) if valid else float("nan")
    return macro, per_class
