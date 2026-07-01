# Pre-registration (dated, committed before results exist)

**Date:** TODO

## Hypotheses
- **H1 (Faithfulness):** CNN->GNN-attention improves region-level insertion/deletion
  AUC and pointing/IoU vs. CNN+XAI, without degrading AUROC.
- **H2 (Right-reasons):** Adding the RRR regulariser reduces reliance on
  non-anatomical regions.

## Test (fixed before running)
- Paired **Wilcoxon signed-rank** across test images on per-image faithfulness scores.
- Multiple-comparison correction: **Benjamini-Hochberg** across metrics x datasets.
- Bootstrap CIs for AUROC/AUPRC; effect sizes reported alongside p-values.

## Sign-off bar (copied verbatim from proposal)
Pass sanity checks; show statistically significant gains on at least one dataset for H1.
