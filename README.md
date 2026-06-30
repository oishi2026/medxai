# medxai

Transparent explanations in medical imaging: CNN baselines vs. region-graph GNNs
with attention, compared under a shared, faithfulness-first protocol.

**This GitHub repo is the source of truth for all code.** Kaggle notebooks are
disposable launchers that clone this repo and run GPU jobs. Data and checkpoints
live in shared **Kaggle Datasets**, never in git.

## Run on Kaggle (every teammate, every session)
GPU on, Internet on. First cells of your launcher notebook:
```bash
git clone --branch <your-branch> https://github.com/<you>/<repo>.git
cd <repo>
pip install -e . -q
pip install -r requirements.txt -q
```
Then run `notebooks/00_smoke_test.py`. Success = "LOOP OK" + CUDA True.

## Collaboration rules
- **Code:** GitHub only. Feature branch -> pull request -> review -> merge to `main`.
  Never co-edit code through Kaggle (no merge safety).
- **`conf/frozen.yaml`:** changing it requires a reviewed PR. It is the binding
  config every result is produced under.
- **Each teammate uses their OWN launcher notebook** (all clone the same repo).
  Never two people running the same Kaggle notebook at once.
- **Experiment tracking:** one shared W&B project (`conf/config.yaml: wandb.entity`).
- **GPU quota is per-user (~30h/week each), not pooled** — parallelise independent work.

## Ownership map (Week 1 — fill in names)
- Chest arm (CheXpert backbone + XAI + CheXlocalize localization): TODO
- Dermatology arm (ISIC backbone + XAI): TODO
- Shared graph/GNN pipeline + faithfulness-evaluation harness: TODO
  (single-owned so the protocol is identical across both arms)

## Layout
```
conf/      Hydra configs; frozen.yaml = binding decisions
src/medxai backbones/ xai/ graph/ eval/ + utils (determinism)
notebooks/ thin Kaggle launchers that import src
data/      gitignored; real data is in Kaggle Datasets
outputs/   gitignored; checkpoints saved out as Kaggle Datasets
PREREG.md  hypotheses + stats plan, committed before results exist
```
