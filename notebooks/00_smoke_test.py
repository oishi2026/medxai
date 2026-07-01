"""00_smoke_test — paste these cells into a Kaggle notebook (GPU on, Internet on).

Goal: prove the loop works END TO END before writing any real code:
  GitHub clone -> pip install -e . -> import medxai -> GPU forward pass -> W&B log.
If you see "LOOP OK" with CUDA True, the whole team's plumbing is verified.
"""

# === CELL 1: get the code (edit the URL/branch) ==============================
# !git clone --branch main https://github.com/<you>/<repo>.git
# %cd <repo>
# !pip install -e . -q
# !pip install -r requirements.txt -q

# === CELL 2: smoke test ======================================================
def main():
    import torch
    from medxai import __version__
    from medxai.utils import set_determinism

    print("medxai version:", __version__)
    print("torch:", torch.__version__)
    cuda = torch.cuda.is_available()
    print("CUDA available:", cuda)
    if cuda:
        print("GPU:", torch.cuda.get_device_name(0))

    set_determinism(0)

    # tiny GPU forward pass to confirm compute actually runs on device
    device = "cuda" if cuda else "cpu"
    conv = torch.nn.Conv2d(3, 8, kernel_size=3).to(device)
    x = torch.randn(2, 3, 32, 32, device=device)
    y = conv(x)
    assert y.shape == (2, 8, 30, 30)
    print("forward pass ok on", device, "->", tuple(y.shape))

    # OPTIONAL: confirm W&B logging works (comment out until `wandb login` set)
    # import wandb
    # wandb.init(project="medxai", mode="online")
    # wandb.log({"smoke/loss": float(y.float().mean())})
    # wandb.finish()

    print("LOOP OK")


if __name__ == "__main__":
    main()
