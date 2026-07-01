"""Determinism utilities.

Call `set_determinism(seed)` at the top of EVERY entrypoint (training, eval,
graph build). For DataLoaders, pass `worker_init_fn=seed_worker` and
`generator=make_generator(seed)` so workers are reproducible too.

Note: full determinism trades a little speed. That is the correct trade for a
benchmark whose numbers must be reproducible across seeds.
"""
from __future__ import annotations


def set_determinism(seed: int) -> None:
    import os
    import random

    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn for reproducible augmentation/shuffling."""
    import random

    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int):
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    return g
