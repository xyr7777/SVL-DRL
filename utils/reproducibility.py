"""Reproducibility helpers for SVL-DRL training.

These utilities seed Python, NumPy and PyTorch without changing the model or
optimization algorithm.
"""
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch in the current process.

    In distributed training, call this after assigning rank and pass a
    rank-specific seed (for example base_seed + rank).
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Determinism can reduce performance; some CUDA operations may not
        # have deterministic implementations.
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy inside each DataLoader worker."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
