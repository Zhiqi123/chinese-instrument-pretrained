"""Seed management and device selection."""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch


def set_global_seed(base_seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)


def get_device(device_cfg: str = "auto") -> torch.device:
    """Resolve device string. auto → cuda > mps > cpu."""
    if device_cfg != "auto":
        return torch.device(device_cfg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_train_generator(base_seed: int) -> torch.Generator:
    """Create a seeded Generator for train DataLoader shuffle."""
    return torch.Generator().manual_seed(base_seed)


class _WorkerInitFn:
    """Picklable worker_init_fn (Python 3.13+ spawn compatible)."""

    def __init__(self, base_seed: int):
        self.base_seed = base_seed

    def __call__(self, worker_id: int) -> None:
        np.random.seed(self.base_seed + worker_id)


def build_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    """Create worker_init_fn for reproducible multi-process loading."""
    return _WorkerInitFn(base_seed)
