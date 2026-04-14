"""Unified dataset / dataloader construction."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
from torch.utils.data import DataLoader

from .dataset_audio import SegmentAudioDataset
from .sampler import build_train_sampler


def _collate_fn(batch: list[dict]) -> dict:
    """Stack waveforms and labels, collect metadata."""
    import torch

    waveforms = torch.stack([item["waveform"] for item in batch])
    label_ids = torch.tensor([item["label_id"] for item in batch], dtype=torch.long)
    metadata = [item["metadata"] for item in batch]

    return {
        "waveform": waveforms,
        "label_id": label_ids,
        "metadata": metadata,
    }


def build_dataloader(
    unified_df: pd.DataFrame,
    split: str,
    batch_size: int = 16,
    num_workers: int = 0,
    use_balanced_sampler: bool = False,
    generator: "torch.Generator | None" = None,
    worker_init_fn: Callable[[int], None] | None = None,
) -> DataLoader:
    """Build a DataLoader for the given split.

    Args:
        unified_df: Unified data table.
        split: train / val / test / external_test.
        batch_size: Batch size.
        num_workers: Data loading workers.
        use_balanced_sampler: Class-balanced sampling (train only).
        generator: Seeded generator for reproducible shuffle (train only).
        worker_init_fn: Worker init for reproducible multi-process loading.
    """
    dataset = SegmentAudioDataset(unified_df, split)

    sampler = None
    shuffle = False

    if split == "train":
        if use_balanced_sampler:
            sampler = build_train_sampler(dataset)
        else:
            shuffle = True
    # val/test/external_test: shuffle=False, no sampler

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=_collate_fn,
        pin_memory=False,
        generator=generator if shuffle else None,
        worker_init_fn=worker_init_fn,
    )


def build_all_dataloaders(
    unified_df: pd.DataFrame,
    batch_size: int = 16,
    num_workers: int = 0,
    use_balanced_sampler: bool = False,
    splits: list[str] | None = None,
    generator: "torch.Generator | None" = None,
    worker_init_fn: Callable[[int], None] | None = None,
) -> dict[str, DataLoader]:
    """Build DataLoaders for multiple splits.

    Args:
        splits: List of splits to build. Defaults to all four.
        generator: Seeded generator (train only).
        worker_init_fn: Worker init function.
    """
    if splits is None:
        splits = ["train", "val", "test", "external_test"]

    loaders: dict[str, DataLoader] = {}
    for sp in splits:
        loaders[sp] = build_dataloader(
            unified_df,
            sp,
            batch_size=batch_size,
            num_workers=num_workers,
            use_balanced_sampler=(use_balanced_sampler and sp == "train"),
            generator=generator,
            worker_init_fn=worker_init_fn,
        )
    return loaders
