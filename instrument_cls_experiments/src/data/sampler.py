"""Train-only class-balanced sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import WeightedRandomSampler

if TYPE_CHECKING:
    from .dataset_audio import SegmentAudioDataset


def build_train_sampler(
    dataset: SegmentAudioDataset,
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler using inverse class frequency."""
    labels = dataset.df["label_id"].astype(int).values
    class_counts = torch.bincount(torch.tensor(labels))
    # Avoid division by zero
    class_weights = 1.0 / class_counts.float().clamp(min=1)
    sample_weights = class_weights[torch.tensor(labels)]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
