"""Audio segment Dataset.

Returns per sample:
    waveform: float32 [num_samples], mono
    label_id: int
    metadata: dict (segment_id, sample_id, record_id, split, ...)
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset
import pandas as pd


class SegmentAudioDataset(Dataset):
    """Audio Dataset backed by the unified manifest table.

    Args:
        unified_df: DataFrame from manifest_loader.load_seed_data()
        split: Split name (train / val / test / external_test)
    """

    def __init__(self, unified_df: pd.DataFrame, split: str):
        self.df = unified_df[unified_df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No segments found for split='{split}'")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        abs_path = row["segment_abs_path"]
        waveform_np, sr = sf.read(abs_path, dtype="float32")

        # Ensure mono
        if waveform_np.ndim > 1:
            waveform_np = waveform_np[:, 0]

        waveform = torch.from_numpy(waveform_np)  # [num_samples]

        return {
            "waveform": waveform,
            "label_id": int(row["label_id"]),
            "metadata": {
                "segment_id": row["segment_id"],
                "sample_id": row["sample_id"],
                "record_id": row["record_id"],
                "split": row["split"],
                "source_dataset": row["source_dataset"],
                "family_label": row["family_label"],
                "segment_abs_path": abs_path,
                "is_padded": row["is_padded"],
            },
        }
