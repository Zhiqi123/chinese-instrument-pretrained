"""Frozen train subset loading and validation for data efficiency experiments."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_train_subset(
    unified_df: pd.DataFrame,
    subset_segments_csv: Path,
    subset_meta_json: Path,
) -> tuple[pd.DataFrame, dict]:
    """Load a frozen train subset and filter the unified DataFrame.

    Args:
        unified_df: Full unified table from load_seed_data().
        subset_segments_csv: Path to train_subset_segments.csv.
        subset_meta_json: Path to subset_meta.json.

    Returns:
        (filtered_df, subset_meta_dict) — filtered_df has subsetted train +
        unchanged val/test/external_test.

    Raises:
        FileNotFoundError: Subset file missing.
        ValueError: Validation failed.
    """
    subset_segments_csv = Path(subset_segments_csv)
    subset_meta_json = Path(subset_meta_json)

    if not subset_segments_csv.exists():
        raise FileNotFoundError(f"train_subset_segments.csv not found: {subset_segments_csv}")
    if not subset_meta_json.exists():
        raise FileNotFoundError(f"subset_meta.json not found: {subset_meta_json}")

    subset_seg_df = pd.read_csv(subset_segments_csv, dtype=str)
    with open(subset_meta_json) as f:
        subset_meta = json.load(f)

    # Ordered subset segment IDs
    subset_segment_ids = subset_seg_df["segment_id"].tolist()
    subset_segment_id_set = set(subset_segment_ids)

    # Separate train from non-train
    train_mask = unified_df["split"] == "train"
    full_train_df = unified_df[train_mask]
    non_train_df = unified_df[~train_mask]

    # Check 1: all subset IDs exist in full train
    full_train_segment_ids = set(full_train_df["segment_id"])
    missing = subset_segment_id_set - full_train_segment_ids
    if missing:
        raise ValueError(
            f"Subset contains {len(missing)} segment_id(s) not in full train, "
            f"examples: {sorted(missing)[:5]}"
        )

    # Check 2: no duplicate segment IDs
    if len(subset_segment_ids) != len(subset_segment_id_set):
        raise ValueError(
            f"Duplicate segment_ids in train_subset_segments.csv: "
            f"{len(subset_segment_ids)} rows vs {len(subset_segment_id_set)} unique"
        )

    # Filter train and reorder by subset CSV
    # Map segment_id → row index
    subset_order = {sid: idx for idx, sid in enumerate(subset_segment_ids)}
    filtered_train = full_train_df[
        full_train_df["segment_id"].isin(subset_segment_id_set)
    ].copy()
    filtered_train["_subset_order"] = filtered_train["segment_id"].map(subset_order)
    filtered_train = filtered_train.sort_values("_subset_order").drop(columns=["_subset_order"])

    # Check 3: row count matches
    if len(filtered_train) != len(subset_segment_ids):
        raise ValueError(
            f"Filtered train row count ({len(filtered_train)}) != "
            f"subset segment count ({len(subset_segment_ids)})"
        )

    # Check 3b: segment_id order matches CSV
    actual_order = filtered_train["segment_id"].tolist()
    if actual_order != subset_segment_ids:
        for i, (a, e) in enumerate(zip(actual_order, subset_segment_ids)):
            if a != e:
                raise ValueError(
                    f"Filtered train segment_id order mismatch at "
                    f"row {i}: actual={a}, expected={e}"
                )
        raise ValueError("Filtered train segment_id order mismatch with CSV")

    # Check 4: non-train splits unchanged
    for split_name in ["val", "test", "external_test"]:
        orig_count = (unified_df["split"] == split_name).sum()
        filtered_count = (non_train_df["split"] == split_name).sum()
        if orig_count != filtered_count:
            raise ValueError(
                f"{split_name} row count changed: original {orig_count} → filtered {filtered_count}"
            )

    # Check 5: subset only contains train
    subset_splits = subset_seg_df["split"].unique()
    if not (len(subset_splits) == 1 and subset_splits[0] == "train"):
        raise ValueError(
            f"train_subset_segments.csv contains non-train splits: {subset_splits}"
        )

    # Merge subsetted train + full non-train
    filtered_df = pd.concat(
        [filtered_train, non_train_df],
        ignore_index=True,
    )

    return filtered_df, subset_meta
