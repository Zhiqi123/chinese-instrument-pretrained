"""Manifest loading, joining, and unified table construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .label_map import load_label_map

SPLIT_MANIFEST_REQUIRED = {
    "record_id",
    "sample_id",
    "source_dataset",
    "family_label",
    "split_seed",
    "split",
    "recording_group_id",
}

SEGMENT_MANIFEST_REQUIRED = {
    "record_id",
    "sample_id",
    "source_dataset",
    "family_label",
    "split_seed",
    "split",
    "segment_id",
    "segment_index",
    "start_time_sec",
    "end_time_sec",
    "segment_path",
    "is_padded",
}

# Fields that must be consistent after join
JOIN_CONSISTENCY_FIELDS = {"record_id", "source_dataset", "family_label"}

# Join keys
JOIN_KEYS = ["sample_id", "split_seed", "split"]


def load_dataset_config(
    dataset_config_path: str | Path,
) -> dict[str, Any]:
    """Load dataset_v1.yaml."""
    with open(dataset_config_path) as f:
        return yaml.safe_load(f)


def load_seed_config(seed_config_path: str | Path) -> dict[str, Any]:
    """Load seed config YAML."""
    with open(seed_config_path) as f:
        return yaml.safe_load(f)


def _check_required_columns(
    df: pd.DataFrame, required: set[str], manifest_name: str
) -> list[str]:
    """Return list of missing required columns."""
    missing = required - set(df.columns)
    return sorted(missing)


def load_manifests(
    seed_cfg: dict[str, Any],
    project_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw split_manifest and segment_manifest CSVs.

    Returns:
        (split_df, segment_df)
    """
    root = Path(project_root)

    split_path = root / seed_cfg["split_manifest"]
    segment_path = root / seed_cfg["segment_manifest"]

    split_df = pd.read_csv(split_path, dtype=str)
    segment_df = pd.read_csv(segment_path, dtype=str)

    # Field completeness check
    split_missing = _check_required_columns(
        split_df, SPLIT_MANIFEST_REQUIRED, "split_manifest"
    )
    if split_missing:
        raise ValueError(
            f"split_manifest missing required columns: {split_missing}"
        )

    seg_missing = _check_required_columns(
        segment_df, SEGMENT_MANIFEST_REQUIRED, "segment_manifest"
    )
    if seg_missing:
        raise ValueError(
            f"segment_manifest missing required columns: {seg_missing}"
        )

    return split_df, segment_df


def join_manifests(
    split_df: pd.DataFrame,
    segment_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join segment_manifest with split_manifest and validate consistency."""
    # Columns to carry from split_df
    split_carry = [
        c for c in split_df.columns
        if c not in set(segment_df.columns) and c not in set(JOIN_KEYS)
    ]
    split_subset_cols = JOIN_KEYS + list(JOIN_CONSISTENCY_FIELDS) + split_carry
    split_subset_cols = list(dict.fromkeys(split_subset_cols))

    split_for_join = split_df[split_subset_cols].copy()

    merged = segment_df.merge(
        split_for_join,
        on=JOIN_KEYS,
        how="left",
        suffixes=("", "_split"),
        validate="many_to_one",
    )

    # Validate no unmatched segments
    join_null_mask = merged["recording_group_id"].isna()
    if join_null_mask.any():
        n_missing = join_null_mask.sum()
        sample_ids = merged.loc[join_null_mask, "segment_id"].head(5).tolist()
        raise ValueError(
            f"join produced {n_missing} unmatched segments. "
            f"Sample segment_ids: {sample_ids}"
        )

    # Validate consistency fields
    for field in JOIN_CONSISTENCY_FIELDS:
        col_seg = field
        col_split = f"{field}_split"
        if col_split in merged.columns:
            mismatch = merged[col_seg] != merged[col_split]
            if mismatch.any():
                n = mismatch.sum()
                examples = merged.loc[
                    mismatch, ["segment_id", col_seg, col_split]
                ].head(5)
                raise ValueError(
                    f"Field '{field}' mismatch after join ({n} rows). "
                    f"Examples:\n{examples.to_string()}"
                )
            merged.drop(columns=[col_split], inplace=True)

    return merged


def build_unified_table(
    merged_df: pd.DataFrame,
    label_map: dict[str, Any],
    dataset_cfg: dict[str, Any],
    seed_cfg: dict[str, Any],
    project_root: str | Path,
) -> pd.DataFrame:
    """Inject label_id and segment_abs_path into the joined DataFrame.

    Returns:
        Unified table with all required fields.
    """
    root = Path(project_root)
    df = merged_df.copy()

    # Map label_id
    l2id = label_map["label_to_id"]
    unknown_labels = set(df["family_label"].unique()) - set(l2id.keys())
    if unknown_labels:
        raise ValueError(f"Unknown family_labels: {unknown_labels}")
    df["label_id"] = df["family_label"].map(l2id).astype(int)

    # Build segment_abs_path
    tpl = dataset_cfg["segment_root_template"]
    df["segment_abs_path"] = df.apply(
        lambda row: str(root / tpl.format(split_seed=int(row["split_seed"])) / row["segment_path"]),
        axis=1,
    )

    # Type casts
    df["split_seed"] = df["split_seed"].astype(int)
    df["segment_index"] = df["segment_index"].astype(int)
    df["start_time_sec"] = df["start_time_sec"].astype(float)
    df["end_time_sec"] = df["end_time_sec"].astype(float)

    return df


def load_seed_data(
    dataset_config_path: str | Path,
    seed_config_path: str | Path,
    project_root: str | Path,
) -> pd.DataFrame:
    """One-stop loader: read manifests → join → inject label_id / abs_path.

    Args:
        dataset_config_path: Path to dataset_v1.yaml.
        seed_config_path: Path to seed{N}.yaml.
        project_root: Project root directory.

    Returns:
        Unified DataFrame.
    """
    dataset_cfg = load_dataset_config(dataset_config_path)
    seed_cfg = load_seed_config(seed_config_path)
    label_map = load_label_map(dataset_config_path)

    split_df, segment_df = load_manifests(seed_cfg, project_root)
    merged_df = join_manifests(split_df, segment_df)
    unified_df = build_unified_table(
        merged_df, label_map, dataset_cfg, seed_cfg, project_root
    )

    return unified_df
