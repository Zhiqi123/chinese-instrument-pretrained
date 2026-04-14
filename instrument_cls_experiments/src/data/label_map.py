"""Canonical label mapping from dataset_v1.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_label_map(dataset_config_path: str | Path) -> dict[str, Any]:
    """Load label mapping from dataset config.

    Returns:
        dict with keys: label_to_id, id_to_label, classes,
        num_classes, label_to_english.
    """
    path = Path(dataset_config_path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    classes_cfg = cfg["classes"]
    # Sort by label_id
    classes_cfg = sorted(classes_cfg, key=lambda c: c["label_id"])

    label_to_id: dict[str, int] = {}
    id_to_label: dict[int, str] = {}
    label_to_english: dict[str, str] = {}
    classes: list[str] = []

    for entry in classes_cfg:
        fl = entry["family_label"]
        lid = entry["label_id"]
        eng = entry["english_name"]
        label_to_id[fl] = lid
        id_to_label[lid] = fl
        label_to_english[fl] = eng
        classes.append(fl)

    # Verify label_ids are contiguous 0..N-1
    expected_ids = list(range(len(classes_cfg)))
    actual_ids = [c["label_id"] for c in classes_cfg]
    if actual_ids != expected_ids:
        raise ValueError(
            f"label_id must be contiguous 0..N-1, got {actual_ids}"
        )

    return {
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "classes": classes,
        "num_classes": len(classes),
        "label_to_english": label_to_english,
    }
