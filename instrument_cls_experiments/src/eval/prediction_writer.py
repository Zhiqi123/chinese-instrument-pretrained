"""Write predictions CSV and metrics JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_predictions(
    predictions: list[dict],
    output_path: str | Path,
) -> None:
    """Write predictions CSV with fixed column order."""
    COLUMNS = [
        "segment_id", "sample_id", "record_id", "split", "source_dataset",
        "true_label", "true_label_id", "pred_label", "pred_label_id", "top1_score",
    ]
    df = pd.DataFrame(predictions)
    df = df[COLUMNS]
    df.to_csv(output_path, index=False)


def write_metrics(
    metrics: dict,
    output_path: str | Path,
) -> None:
    """Write metrics JSON."""
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def write_per_class_metrics(
    per_class_metrics: list[dict],
    output_path: str | Path,
) -> None:
    """Write per-class metrics CSV."""
    COLUMNS = ["label_id", "label", "precision", "recall", "f1", "support"]
    df = pd.DataFrame(per_class_metrics)
    df = df[COLUMNS]
    df.to_csv(output_path, index=False)
