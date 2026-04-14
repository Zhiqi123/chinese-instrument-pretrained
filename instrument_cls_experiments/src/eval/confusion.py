"""Confusion matrix computation and export."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix


def compute_confusion_matrix(
    true_labels: list[int],
    pred_labels: list[int],
    label_map: dict,
) -> pd.DataFrame:
    """Compute confusion matrix as a DataFrame.

    Args:
        true_labels: Ground truth label_ids.
        pred_labels: Predicted label_ids.
        label_map: From load_label_map().

    Returns:
        DataFrame with Chinese class names as index/columns.
    """
    classes = label_map["classes"]  # sorted by label_id
    label_ids = [label_map["label_to_id"][c] for c in classes]

    cm = confusion_matrix(true_labels, pred_labels, labels=label_ids)

    # Sanity check
    assert cm.sum() == len(true_labels), (
        f"confusion_matrix sum={cm.sum()} != num_samples={len(true_labels)}"
    )

    return pd.DataFrame(cm, index=classes, columns=classes)


def write_confusion_matrix(
    cm_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Write confusion matrix CSV."""
    cm_df.to_csv(output_path)
