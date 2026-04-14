"""Classification metrics: accuracy, macro F1, per-class metrics."""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


def compute_metrics(
    true_labels: list[int],
    pred_labels: list[int],
    split: str,
) -> dict:
    """Compute split-level classification metrics.

    Returns:
        dict: split, num_samples, accuracy, macro_f1,
        macro_precision, macro_recall.
    """
    return {
        "split": split,
        "num_samples": len(true_labels),
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
        "macro_f1": float(f1_score(
            true_labels, pred_labels, average="macro", zero_division=0,
        )),
        "macro_precision": float(precision_score(
            true_labels, pred_labels, average="macro", zero_division=0,
        )),
        "macro_recall": float(recall_score(
            true_labels, pred_labels, average="macro", zero_division=0,
        )),
    }


def compute_per_class_metrics(
    true_labels: list[int],
    pred_labels: list[int],
    label_map: dict,
) -> list[dict]:
    """Compute per-class precision, recall, F1, support.

    Returns:
        list[dict] sorted by label_id.
    """
    classes = label_map["classes"]  # sorted by label_id
    label_ids = [label_map["label_to_id"][c] for c in classes]

    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        true_labels, pred_labels,
        labels=label_ids,
        average=None,
        zero_division=0,
    )

    results = []
    for i, cls in enumerate(classes):
        results.append({
            "label_id": label_ids[i],
            "label": cls,
            "precision": float(precisions[i]),
            "recall": float(recalls[i]),
            "f1": float(f1s[i]),
            "support": int(supports[i]),
        })

    # Sanity check
    total_support = sum(r["support"] for r in results)
    assert total_support == len(true_labels), (
        f"sum(per-class support)={total_support} != num_samples={len(true_labels)}"
    )

    return results
