"""Result aggregation across seeds for all experiment phases."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = [
    {
        "method_short": "mfcc_svm",
        "method_display": "MFCC+SVM",
        "model": "mfcc_svm",
        "experiment_id": "exp01_mfcc_svm",
        "eval_splits": ["val", "test", "external_test"],
        "val_metrics_file": "val_metrics.json",
    },
    {
        "method_short": "clap_zeroshot",
        "method_display": "CLAP zero-shot",
        "model": "clap_htsat_unfused",
        "experiment_id": "exp01_clap_zeroshot",
        "eval_splits": ["test", "external_test"],
        "val_metrics_file": None,
    },
    {
        "method_short": "clap_linear",
        "method_display": "CLAP linear probe",
        "model": "clap_htsat_unfused",
        "experiment_id": "exp01_clap_linear",
        "eval_splits": ["val", "test", "external_test"],
        "val_metrics_file": "val_metrics_best.json",
    },
    {
        "method_short": "mert_linear",
        "method_display": "MERT linear probe",
        "model": "mert_v1_95m",
        "experiment_id": "exp01_mert_linear",
        "eval_splits": ["val", "test", "external_test"],
        "val_metrics_file": "val_metrics_best.json",
    },
]

SEED_NAMES = ["seed0", "seed1", "seed2"]


def _get_metrics_filename(method: dict, split: str) -> str:
    """Get metrics filename. Val uses _best suffix for linear probes."""
    if split == "val" and method.get("val_metrics_file"):
        return method["val_metrics_file"]
    return f"{split}_metrics.json"


def collect_run_metrics(
    runs_base_dir: Path,
) -> pd.DataFrame:
    """Collect run metrics from output directories.

    Returns:
        DataFrame with columns: experiment_id, run_id, method, model,
            seed_name, split, num_samples, accuracy, macro_f1,
            macro_precision, macro_recall
    """
    rows = []
    for method in METHODS:
        for seed_name in SEED_NAMES:
            run_dir = runs_base_dir / method["method_short"] / seed_name
            run_id = f"run_exp01_{method['method_short']}_{seed_name}"

            for split in method["eval_splits"]:
                metrics_file = run_dir / _get_metrics_filename(method, split)
                if not metrics_file.exists():
                    continue

                with open(metrics_file) as f:
                    m = json.load(f)

                rows.append({
                    "experiment_id": method["experiment_id"],
                    "run_id": run_id,
                    "method": method["method_display"],
                    "model": method["model"],
                    "seed_name": seed_name,
                    "split": split,
                    "num_samples": m["num_samples"],
                    "accuracy": m["accuracy"],
                    "macro_f1": m["macro_f1"],
                    "macro_precision": m.get("macro_precision", float("nan")),
                    "macro_recall": m.get("macro_recall", float("nan")),
                })

    return pd.DataFrame(rows)


def validate_completeness(run_metrics_df: pd.DataFrame) -> None:
    """Validate each method × split has exactly 3 seeds."""
    # 1. Check for entirely missing method × split groups
    expected = {
        (m["method_display"], s)
        for m in METHODS
        for s in m["eval_splits"]
    }
    actual = set(run_metrics_df.groupby(["method", "split"]).groups.keys()) if len(run_metrics_df) > 0 else set()
    missing = expected - actual
    if missing:
        raise RuntimeError(
            f"Completeness check failed: method × split entirely missing: {missing}"
        )

    # 2. Check seed count per group
    grouped = run_metrics_df.groupby(["method", "split"])

    for (method, split), group in grouped:
        seed_count = len(group)
        if seed_count != 3:
            raise RuntimeError(
                f"Completeness check failed: {method} × {split} has {seed_count} seeds "
                f"(expected 3). missing: {set(SEED_NAMES) - set(group['seed_name'])}"
            )

        # Warn if num_samples varies across seeds
        unique_counts = group["num_samples"].unique()
        if len(unique_counts) != 1:
            print(f"  WARNING: {method} × {split} num_samples not identical across seeds: "
                  f"{group[['seed_name', 'num_samples']].values.tolist()}")


def aggregate_summary_mean_std(
    run_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate mean ± std by method × split.

    Returns:
        DataFrame with columns: method, model, split, seed_count,
            num_samples_per_seed, accuracy_mean, accuracy_std,
            macro_f1_mean, macro_f1_std, macro_precision_mean,
            macro_precision_std, macro_recall_mean, macro_recall_std
    """
    rows = []
    grouped = run_metrics_df.groupby(["method", "model", "split"])

    for (method, model, split), group in grouped:
        # Use consistent value or median
        unique_ns = group["num_samples"].unique()
        if len(unique_ns) == 1:
            nps = int(unique_ns[0])
        else:
            nps = int(group["num_samples"].median())

        rows.append({
            "method": method,
            "model": model,
            "split": split,
            "seed_count": len(group),
            "num_samples_per_seed": nps,
            "accuracy_mean": float(group["accuracy"].mean()),
            "accuracy_std": float(group["accuracy"].std(ddof=1)),
            "macro_f1_mean": float(group["macro_f1"].mean()),
            "macro_f1_std": float(group["macro_f1"].std(ddof=1)),
            "macro_precision_mean": float(group["macro_precision"].mean()),
            "macro_precision_std": float(group["macro_precision"].std(ddof=1)),
            "macro_recall_mean": float(group["macro_recall"].mean()),
            "macro_recall_std": float(group["macro_recall"].std(ddof=1)),
        })

    return pd.DataFrame(rows)


def _get_per_class_filename(method: dict, split: str) -> str:
    """Get per-class metrics filename."""
    if split == "val" and method.get("val_metrics_file", "").endswith("_best.json"):
        return "val_per_class_metrics_best.csv"
    return f"{split}_per_class_metrics.csv"


def collect_per_class_metrics(
    runs_base_dir: Path,
) -> pd.DataFrame:
    """Collect per-class metrics from output directories.

    Returns:
        DataFrame with columns: method, model, split, seed_name,
            label_id, label, precision, recall, f1, support
    """
    rows = []
    for method in METHODS:
        for seed_name in SEED_NAMES:
            run_dir = runs_base_dir / method["method_short"] / seed_name

            for split in method["eval_splits"]:
                pc_file = run_dir / _get_per_class_filename(method, split)
                if not pc_file.exists():
                    continue

                pc_df = pd.read_csv(pc_file)
                for _, row in pc_df.iterrows():
                    rows.append({
                        "method": method["method_display"],
                        "model": method["model"],
                        "split": split,
                        "seed_name": seed_name,
                        "label_id": row["label_id"],
                        "label": row["label"],
                        "precision": row["precision"],
                        "recall": row["recall"],
                        "f1": row["f1"],
                        "support": row["support"],
                    })

    return pd.DataFrame(rows)


def aggregate_per_class_summary(
    per_class_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate per-class mean ± std by method × split × class.

    Returns:
        DataFrame with columns: method, model, split, label_id, label,
            f1_mean, f1_std, precision_mean, precision_std,
            recall_mean, recall_std
    """
    rows = []
    grouped = per_class_df.groupby(["method", "model", "split", "label_id", "label"])

    for (method, model, split, label_id, label), group in grouped:
        rows.append({
            "method": method,
            "model": model,
            "split": split,
            "label_id": int(label_id),
            "label": label,
            "f1_mean": float(group["f1"].mean()),
            "f1_std": float(group["f1"].std(ddof=1)),
            "precision_mean": float(group["precision"].mean()),
            "precision_std": float(group["precision"].std(ddof=1)),
            "recall_mean": float(group["recall"].mean()),
            "recall_std": float(group["recall"].std(ddof=1)),
        })

    return pd.DataFrame(rows)


def run_aggregation(
    runs_base_dir: Path,
    output_dir: Path,
) -> None:
    """Main entry: collect, validate, aggregate, write tables.

    Args:
        runs_base_dir: runs/exp01_transfer/
        output_dir: reports/tables/
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect run-level metrics
    run_metrics_df = collect_run_metrics(runs_base_dir)

    # Write run_metrics (may be incomplete)
    run_metrics_df.to_csv(
        output_dir / "exp01_transfer_run_metrics.csv", index=False,
    )
    print(f"  run_metrics: {len(run_metrics_df)} rows written")

    # 2. Completeness check
    validate_completeness(run_metrics_df)

    # 3. Aggregate summary_mean_std
    summary_df = aggregate_summary_mean_std(run_metrics_df)
    summary_df.to_csv(
        output_dir / "exp01_transfer_summary_mean_std.csv", index=False,
    )
    print(f"  summary_mean_std: {len(summary_df)} rows written")

    # 4. Collect and aggregate per_class
    per_class_df = collect_per_class_metrics(runs_base_dir)
    per_class_summary_df = aggregate_per_class_summary(per_class_df)
    per_class_summary_df.to_csv(
        output_dir / "exp01_transfer_per_class_summary.csv", index=False,
    )
    print(f"  per_class_summary: {len(per_class_summary_df)} rows written")

    print("  Aggregation: DONE")


# Phase 2: exp02_adaptation aggregation

EXP02_METHODS = [
    {
        "method_short": "mert_linear",
        "method_display": "MERT linear probe",
        "model": "mert_v1_95m",
        "experiment_id": "exp02_mert_linear",
        "eval_splits": ["val", "test", "external_test"],
        "val_metrics_file": "val_metrics_best.json",
    },
    {
        "method_short": "mert_lora_r4",
        "method_display": "MERT+LoRA (r=4)",
        "model": "mert_v1_95m",
        "experiment_id": "exp02_mert_lora_r4",
        "eval_splits": ["val", "test", "external_test"],
        "val_metrics_file": "val_metrics_best.json",
    },
    {
        "method_short": "mert_lora_r8",
        "method_display": "MERT+LoRA (r=8)",
        "model": "mert_v1_95m",
        "experiment_id": "exp02_mert_lora_r8",
        "eval_splits": ["val", "test", "external_test"],
        "val_metrics_file": "val_metrics_best.json",
    },
    {
        "method_short": "mert_full_ft",
        "method_display": "MERT full fine-tuning",
        "model": "mert_v1_95m",
        "experiment_id": "exp02_mert_full_ft",
        "eval_splits": ["val", "test", "external_test"],
        "val_metrics_file": "val_metrics_best.json",
    },
]


def _collect_run_metrics_generic(
    runs_base_dir: Path,
    methods: list[dict],
    run_id_prefix: str,
) -> pd.DataFrame:
    """Generic run metrics collector."""
    rows = []
    for method in methods:
        for seed_name in SEED_NAMES:
            run_dir = runs_base_dir / method["method_short"] / seed_name
            run_id = f"{run_id_prefix}_{method['method_short']}_{seed_name}"

            for split in method["eval_splits"]:
                metrics_file = run_dir / _get_metrics_filename(method, split)
                if not metrics_file.exists():
                    continue

                with open(metrics_file) as f:
                    m = json.load(f)

                rows.append({
                    "experiment_id": method["experiment_id"],
                    "run_id": run_id,
                    "method": method["method_display"],
                    "model": method["model"],
                    "seed_name": seed_name,
                    "split": split,
                    "num_samples": m["num_samples"],
                    "accuracy": m["accuracy"],
                    "macro_f1": m["macro_f1"],
                    "macro_precision": m.get("macro_precision", float("nan")),
                    "macro_recall": m.get("macro_recall", float("nan")),
                })

    return pd.DataFrame(rows)


def _validate_completeness_generic(
    run_metrics_df: pd.DataFrame,
    methods: list[dict],
) -> None:
    """Generic completeness check."""
    expected = {
        (m["method_display"], s)
        for m in methods
        for s in m["eval_splits"]
    }
    actual = set(run_metrics_df.groupby(["method", "split"]).groups.keys()) if len(run_metrics_df) > 0 else set()
    missing = expected - actual
    if missing:
        raise RuntimeError(
            f"Completeness check failed: method × split entirely missing: {missing}"
        )

    grouped = run_metrics_df.groupby(["method", "split"])
    for (method, split), group in grouped:
        seed_count = len(group)
        if seed_count != 3:
            raise RuntimeError(
                f"Completeness check failed: {method} × {split} has {seed_count} seeds "
                f"(expected 3). missing: {set(SEED_NAMES) - set(group['seed_name'])}"
            )


def _collect_per_class_generic(
    runs_base_dir: Path,
    methods: list[dict],
) -> pd.DataFrame:
    """Generic per-class metrics collector."""
    rows = []
    for method in methods:
        for seed_name in SEED_NAMES:
            run_dir = runs_base_dir / method["method_short"] / seed_name

            for split in method["eval_splits"]:
                pc_file = run_dir / _get_per_class_filename(method, split)
                if not pc_file.exists():
                    continue

                pc_df = pd.read_csv(pc_file)
                for _, row in pc_df.iterrows():
                    rows.append({
                        "method": method["method_display"],
                        "model": method["model"],
                        "split": split,
                        "seed_name": seed_name,
                        "label_id": row["label_id"],
                        "label": row["label"],
                        "precision": row["precision"],
                        "recall": row["recall"],
                        "f1": row["f1"],
                        "support": row["support"],
                    })

    return pd.DataFrame(rows)


def collect_exp02_run_metrics(runs_base_dir: Path) -> pd.DataFrame:
    """Collect Phase 2 run-level metrics."""
    return _collect_run_metrics_generic(runs_base_dir, EXP02_METHODS, "run_exp02")


def validate_exp02_completeness(run_metrics_df: pd.DataFrame) -> None:
    """Phase 2 completeness check."""
    _validate_completeness_generic(run_metrics_df, EXP02_METHODS)


def collect_exp02_per_class_metrics(runs_base_dir: Path) -> pd.DataFrame:
    """Collect Phase 2 per-class metrics."""
    return _collect_per_class_generic(runs_base_dir, EXP02_METHODS)


def collect_exp02_cost_metrics(runs_base_dir: Path) -> pd.DataFrame:
    """Collect Phase 2 cost metrics."""
    rows = []
    for method in EXP02_METHODS:
        for seed_name in SEED_NAMES:
            cost_file = runs_base_dir / method["method_short"] / seed_name / "cost_metrics.json"
            if not cost_file.exists():
                continue

            with open(cost_file) as f:
                cm = json.load(f)

            rows.append({
                "experiment_id": method["experiment_id"],
                "run_id": f"run_exp02_{method['method_short']}_{seed_name}",
                "method": method["method_display"],
                "model": method["model"],
                "seed_name": seed_name,
                "device_type": cm["device_type"],
                "total_params": cm["total_params"],
                "trainable_params": cm["trainable_params"],
                "trainable_ratio": cm["trainable_ratio"],
                "feature_prep_time_sec": cm["feature_prep_time_sec"],
                "train_wall_time_sec": cm["train_wall_time_sec"],
                "total_method_time_sec": cm["total_method_time_sec"],
                "mean_epoch_time_sec": cm["mean_epoch_time_sec"],
                "mean_train_step_time_ms": cm["mean_train_step_time_ms"],
                "peak_device_memory_mb": cm["peak_device_memory_mb"],
                "memory_metric_available": cm["memory_metric_available"],
            })

    return pd.DataFrame(rows)


def validate_exp02_cost_completeness(cost_df: pd.DataFrame) -> None:
    """Validate each method has 3 seeds of cost data."""
    for method in EXP02_METHODS:
        method_rows = cost_df[cost_df["method"] == method["method_display"]]
        if len(method_rows) != 3:
            raise RuntimeError(
                f"Cost completeness check failed: {method['method_display']} has "
                f"{len(method_rows)} seeds (expected 3)"
            )


def aggregate_exp02_cost_summary(cost_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cost mean ± std by method."""
    rows = []
    grouped = cost_df.groupby(["method", "model"])

    numeric_fields = [
        "trainable_params", "trainable_ratio",
        "feature_prep_time_sec", "train_wall_time_sec",
        "total_method_time_sec", "mean_epoch_time_sec",
        "mean_train_step_time_ms",
    ]

    for (method, model), group in grouped:
        row = {
            "method": method,
            "model": model,
            "seed_count": len(group),
            "total_params": int(group["total_params"].iloc[0]),
        }

        for field in numeric_fields:
            vals = group[field].astype(float)
            row[f"{field}_mean"] = float(vals.mean())
            row[f"{field}_std"] = float(vals.std(ddof=1))

        # Handle unavailable memory metrics
        mem_available = group["memory_metric_available"].astype(bool)
        if mem_available.any():
            mem_vals = group.loc[mem_available.values, "peak_device_memory_mb"].astype(float)
            row["peak_device_memory_mb_mean"] = float(mem_vals.mean())
            row["peak_device_memory_mb_std"] = float(mem_vals.std(ddof=1)) if len(mem_vals) > 1 else 0.0
        else:
            row["peak_device_memory_mb_mean"] = None
            row["peak_device_memory_mb_std"] = None

        rows.append(row)

    return pd.DataFrame(rows)


def run_exp02_aggregation(
    runs_base_dir: Path,
    output_dir: Path,
) -> None:
    """Phase 2 main aggregation entry point."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Performance metrics
    run_metrics_df = collect_exp02_run_metrics(runs_base_dir)
    run_metrics_df.to_csv(
        output_dir / "exp02_adaptation_run_metrics.csv", index=False,
    )
    print(f"  run_metrics: {len(run_metrics_df)} rows written")

    validate_exp02_completeness(run_metrics_df)

    summary_df = aggregate_summary_mean_std(run_metrics_df)
    summary_df.to_csv(
        output_dir / "exp02_adaptation_summary_mean_std.csv", index=False,
    )
    print(f"  summary_mean_std: {len(summary_df)} rows written")

    # 2. Per-class
    per_class_df = collect_exp02_per_class_metrics(runs_base_dir)
    per_class_summary_df = aggregate_per_class_summary(per_class_df)
    per_class_summary_df.to_csv(
        output_dir / "exp02_adaptation_per_class_summary.csv", index=False,
    )
    print(f"  per_class_summary: {len(per_class_summary_df)} rows written")

    # 3. Cost metrics
    cost_df = collect_exp02_cost_metrics(runs_base_dir)
    cost_df.to_csv(
        output_dir / "exp02_adaptation_run_costs.csv", index=False,
    )
    print(f"  run_costs: {len(cost_df)} rows written")

    validate_exp02_cost_completeness(cost_df)

    cost_summary_df = aggregate_exp02_cost_summary(cost_df)
    cost_summary_df.to_csv(
        output_dir / "exp02_adaptation_cost_summary_mean_std.csv", index=False,
    )
    print(f"  cost_summary: {len(cost_summary_df)} rows written")

    print("  Phase 2 Aggregation: DONE")


# Phase 3: exp03_data_efficiency aggregation

EXP03_METHODS = [
    {
        "method_short": "mert_linear",
        "method_display": "MERT linear probe",
        "model": "mert_v1_95m",
        "experiment_id": "exp03_mert_linear",
        "eval_splits": ["val", "test", "external_test"],
        "val_metrics_file": "val_metrics_best.json",
    },
    {
        "method_short": "mert_lora_r4",
        "method_display": "MERT+LoRA (r=4)",
        "model": "mert_v1_95m",
        "experiment_id": "exp03_mert_lora_r4",
        "eval_splits": ["val", "test", "external_test"],
        "val_metrics_file": "val_metrics_best.json",
    },
    {
        "method_short": "mert_lora_r8",
        "method_display": "MERT+LoRA (r=8)",
        "model": "mert_v1_95m",
        "experiment_id": "exp03_mert_lora_r8",
        "eval_splits": ["val", "test", "external_test"],
        "val_metrics_file": "val_metrics_best.json",
    },
    {
        "method_short": "mert_full_ft",
        "method_display": "MERT full fine-tuning",
        "model": "mert_v1_95m",
        "experiment_id": "exp03_mert_full_ft",
        "eval_splits": ["val", "test", "external_test"],
        "val_metrics_file": "val_metrics_best.json",
    },
]

EXP03_RATIOS = ["train10", "train25", "train50", "train100"]

EXP03_RATIO_VALUES = {
    "train10": 0.10,
    "train25": 0.25,
    "train50": 0.50,
    "train100": 1.00,
}


def validate_exp03_run_output_contract(runs_base_dir: Path) -> None:
    """Validate output contract for all 48 runs."""
    required_files = [
        "val_predictions_best.csv",
        "val_metrics_best.json",
        "val_per_class_metrics_best.csv",
        "val_confusion_matrix_best.csv",
        "test_predictions.csv",
        "test_metrics.json",
        "test_per_class_metrics.csv",
        "test_confusion_matrix.csv",
        "external_test_predictions.csv",
        "external_test_metrics.json",
        "external_test_per_class_metrics.csv",
        "external_test_confusion_matrix.csv",
        "cost_metrics.json",
        "artifacts/train_subset_segments.csv",
        "artifacts/train_subset_groups.csv",
    ]

    missing_list = []
    for method in EXP03_METHODS:
        for ratio_name in EXP03_RATIOS:
            for seed_name in SEED_NAMES:
                run_dir = runs_base_dir / method["method_short"] / ratio_name / seed_name
                for f in required_files:
                    if not (run_dir / f).exists():
                        missing_list.append(
                            f"{method['method_short']}/{ratio_name}/{seed_name}/{f}"
                        )

    if missing_list:
        raise RuntimeError(
            f"Run output contract failed: {len(missing_list)} files missing. "
            f"first 10: {missing_list[:10]}"
        )


def validate_exp03_per_class_completeness(runs_base_dir: Path) -> None:
    """Validate per-class files exist for all 48 runs."""
    per_class_files = {
        "val": "val_per_class_metrics_best.csv",
        "test": "test_per_class_metrics.csv",
        "external_test": "external_test_per_class_metrics.csv",
    }
    missing_list = []
    for method in EXP03_METHODS:
        for ratio_name in EXP03_RATIOS:
            for seed_name in SEED_NAMES:
                run_dir = runs_base_dir / method["method_short"] / ratio_name / seed_name
                for split, filename in per_class_files.items():
                    if not (run_dir / filename).exists():
                        missing_list.append(
                            f"{method['method_short']}/{ratio_name}/{seed_name}/{filename}"
                        )

    if missing_list:
        raise RuntimeError(
            f"Per-class completeness check failed: {len(missing_list)} files missing. "
            f"first 10: {missing_list[:10]}"
        )


def collect_exp03_run_metrics(runs_base_dir: Path) -> pd.DataFrame:
    """Collect Phase 3 run-level metrics."""
    rows = []
    for method in EXP03_METHODS:
        for ratio_name in EXP03_RATIOS:
            for seed_name in SEED_NAMES:
                run_dir = runs_base_dir / method["method_short"] / ratio_name / seed_name
                run_id = f"run_exp03_{method['method_short']}_{ratio_name}_{seed_name}"

                for split in method["eval_splits"]:
                    metrics_file = run_dir / _get_metrics_filename(method, split)
                    if not metrics_file.exists():
                        continue

                    with open(metrics_file) as f:
                        m = json.load(f)

                    rows.append({
                        "experiment_id": method["experiment_id"],
                        "run_id": run_id,
                        "method": method["method_display"],
                        "model": method["model"],
                        "ratio_name": ratio_name,
                        "ratio_value": EXP03_RATIO_VALUES[ratio_name],
                        "seed_name": seed_name,
                        "split": split,
                        "num_samples": m["num_samples"],
                        "accuracy": m["accuracy"],
                        "macro_f1": m["macro_f1"],
                        "macro_precision": m.get("macro_precision", float("nan")),
                        "macro_recall": m.get("macro_recall", float("nan")),
                    })

    return pd.DataFrame(rows)


def validate_exp03_completeness(run_metrics_df: pd.DataFrame) -> None:
    """Phase 3 completeness check: 4 methods × 4 ratios × 3 seeds = 48 runs.

    Checks:
      1. All 48 run_ids present
      2. Each method × ratio × split has seed0/seed1/seed2
      3. Each method × ratio has val/test/external_test
    """
    # Level 1: all 48 run_ids present
    expected_runs = set()
    for method in EXP03_METHODS:
        for ratio_name in EXP03_RATIOS:
            for seed_name in SEED_NAMES:
                expected_runs.add(
                    f"run_exp03_{method['method_short']}_{ratio_name}_{seed_name}"
                )

    if len(expected_runs) != 48:
        raise RuntimeError(f"expected 48 run_ids, got {len(expected_runs)}")

    actual_runs = set(run_metrics_df["run_id"].unique())
    missing = expected_runs - actual_runs
    if missing:
        raise RuntimeError(
            f"Completeness check failed: missing {len(missing)} runs: "
            f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}"
        )

    # Level 2: each method × ratio × split has seed0/seed1/seed2
    expected_seeds = set(SEED_NAMES)
    grouped = run_metrics_df.groupby(["method", "ratio_name", "split"])
    for (method, ratio, split), group in grouped:
        actual_seeds = set(group["seed_name"])
        if actual_seeds != expected_seeds:
            raise RuntimeError(
                f"Completeness check failed: {method} × {ratio} × {split} "
                f"seeds {actual_seeds} != expected {expected_seeds}"
            )

    # Level 3: each method × ratio has val/test/external_test
    expected_splits = {"val", "test", "external_test"}
    method_ratio_grouped = run_metrics_df.groupby(["method", "ratio_name"])
    for (method, ratio), group in method_ratio_grouped:
        actual_splits = set(group["split"].unique())
        missing_splits = expected_splits - actual_splits
        if missing_splits:
            raise RuntimeError(
                f"Completeness check failed: {method} × {ratio} missing splits: {missing_splits}"
            )


def collect_exp03_per_class_metrics(runs_base_dir: Path) -> pd.DataFrame:
    """Collect Phase 3 per-class metrics."""
    rows = []
    for method in EXP03_METHODS:
        for ratio_name in EXP03_RATIOS:
            for seed_name in SEED_NAMES:
                run_dir = runs_base_dir / method["method_short"] / ratio_name / seed_name

                for split in method["eval_splits"]:
                    pc_file = run_dir / _get_per_class_filename(method, split)
                    if not pc_file.exists():
                        continue

                    pc_df = pd.read_csv(pc_file)
                    for _, row in pc_df.iterrows():
                        rows.append({
                            "method": method["method_display"],
                            "model": method["model"],
                            "ratio_name": ratio_name,
                            "ratio_value": EXP03_RATIO_VALUES[ratio_name],
                            "split": split,
                            "seed_name": seed_name,
                            "label_id": row["label_id"],
                            "label": row["label"],
                            "precision": row["precision"],
                            "recall": row["recall"],
                            "f1": row["f1"],
                            "support": row["support"],
                        })

    return pd.DataFrame(rows)


def aggregate_exp03_per_class_summary(per_class_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-class mean ± std by method × ratio × split."""
    rows = []
    grouped = per_class_df.groupby(
        ["method", "model", "ratio_name", "ratio_value", "split", "label_id", "label"]
    )

    for (method, model, ratio_name, ratio_value, split, label_id, label), group in grouped:
        rows.append({
            "method": method,
            "model": model,
            "ratio_name": ratio_name,
            "ratio_value": ratio_value,
            "split": split,
            "label_id": int(label_id),
            "label": label,
            "f1_mean": float(group["f1"].mean()),
            "f1_std": float(group["f1"].std(ddof=1)),
            "precision_mean": float(group["precision"].mean()),
            "precision_std": float(group["precision"].std(ddof=1)),
            "recall_mean": float(group["recall"].mean()),
            "recall_std": float(group["recall"].std(ddof=1)),
        })

    return pd.DataFrame(rows)


def collect_exp03_cost_metrics(runs_base_dir: Path) -> pd.DataFrame:
    """Collect Phase 3 cost metrics with subset stats."""
    rows = []
    for method in EXP03_METHODS:
        for ratio_name in EXP03_RATIOS:
            for seed_name in SEED_NAMES:
                run_dir = runs_base_dir / method["method_short"] / ratio_name / seed_name
                cost_file = run_dir / "cost_metrics.json"
                meta_file = run_dir / "run_meta.json"
                if not cost_file.exists():
                    continue

                with open(cost_file) as f:
                    cm = json.load(f)

                # Read subset stats from run_meta.json
                subset_group_count = None
                subset_segment_count = None
                if meta_file.exists():
                    with open(meta_file) as f:
                        rm = json.load(f)
                    subset_group_count = rm.get("subset_train_group_count")
                    subset_segment_count = rm.get("subset_train_segment_count")

                rows.append({
                    "experiment_id": method["experiment_id"],
                    "run_id": f"run_exp03_{method['method_short']}_{ratio_name}_{seed_name}",
                    "method": method["method_display"],
                    "model": method["model"],
                    "ratio_name": ratio_name,
                    "ratio_value": EXP03_RATIO_VALUES[ratio_name],
                    "seed_name": seed_name,
                    "device_type": cm["device_type"],
                    "total_params": cm["total_params"],
                    "trainable_params": cm["trainable_params"],
                    "trainable_ratio": cm["trainable_ratio"],
                    "feature_prep_time_sec": cm["feature_prep_time_sec"],
                    "train_wall_time_sec": cm["train_wall_time_sec"],
                    "total_method_time_sec": cm["total_method_time_sec"],
                    "mean_epoch_time_sec": cm["mean_epoch_time_sec"],
                    "mean_train_step_time_ms": cm["mean_train_step_time_ms"],
                    "peak_device_memory_mb": cm["peak_device_memory_mb"],
                    "memory_metric_available": cm["memory_metric_available"],
                    "subset_train_group_count": subset_group_count,
                    "subset_train_segment_count": subset_segment_count,
                })

    return pd.DataFrame(rows)


def aggregate_exp03_cost_summary(cost_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cost mean ± std by method × ratio."""
    rows = []
    grouped = cost_df.groupby(["method", "model", "ratio_name", "ratio_value"])

    numeric_fields = [
        "trainable_params", "trainable_ratio",
        "feature_prep_time_sec", "train_wall_time_sec",
        "total_method_time_sec", "mean_epoch_time_sec",
        "mean_train_step_time_ms",
    ]

    for (method, model, ratio_name, ratio_value), group in grouped:
        row = {
            "method": method,
            "model": model,
            "ratio_name": ratio_name,
            "ratio_value": ratio_value,
            "seed_count": len(group),
            "total_params": int(group["total_params"].iloc[0]),
        }

        for field in numeric_fields:
            vals = group[field].astype(float)
            row[f"{field}_mean"] = float(vals.mean())
            row[f"{field}_std"] = float(vals.std(ddof=1))

        mem_available = group["memory_metric_available"].astype(bool)
        if mem_available.any():
            mem_vals = group.loc[mem_available.values, "peak_device_memory_mb"].astype(float)
            row["peak_device_memory_mb_mean"] = float(mem_vals.mean())
            row["peak_device_memory_mb_std"] = float(mem_vals.std(ddof=1)) if len(mem_vals) > 1 else 0.0
        else:
            row["peak_device_memory_mb_mean"] = None
            row["peak_device_memory_mb_std"] = None

        rows.append(row)

    return pd.DataFrame(rows)


def collect_exp03_learning_curve_points(
    run_metrics_df: pd.DataFrame,
    cost_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate learning curve plot data."""
    rows = []

    # Performance metrics
    perf_grouped = run_metrics_df.groupby(
        ["method", "model", "ratio_name", "ratio_value", "split"]
    )
    perf_stats = {}
    for (method, model, ratio_name, ratio_value, split), group in perf_grouped:
        key = (method, model, ratio_name, ratio_value, split)
        perf_stats[key] = {
            "macro_f1_mean": float(group["macro_f1"].mean()),
            "macro_f1_std": float(group["macro_f1"].std(ddof=1)),
            "accuracy_mean": float(group["accuracy"].mean()),
            "accuracy_std": float(group["accuracy"].std(ddof=1)),
        }

    # Cost metrics
    cost_grouped = cost_df.groupby(["method", "model", "ratio_name", "ratio_value"])
    cost_stats = {}
    for (method, model, ratio_name, ratio_value), group in cost_grouped:
        cost_stats[(method, model, ratio_name, ratio_value)] = {
            "total_method_time_sec_mean": float(group["total_method_time_sec"].mean()),
            "total_method_time_sec_std": float(group["total_method_time_sec"].std(ddof=1)),
            "trainable_params_mean": float(group["trainable_params"].astype(float).mean()),
        }

    for key, stats in perf_stats.items():
        method, model, ratio_name, ratio_value, split = key
        cost_key = (method, model, ratio_name, ratio_value)
        cs = cost_stats.get(cost_key, {})
        rows.append({
            "method": method,
            "model": model,
            "ratio_name": ratio_name,
            "ratio_value": ratio_value,
            "split": split,
            "macro_f1_mean": stats["macro_f1_mean"],
            "macro_f1_std": stats["macro_f1_std"],
            "accuracy_mean": stats["accuracy_mean"],
            "accuracy_std": stats["accuracy_std"],
            "total_method_time_sec_mean": cs.get("total_method_time_sec_mean"),
            "total_method_time_sec_std": cs.get("total_method_time_sec_std"),
            "trainable_params_mean": cs.get("trainable_params_mean"),
        })

    return pd.DataFrame(rows)


def run_exp03_aggregation(
    runs_base_dir: Path,
    output_dir: Path,
) -> None:
    """Phase 3 main aggregation entry point."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 0. Validate run output contract
    validate_exp03_run_output_contract(runs_base_dir)
    print("  run output contract: PASSED")

    # 1. Performance metrics
    run_metrics_df = collect_exp03_run_metrics(runs_base_dir)
    run_metrics_df.to_csv(
        output_dir / "exp03_data_efficiency_run_metrics.csv", index=False,
    )
    print(f"  run_metrics: {len(run_metrics_df)} rows written")

    validate_exp03_completeness(run_metrics_df)

    # Summary mean ± std
    summary_rows = []
    grouped = run_metrics_df.groupby(
        ["method", "model", "ratio_name", "ratio_value", "split"]
    )
    for (method, model, ratio_name, ratio_value, split), group in grouped:
        unique_ns = group["num_samples"].unique()
        nps = int(unique_ns[0]) if len(unique_ns) == 1 else int(group["num_samples"].median())
        summary_rows.append({
            "method": method,
            "model": model,
            "ratio_name": ratio_name,
            "ratio_value": ratio_value,
            "split": split,
            "seed_count": len(group),
            "num_samples_per_seed": nps,
            "accuracy_mean": float(group["accuracy"].mean()),
            "accuracy_std": float(group["accuracy"].std(ddof=1)),
            "macro_f1_mean": float(group["macro_f1"].mean()),
            "macro_f1_std": float(group["macro_f1"].std(ddof=1)),
            "macro_precision_mean": float(group["macro_precision"].mean()),
            "macro_precision_std": float(group["macro_precision"].std(ddof=1)),
            "macro_recall_mean": float(group["macro_recall"].mean()),
            "macro_recall_std": float(group["macro_recall"].std(ddof=1)),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        output_dir / "exp03_data_efficiency_summary_mean_std.csv", index=False,
    )
    print(f"  summary_mean_std: {len(summary_df)} rows written")

    # 2. Per-class
    validate_exp03_per_class_completeness(runs_base_dir)
    print("  per-class completeness: PASSED")

    per_class_df = collect_exp03_per_class_metrics(runs_base_dir)
    per_class_summary_df = aggregate_exp03_per_class_summary(per_class_df)
    per_class_summary_df.to_csv(
        output_dir / "exp03_data_efficiency_per_class_summary.csv", index=False,
    )
    print(f"  per_class_summary: {len(per_class_summary_df)} rows written")

    # 3. Cost metrics
    cost_df = collect_exp03_cost_metrics(runs_base_dir)
    cost_df.to_csv(
        output_dir / "exp03_data_efficiency_run_costs.csv", index=False,
    )
    print(f"  run_costs: {len(cost_df)} rows written")

    # Cost completeness check
    expected_cost_runs = 48
    if len(cost_df) != expected_cost_runs:
        raise RuntimeError(
            f"Cost completeness check failed: expected {expected_cost_runs} runs, "
            f"got {len(cost_df)}"
        )

    cost_summary_df = aggregate_exp03_cost_summary(cost_df)
    cost_summary_df.to_csv(
        output_dir / "exp03_data_efficiency_cost_summary_mean_std.csv", index=False,
    )
    print(f"  cost_summary: {len(cost_summary_df)} rows written")

    # 4. Learning curve data
    lc_df = collect_exp03_learning_curve_points(run_metrics_df, cost_df)
    lc_df.to_csv(
        output_dir / "exp03_learning_curve_points.csv", index=False,
    )
    print(f"  learning_curve_points: {len(lc_df)} rows written")

    print("  Phase 3 Aggregation: DONE")


# Phase 4: exp04_mert_layer_probe aggregation

EXP04_METHOD = {
    "method_short": "mert_layer_probe",
    "method_display": "MERT layer probe",
    "model": "mert_v1_95m",
    "experiment_id": "exp04_mert_layer_probe",
    "eval_splits": ["val", "test", "external_test"],
    "val_metrics_file": "val_metrics_best.json",
}


def collect_exp04_layer_probe_run_metrics(
    runs_base_dir: Path,
    num_layers: int,
) -> pd.DataFrame:
    """Collect Phase 4 layer probe run metrics."""
    rows = []
    method = EXP04_METHOD

    for layer_idx in range(num_layers):
        layer_name = "embedding_output" if layer_idx == 0 else f"transformer_layer_{layer_idx}"
        layer_dir_name = f"layer_{layer_idx:02d}"

        for seed_name in SEED_NAMES:
            run_dir = runs_base_dir / layer_dir_name / seed_name
            run_id = f"run_exp04_{method['method_short']}_layer{layer_idx:02d}_{seed_name}"

            for split in method["eval_splits"]:
                metrics_file = run_dir / _get_metrics_filename(method, split)
                if not metrics_file.exists():
                    continue

                with open(metrics_file) as f:
                    m = json.load(f)

                rows.append({
                    "experiment_id": method["experiment_id"],
                    "run_id": run_id,
                    "method": method["method_display"],
                    "model": method["model"],
                    "layer_index": layer_idx,
                    "layer_name": layer_name,
                    "seed_name": seed_name,
                    "split": split,
                    "num_samples": m["num_samples"],
                    "accuracy": m["accuracy"],
                    "macro_f1": m["macro_f1"],
                    "macro_precision": m.get("macro_precision", float("nan")),
                    "macro_recall": m.get("macro_recall", float("nan")),
                })

    return pd.DataFrame(rows)


def validate_exp04_layer_probe_completeness(
    run_metrics_df: pd.DataFrame,
    runs_base_dir: Path,
    num_layers: int,
) -> None:
    """Phase 4 layer probe completeness check."""
    expected_runs = num_layers * len(SEED_NAMES)
    actual_run_ids = set(run_metrics_df["run_id"].unique()) if len(run_metrics_df) > 0 else set()

    # Check run_id completeness
    expected_run_ids = set()
    method = EXP04_METHOD
    for layer_idx in range(num_layers):
        for seed_name in SEED_NAMES:
            expected_run_ids.add(
                f"run_exp04_{method['method_short']}_layer{layer_idx:02d}_{seed_name}"
            )

    missing_runs = expected_run_ids - actual_run_ids
    if missing_runs:
        raise RuntimeError(
            f"Layer probe completeness check failed: missing {len(missing_runs)} runs: "
            f"{sorted(missing_runs)[:10]}"
        )

    # Check split completeness per run
    expected_splits = set(method["eval_splits"])
    grouped = run_metrics_df.groupby(["layer_index", "seed_name"])
    for (layer_idx, seed_name), group in grouped:
        actual_splits = set(group["split"])
        if actual_splits != expected_splits:
            raise RuntimeError(
                f"Layer probe completeness check failed: layer_{layer_idx:02d}/{seed_name} "
                f"splits {actual_splits} != expected {expected_splits}"
            )

    # Check cost_metrics.json existence
    missing_cost = []
    for layer_idx in range(num_layers):
        for seed_name in SEED_NAMES:
            cost_path = runs_base_dir / f"layer_{layer_idx:02d}" / seed_name / "cost_metrics.json"
            if not cost_path.exists():
                missing_cost.append(f"layer_{layer_idx:02d}/{seed_name}")

    if missing_cost:
        raise RuntimeError(
            f"Layer probe cost files missing: {len(missing_cost)} cost_metrics.json "
            f"not found: {missing_cost[:10]}"
        )


def aggregate_exp04_layer_probe_summary(
    run_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate mean ± std by layer and split."""
    rows = []
    grouped = run_metrics_df.groupby(
        ["method", "model", "layer_index", "layer_name", "split"]
    )

    for (method, model, layer_index, layer_name, split), group in grouped:
        unique_ns = group["num_samples"].unique()
        nps = int(unique_ns[0]) if len(unique_ns) == 1 else int(group["num_samples"].median())

        rows.append({
            "method": method,
            "model": model,
            "layer_index": int(layer_index),
            "layer_name": layer_name,
            "split": split,
            "seed_count": len(group),
            "num_samples_per_seed": nps,
            "accuracy_mean": float(group["accuracy"].mean()),
            "accuracy_std": float(group["accuracy"].std(ddof=1)),
            "macro_f1_mean": float(group["macro_f1"].mean()),
            "macro_f1_std": float(group["macro_f1"].std(ddof=1)),
            "macro_precision_mean": float(group["macro_precision"].mean()),
            "macro_precision_std": float(group["macro_precision"].std(ddof=1)),
            "macro_recall_mean": float(group["macro_recall"].mean()),
            "macro_recall_std": float(group["macro_recall"].std(ddof=1)),
        })

    return pd.DataFrame(rows)


def collect_exp04_layer_probe_per_class_metrics(
    runs_base_dir: Path,
    num_layers: int,
) -> pd.DataFrame:
    """Collect Phase 4 layer probe per-class metrics."""
    rows = []
    method = EXP04_METHOD

    for layer_idx in range(num_layers):
        layer_name = "embedding_output" if layer_idx == 0 else f"transformer_layer_{layer_idx}"
        layer_dir_name = f"layer_{layer_idx:02d}"

        for seed_name in SEED_NAMES:
            run_dir = runs_base_dir / layer_dir_name / seed_name

            for split in method["eval_splits"]:
                pc_file = run_dir / _get_per_class_filename(method, split)
                if not pc_file.exists():
                    continue

                pc_df = pd.read_csv(pc_file)
                for _, row in pc_df.iterrows():
                    rows.append({
                        "method": method["method_display"],
                        "model": method["model"],
                        "layer_index": layer_idx,
                        "layer_name": layer_name,
                        "split": split,
                        "seed_name": seed_name,
                        "label_id": row["label_id"],
                        "label": row["label"],
                        "precision": row["precision"],
                        "recall": row["recall"],
                        "f1": row["f1"],
                        "support": row["support"],
                    })

    return pd.DataFrame(rows)


def aggregate_exp04_layer_probe_per_class_summary(
    per_class_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate per-class mean ± std by layer."""
    rows = []
    grouped = per_class_df.groupby(
        ["method", "model", "layer_index", "layer_name", "split", "label_id", "label"]
    )

    for (method, model, layer_index, layer_name, split, label_id, label), group in grouped:
        rows.append({
            "method": method,
            "model": model,
            "layer_index": int(layer_index),
            "layer_name": layer_name,
            "split": split,
            "label_id": int(label_id),
            "label": label,
            "f1_mean": float(group["f1"].mean()),
            "f1_std": float(group["f1"].std(ddof=1)),
            "precision_mean": float(group["precision"].mean()),
            "precision_std": float(group["precision"].std(ddof=1)),
            "recall_mean": float(group["recall"].mean()),
            "recall_std": float(group["recall"].std(ddof=1)),
        })

    return pd.DataFrame(rows)


def run_exp04_layer_probe_aggregation(
    runs_base_dir: Path,
    output_dir: Path,
    num_layers: int,
) -> None:
    """Phase 4 layer probe main aggregation entry point."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. run_metrics
    run_metrics_df = collect_exp04_layer_probe_run_metrics(runs_base_dir, num_layers)
    run_metrics_df.to_csv(
        output_dir / "exp04_layer_probe_run_metrics.csv", index=False,
    )
    print(f"  run_metrics: {len(run_metrics_df)} rows written")

    # 2. Completeness check
    validate_exp04_layer_probe_completeness(run_metrics_df, runs_base_dir, num_layers)
    print("  completeness check: PASSED")

    # 3. summary mean±std
    summary_df = aggregate_exp04_layer_probe_summary(run_metrics_df)
    summary_df.to_csv(
        output_dir / "exp04_layer_probe_summary_mean_std.csv", index=False,
    )
    print(f"  summary_mean_std: {len(summary_df)} rows written")

    # 4. per-class
    per_class_df = collect_exp04_layer_probe_per_class_metrics(runs_base_dir, num_layers)
    per_class_summary_df = aggregate_exp04_layer_probe_per_class_summary(per_class_df)
    per_class_summary_df.to_csv(
        output_dir / "exp04_layer_probe_per_class_summary.csv", index=False,
    )
    print(f"  per_class_summary: {len(per_class_summary_df)} rows written")

    print("  Phase 4 Layer Probe Aggregation: DONE")
