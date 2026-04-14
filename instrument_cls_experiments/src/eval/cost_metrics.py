"""Training cost metrics: parameter counts, wall time, peak memory."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def count_parameters(model: torch.nn.Module) -> dict:
    """Count total and trainable parameters.

    Returns:
        {total_params, trainable_params, trainable_ratio}.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable / total if total > 0 else 0.0
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_ratio": ratio,
    }


def reset_peak_memory() -> None:
    """Reset CUDA peak memory stats before training."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def read_peak_memory() -> tuple[float | None, bool]:
    """Read peak memory after training.

    Returns:
        (peak_mb_or_None, is_available)
    """
    if torch.cuda.is_available():
        peak_bytes = torch.cuda.max_memory_allocated()
        peak_mb = peak_bytes / (1024 ** 2)
        return peak_mb, True
    return None, False


def build_cost_metrics(
    method: str,
    model_name: str,
    seed_name: str,
    device_type: str,
    param_stats: dict,
    feature_prep_time_sec: float,
    train_wall_time_sec: float,
    epoch_times: list[float],
    step_times_ms: list[float],
    peak_memory: tuple[float | None, bool],
    best_epoch: int,
    stopped_epoch: int,
) -> dict:
    """Assemble the cost_metrics.json dict."""
    peak_mb, mem_available = peak_memory

    mean_epoch_time = float(np.mean(epoch_times)) if epoch_times else 0.0
    mean_step_time = float(np.mean(step_times_ms)) if step_times_ms else 0.0

    return {
        "method": method,
        "model": model_name,
        "seed_name": seed_name,
        "device_type": device_type,
        "total_params": param_stats["total_params"],
        "trainable_params": param_stats["trainable_params"],
        "trainable_ratio": param_stats["trainable_ratio"],
        "feature_prep_time_sec": round(feature_prep_time_sec, 3),
        "train_wall_time_sec": round(train_wall_time_sec, 3),
        "total_method_time_sec": round(feature_prep_time_sec + train_wall_time_sec, 3),
        "mean_epoch_time_sec": round(mean_epoch_time, 3),
        "mean_train_step_time_ms": round(mean_step_time, 3),
        "peak_device_memory_mb": round(peak_mb, 2) if peak_mb is not None else None,
        "memory_metric_available": mem_available,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
    }


def write_cost_metrics(cost_dict: dict, output_path: Path) -> None:
    """Write cost_metrics.json."""
    with open(output_path, "w") as f:
        json.dump(cost_dict, f, indent=2, ensure_ascii=False)
