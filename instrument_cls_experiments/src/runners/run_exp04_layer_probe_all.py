"""
Phase 4 layer probe orchestrator.
Runs gate check, cache build, per-layer probe training, aggregation, and plots.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = PROJECT_ROOT / "instrument_cls_experiments"
sys.path.insert(0, str(EXP_ROOT))

CONFIG_PATH = "configs/experiments/exp04_layer_probe/mert_layer_probe_v1.yaml"
CACHE_BUILDER = "src/runners/run_exp04_layer_cache_build.py"
PROBE_RUNNER = "src/runners/run_exp04_layer_probe.py"
SEED_NAMES = ["seed0", "seed1", "seed2"]


def _run(cmd: list[str], label: str) -> None:
    """Run subprocess; exit(1) on failure."""
    print(f"\n{'=' * 60}")
    print(f"[{label}]")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  FAILED: {label}")
        sys.exit(1)
    print(f"  PASSED: {label}")


def _detect_num_layers(cache_dir: Path) -> int:
    """Detect number of layers from first available NPZ file."""
    for seed in SEED_NAMES:
        npz_path = cache_dir / seed / "train_all_layers.npz"
        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=True)
            layer_keys = [k for k in data.keys() if k.startswith("layer_")]
            data.close()
            return len(layer_keys)
    raise FileNotFoundError("No train_all_layers.npz found for any seed")


def main():
    print("Phase 4 — MERT Layer Probe (full pipeline)")
    print("=" * 60)

    # 0. Re-run gate check (refresh manifest)
    _run(
        [sys.executable, str(EXP_ROOT / "src/runners/run_phase4_gate_check.py")],
        "Phase 4 Gate Check (refresh manifest)",
    )

    # 1. Layer cache build (3 seeds)
    for seed_name in SEED_NAMES:
        _run(
            [
                sys.executable, str(EXP_ROOT / CACHE_BUILDER),
                "--config", str(EXP_ROOT / CONFIG_PATH),
                "--seed-name", seed_name,
            ],
            f"Cache Build — {seed_name}",
        )

    # 2. Detect layer count
    import yaml
    with open(EXP_ROOT / CONFIG_PATH) as f:
        method_cfg = yaml.safe_load(f)
    cache_dir = EXP_ROOT / method_cfg["output"]["cache_dir"]
    num_layers = _detect_num_layers(cache_dir)
    print(f"\n  Detected {num_layers} layers from NPZ cache")

    # 3. Per-layer per-seed training (fail-fast)
    for layer_idx in range(num_layers):
        for seed_name in SEED_NAMES:
            _run(
                [
                    sys.executable, str(EXP_ROOT / PROBE_RUNNER),
                    "--config", str(EXP_ROOT / CONFIG_PATH),
                    "--seed-name", seed_name,
                    "--layer-index", str(layer_idx),
                ],
                f"Probe — layer_{layer_idx:02d}/{seed_name}",
            )

    # 4. Aggregation
    _run(
        [sys.executable, str(EXP_ROOT / "src/analysis/build_exp04_layer_probe_summary.py")],
        "Layer Probe Aggregation",
    )

    # 5. Plots
    _run(
        [sys.executable, str(EXP_ROOT / "src/analysis/plot_exp04_layer_probe.py")],
        "Layer Probe Plots",
    )

    print("\n" + "=" * 60)
    print(f"Phase 4 Layer Probe: ALL {num_layers} layers × {len(SEED_NAMES)} seeds COMPLETED")
    print(f"  Total runs: {num_layers * len(SEED_NAMES)}")


if __name__ == "__main__":
    main()
