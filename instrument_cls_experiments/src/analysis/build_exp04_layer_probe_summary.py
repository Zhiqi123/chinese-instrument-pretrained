"""Aggregate layer probe run results into summary CSVs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = PROJECT_ROOT / "instrument_cls_experiments"
sys.path.insert(0, str(EXP_ROOT))

RUNS_DIR = EXP_ROOT / "runs/exp04_layer_probe"
TABLES_DIR = EXP_ROOT / "reports/tables"


def _detect_num_layers(runs_dir: Path) -> int:
    """Detect number of layers from run directory structure."""
    layer_dirs = sorted(
        d for d in runs_dir.iterdir()
        if d.is_dir() and d.name.startswith("layer_")
    )
    if not layer_dirs:
        # try detecting from cache NPZ
        cache_dir = EXP_ROOT / "artifacts/exp04_analysis/layer_embeddings"
        for seed in ["seed0", "seed1", "seed2"]:
            npz_path = cache_dir / seed / "train_all_layers.npz"
            if npz_path.exists():
                data = np.load(npz_path, allow_pickle=True)
                n = len([k for k in data.keys() if k.startswith("layer_")])
                data.close()
                return n
        raise FileNotFoundError(
            f"Cannot detect layer count: no layer_XX dirs under {runs_dir}, "
            "and no NPZ cache found"
        )
    return len(layer_dirs)


def main() -> None:
    print("build_exp04_layer_probe_summary: aggregating layer probe results")
    print("=" * 60)

    if not RUNS_DIR.exists():
        print(f"  ERROR: {RUNS_DIR.relative_to(EXP_ROOT)} not found")
        sys.exit(1)

    num_layers = _detect_num_layers(RUNS_DIR)
    print(f"  Detected {num_layers} layers")

    from src.eval.summary_aggregator import run_exp04_layer_probe_aggregation

    run_exp04_layer_probe_aggregation(
        runs_base_dir=RUNS_DIR,
        output_dir=TABLES_DIR,
        num_layers=num_layers,
    )

    # print best layer summary
    import pandas as pd
    summary_path = TABLES_DIR / "exp04_layer_probe_summary_mean_std.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        print("-" * 60)
        for split in ["test", "external_test"]:
            split_df = summary[summary["split"] == split]
            if split_df.empty:
                continue
            best_row = split_df.loc[split_df["macro_f1_mean"].idxmax()]
            print(f"  {split}: best layer={int(best_row['layer_index'])}, "
                  f"macro_f1={best_row['macro_f1_mean']:.4f}"
                  f"±{best_row['macro_f1_std']:.4f}")

    print("=" * 60)
    print("build_exp04_layer_probe_summary: done")


if __name__ == "__main__":
    main()
