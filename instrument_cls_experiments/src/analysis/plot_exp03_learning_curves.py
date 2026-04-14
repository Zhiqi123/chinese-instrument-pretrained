"""Plot Phase 3 learning curves and cost curves."""

from __future__ import annotations

import sys
from pathlib import Path

import os as _os
# fix matplotlib cache dir to avoid permission issues
_os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[3] / "instrument_cls_experiments/artifacts/.mpl_cache"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = PROJECT_ROOT / "instrument_cls_experiments"

REPORTS_DIR = EXP_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"

# method plot styles
METHOD_STYLES = {
    "MERT linear probe": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "MERT+LoRA (r=4)": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
    "MERT+LoRA (r=8)": {"color": "#2ca02c", "marker": "^", "linestyle": "--"},
    "MERT full fine-tuning": {"color": "#d62728", "marker": "D", "linestyle": "-."},
}

# fixed method order
METHOD_ORDER = [
    "MERT linear probe",
    "MERT+LoRA (r=4)",
    "MERT+LoRA (r=8)",
    "MERT full fine-tuning",
]

RATIO_ORDER = [0.10, 0.25, 0.50, 1.00]


def _setup_figure(title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(RATIO_ORDER)
    ax.set_xticklabels(["10%", "25%", "50%", "100%"])
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_learning_curve(
    df: pd.DataFrame,
    split: str,
    metric_mean: str,
    metric_std: str,
    title: str,
    ylabel: str,
    output_path: Path,
):
    """Plot a single learning curve."""
    fig, ax = _setup_figure(title, "Train Data Ratio", ylabel)

    split_df = df[df["split"] == split]

    for method in METHOD_ORDER:
        method_df = split_df[split_df["method"] == method].sort_values("ratio_value")
        if method_df.empty:
            continue

        style = METHOD_STYLES.get(method, {})
        x = method_df["ratio_value"].values
        y = method_df[metric_mean].values
        yerr = method_df[metric_std].values

        ax.errorbar(
            x, y, yerr=yerr,
            label=method,
            color=style.get("color", "black"),
            marker=style.get("marker", "o"),
            linestyle=style.get("linestyle", "-"),
            linewidth=1.5,
            markersize=6,
            capsize=3,
        )

    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_cost_curve(
    df: pd.DataFrame,
    output_path: Path,
):
    """Plot cost curve (total_method_time_sec)."""
    fig, ax = _setup_figure(
        "Training Cost vs. Data Ratio",
        "Train Data Ratio",
        "Total Method Time (sec)",
    )

    # cost data is split-independent; use test rows (one cost row per method × ratio)
    cost_df = df[df["split"] == "test"].copy()

    for method in METHOD_ORDER:
        method_df = cost_df[cost_df["method"] == method].sort_values("ratio_value")
        if method_df.empty:
            continue

        style = METHOD_STYLES.get(method, {})
        x = method_df["ratio_value"].values
        y = method_df["total_method_time_sec_mean"].values
        yerr = method_df["total_method_time_sec_std"].values

        ax.errorbar(
            x, y, yerr=yerr,
            label=method,
            color=style.get("color", "black"),
            marker=style.get("marker", "o"),
            linestyle=style.get("linestyle", "-"),
            linewidth=1.5,
            markersize=6,
            capsize=3,
        )

    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def main():
    print("Phase 3 — Learning Curve Plots")
    print("=" * 60)

    lc_path = TABLES_DIR / "exp03_learning_curve_points.csv"
    if not lc_path.exists():
        print(f"  ERROR: {lc_path} not found")
        sys.exit(1)

    df = pd.read_csv(lc_path)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Test Macro-F1 learning curve
    plot_learning_curve(
        df, "test", "macro_f1_mean", "macro_f1_std",
        "Data Efficiency: Test Macro-F1",
        "Macro-F1",
        FIGURES_DIR / "exp03_learning_curve_test_macro_f1.pdf",
    )

    # 2. External Test Macro-F1 learning curve
    plot_learning_curve(
        df, "external_test", "macro_f1_mean", "macro_f1_std",
        "Data Efficiency: External Test Macro-F1",
        "Macro-F1",
        FIGURES_DIR / "exp03_learning_curve_external_test_macro_f1.pdf",
    )

    # 3. Cost curve
    plot_cost_curve(
        df,
        FIGURES_DIR / "exp03_cost_curve_total_method_time.pdf",
    )

    print("\nPhase 3 Plots: DONE")


if __name__ == "__main__":
    main()
