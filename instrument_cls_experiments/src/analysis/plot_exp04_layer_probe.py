"""Plot layer probe macro_f1 line charts with error bands."""

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
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = PROJECT_ROOT / "instrument_cls_experiments"

TABLES_DIR = EXP_ROOT / "reports/tables"
FIGURES_DIR = EXP_ROOT / "reports/figures"


def _load_analysis_cfg() -> dict:
    with open(EXP_ROOT / "configs/analysis/phase4_analysis_v1.yaml") as f:
        return yaml.safe_load(f)


def main() -> None:
    print("plot_exp04_layer_probe: plotting layer probe line charts")
    print("=" * 60)

    csv_path = TABLES_DIR / "exp04_layer_probe_summary_mean_std.csv"
    if not csv_path.exists():
        print(f"  ERROR: {csv_path} not found")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    cfg = _load_analysis_cfg()
    fig_defaults = cfg.get("figure_defaults", {})

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 两条折线合并到同一张图
    split_styles = [
        ("test", {"color": "#2196F3", "marker": "o", "label": "Test", "above": False}),
        ("external_test", {"color": "#FF9800", "marker": "s", "label": "External Test", "above": True}),
    ]

    max_layer = int(df["layer_index"].max())
    fig, ax = plt.subplots(figsize=(6, 5))

    for split, style in split_styles:
        split_df = df[df["split"] == split].sort_values("layer_index")
        if split_df.empty:
            continue

        x = split_df["layer_index"].values
        y = split_df["macro_f1_mean"].values
        yerr = split_df["macro_f1_std"].values

        ax.plot(x, y, marker=style["marker"], color=style["color"],
                label=style["label"], linewidth=2, markersize=6)
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=style["color"])

        # annotate best layer
        best_idx = np.argmax(y)
        _y_off = 28 if style["above"] else -20
        _va = "bottom" if style["above"] else "top"
        ax.annotate(
            f"L{x[best_idx]}: {y[best_idx]:.3f}",
            xy=(x[best_idx], y[best_idx]),
            xytext=(0, _y_off), textcoords="offset points",
            ha="center", va=_va, fontsize=10,
            arrowprops=dict(arrowstyle="->", color=style["color"], lw=1),
            color=style["color"], fontweight="bold",
        )

    ax.set_xlabel("MERT Layer Index",
                  fontsize=fig_defaults.get("label_fontsize", 12))
    ax.set_ylabel("Macro F1 (mean ± std)",
                  fontsize=fig_defaults.get("label_fontsize", 12))
    ax.set_title("MERT Layer-wise Probing",
                 fontsize=fig_defaults.get("title_fontsize", 13), pad=10)

    ax.set_xticks(range(max_layer + 1))
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path = FIGURES_DIR / "exp04_layer_probe_macro_f1.pdf"
    fig.savefig(output_path, dpi=fig_defaults.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.relative_to(EXP_ROOT)}")

    print("=" * 60)
    print("plot_exp04_layer_probe: done")


if __name__ == "__main__":
    main()
