"""Plot UMAP 1x3 panel visualization colored by instrument class."""

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
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# CJK font fallback chain
_CJK_FONT_CANDIDATES = [
    "Hiragino Sans GB", "PingFang SC", "Heiti TC",
    "WenQuanYi Micro Hei", "Noto Sans CJK SC", "Arial Unicode MS",
]

def _setup_cjk_font() -> None:
    available = {f.name for f in fm.fontManager.ttflist}
    for font_name in _CJK_FONT_CANDIDATES:
        if font_name in available:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return

_setup_cjk_font()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = PROJECT_ROOT / "instrument_cls_experiments"

TABLES_DIR = EXP_ROOT / "reports/tables"
FIGURES_DIR = EXP_ROOT / "reports/figures"

# class order (matches dataset_v1.yaml); CSV uses Chinese, plot shows English
CLASS_ORDER_ZH = ["二胡", "琵琶", "中阮", "笛子", "唢呐", "笙"]
CLASS_ORDER_EN = ["Erhu", "Pipa", "Zhongruan", "Dizi", "Suona", "Sheng"]
ZH_TO_EN = dict(zip(CLASS_ORDER_ZH, CLASS_ORDER_EN))

# panel order: frozen_mert → best_lora → mert_full_ft
# panel titles read directly from CSV method_display


def _load_analysis_cfg() -> dict:
    with open(EXP_ROOT / "configs/analysis/phase4_analysis_v1.yaml") as f:
        return yaml.safe_load(f)


def main() -> None:
    print("plot_exp04_umap: UMAP 1x3 panel visualization")
    print("=" * 60)

    cfg = _load_analysis_cfg()
    fig_defaults = cfg.get("figure_defaults", {})
    seed_name = cfg["qualitative_seed_name"]
    split = cfg["umap_split"]

    csv_path = TABLES_DIR / "exp04_umap_points.csv"
    if not csv_path.exists():
        print(f"  ERROR: {csv_path} not found, run build_exp04_umap_points.py first")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"  loaded: {len(df)} rows")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # determine 3 panel method_short order
    # frozen_mert → best_lora → mert_full_ft
    unique_methods = df["method_short"].unique().tolist()
    # panel order: frozen_mert, best_lora (middle), mert_full_ft
    panel_order = []
    if "frozen_mert" in unique_methods:
        panel_order.append("frozen_mert")
    for ms in unique_methods:
        if ms not in ("frozen_mert", "mert_full_ft"):
            panel_order.append(ms)
    if "mert_full_ft" in unique_methods:
        panel_order.append("mert_full_ft")

    # 1x3 panel
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, len(CLASS_ORDER_ZH)))
    color_map = {label: colors[i] for i, label in enumerate(CLASS_ORDER_ZH)}

    for ax_idx, method_short in enumerate(panel_order):
        ax = axes[ax_idx]
        sub = df[df["method_short"] == method_short]
        if sub.empty:
            ax.set_title(f"({method_short}: no data)")
            continue

        # panel title from CSV method_display
        display_name = sub["method_display"].iloc[0]

        for label in CLASS_ORDER_ZH:
            mask = sub["label"] == label
            if not mask.any():
                continue
            ax.scatter(
                sub.loc[mask, "umap_x"],
                sub.loc[mask, "umap_y"],
                c=[color_map[label]],
                label=ZH_TO_EN[label],
                s=20, alpha=0.7, edgecolors="none",
            )

        ax.set_title(display_name,
                     fontsize=fig_defaults.get("title_fontsize", 13), pad=10)
        ax.set_xlabel("UMAP-1", fontsize=fig_defaults.get("label_fontsize", 11))
        if ax_idx == 0:
            ax.set_ylabel("UMAP-2", fontsize=fig_defaults.get("label_fontsize", 11))
        ax.grid(True, alpha=0.2)

    # shared legend (rightmost panel only)
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(
            handles, labels,
            fontsize=fig_defaults.get("legend_fontsize", 9),
            loc="best", markerscale=2,
        )

    fig.suptitle(f"UMAP Embedding Visualization ({split}, {seed_name})",
                 fontsize=fig_defaults.get("title_fontsize", 13) + 1)
    fig.tight_layout(rect=[0, 0, 1, 1])

    output_path = FIGURES_DIR / f"exp04_umap_{seed_name}_{split}.pdf"
    fig.savefig(output_path, dpi=fig_defaults.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.relative_to(EXP_ROOT)}")

    print("=" * 60)
    print("plot_exp04_umap: done")


if __name__ == "__main__":
    main()
