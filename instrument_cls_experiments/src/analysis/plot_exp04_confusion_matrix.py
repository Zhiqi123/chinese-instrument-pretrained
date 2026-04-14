"""Plot Phase 4 confusion matrix heatmaps for best method."""

from __future__ import annotations

import json
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

# CJK font fallback chain (macOS → Linux)
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

# class order: CSV uses Chinese labels, plot shows English
CLASS_ORDER_ZH = ["二胡", "琵琶", "中阮", "笛子", "唢呐", "笙"]
CLASS_ORDER_EN = ["Erhu", "Pipa", "Zhongruan", "Dizi", "Suona", "Sheng"]


def _load_analysis_cfg() -> dict:
    with open(EXP_ROOT / "configs/analysis/phase4_analysis_v1.yaml") as f:
        return yaml.safe_load(f)


def _load_manifest() -> dict:
    with open(EXP_ROOT / "artifacts/exp04_analysis/phase4_selection_manifest.json") as f:
        return json.load(f)


def build_cm_from_count_mean(
    df: pd.DataFrame,
    method: str,
    split: str,
    class_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build row-normalized confusion matrix from count_mean values."""
    n = len(class_names)
    cm_count = np.zeros((n, n))
    cm_rate_std = np.zeros((n, n))

    sub = df[(df["method"] == method) & (df["split"] == split)]

    for _, row in sub.iterrows():
        try:
            i = class_names.index(row["true_label"])
            j = class_names.index(row["pred_label"])
        except ValueError:
            continue
        cm_count[i, j] = row["count_mean"]
        cm_rate_std[i, j] = row["row_rate_std"]

    # row normalize
    cm_norm = np.zeros_like(cm_count)
    for i in range(n):
        row_sum = cm_count[i].sum()
        if row_sum > 0:
            cm_norm[i] = cm_count[i] / row_sum

    return cm_norm, cm_rate_std


def plot_single_panel(
    ax,
    cm_norm: np.ndarray,
    cm_std: np.ndarray,
    class_names: list[str],
    title: str,
    fig_defaults: dict,
) -> matplotlib.image.AxesImage:
    """Draw a single confusion matrix panel on the given axes."""
    n = len(class_names)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")

    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            std = cm_std[i, j]
            if i == j:
                text = f"{val:.2f}\n±{std:.2f}"
            elif val >= 0.01:
                text = f"{val:.2f}"
            elif val == 0:
                text = "0"
            else:
                text = "<.01"

            color = "white" if val > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, fontsize=fig_defaults.get("tick_fontsize", 10))
    ax.set_yticklabels(class_names, fontsize=fig_defaults.get("tick_fontsize", 10))
    ax.set_xlabel("Predicted", fontsize=fig_defaults.get("label_fontsize", 11))
    ax.set_ylabel("True", fontsize=fig_defaults.get("label_fontsize", 11))
    ax.set_title(title, fontsize=fig_defaults.get("title_fontsize", 13), pad=10)

    return im


def main() -> None:
    print("plot_exp04_confusion_matrix: plotting best method 1x2 confusion matrix")
    print("=" * 60)

    cfg = _load_analysis_cfg()
    manifest = _load_manifest()
    fig_defaults = cfg.get("figure_defaults", {})

    best_display = manifest["best_phase2_method_display"]
    best_short = manifest["best_phase2_method_short"]

    mean_std_path = TABLES_DIR / "exp04_confusion_mean_std.csv"
    if not mean_std_path.exists():
        print(f"  ERROR: {mean_std_path} not found, run build_exp04_confusion_summary.py first")
        sys.exit(1)

    df = pd.read_csv(mean_std_path)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Best method: {best_display} ({best_short})")

    # 1x2 panel: test / external_test
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    splits = ["test", "external_test"]
    split_titles = {
        "test": f"{best_display} — Test",
        "external_test": f"{best_display} — External Test",
    }

    for ax_idx, split in enumerate(splits):
        cm_norm, cm_std = build_cm_from_count_mean(df, best_display, split, CLASS_ORDER_ZH)
        plot_single_panel(
            axes[ax_idx], cm_norm, cm_std, CLASS_ORDER_EN,
            title=split_titles[split], fig_defaults=fig_defaults,
        )

    fig.tight_layout()

    output_path = FIGURES_DIR / "exp04_confusion_best_method.pdf"
    fig.savefig(output_path, dpi=fig_defaults.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.relative_to(EXP_ROOT)}")

    # single panel: MERT full fine-tuning external_test (Fig. 9)
    print("  plotting MERT full fine-tuning external_test single heatmap ...")
    fig9, ax9 = plt.subplots(figsize=(5.5, 5))
    cm_norm_ft, cm_std_ft = build_cm_from_count_mean(
        df, "MERT full fine-tuning", "external_test", CLASS_ORDER_ZH,
    )
    plot_single_panel(
        ax9, cm_norm_ft, cm_std_ft, CLASS_ORDER_EN,
        title="MERT Full Fine-tuning — External Test",
        fig_defaults=fig_defaults,
    )
    fig9.tight_layout()
    output_fig9 = FIGURES_DIR / "exp04_confusion_mert_full_ft_external_test.pdf"
    fig9.savefig(output_fig9, dpi=fig_defaults.get("dpi", 150), bbox_inches="tight")
    plt.close(fig9)
    print(f"  Saved: {output_fig9.relative_to(EXP_ROOT)}")

    print("=" * 60)
    print("plot_exp04_confusion_matrix: done")


if __name__ == "__main__":
    main()
