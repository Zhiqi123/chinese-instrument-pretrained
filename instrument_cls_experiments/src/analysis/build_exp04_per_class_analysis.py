"""Merge Phase 1/2 per-class summaries into cross-method comparison table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = PROJECT_ROOT / "instrument_cls_experiments"

TABLES_DIR = EXP_ROOT / "reports/tables"


def _load_analysis_cfg() -> dict:
    with open(EXP_ROOT / "configs/analysis/phase4_analysis_v1.yaml") as f:
        return yaml.safe_load(f)


def _load_manifest() -> dict:
    with open(EXP_ROOT / "artifacts/exp04_analysis/phase4_selection_manifest.json") as f:
        return json.load(f)


def main() -> None:
    print("build_exp04_per_class_analysis: merging Phase 1/2 per-class tables")
    print("=" * 60)

    cfg = _load_analysis_cfg()
    manifest = _load_manifest()

    # load Phase 1/2 per-class tables
    p1_path = TABLES_DIR / "exp01_transfer_per_class_summary.csv"
    p2_path = TABLES_DIR / "exp02_adaptation_per_class_summary.csv"

    if not p1_path.exists():
        print(f"  ERROR: {p1_path} not found")
        sys.exit(1)
    if not p2_path.exists():
        print(f"  ERROR: {p2_path} not found")
        sys.exit(1)

    p1_df = pd.read_csv(p1_path)
    p2_df = pd.read_csv(p2_path)

    # tag source phase
    p1_df["source_phase"] = 1
    p2_df["source_phase"] = 2

    # merge
    combined = pd.concat([p1_df, p2_df], ignore_index=True)

    # keep only test & external_test
    combined = combined[combined["split"].isin(["test", "external_test"])].reset_index(drop=True)

    # add method_short via reverse mapping
    display_to_short = {}
    for short, display in cfg["method_display_map"].items():
        display_to_short[display] = short

    combined["method_short"] = combined["method"].map(display_to_short)

    # check for unmapped methods
    unmapped = combined[combined["method_short"].isna()]["method"].unique()
    if len(unmapped) > 0:
        print(f"  [WARN] unmapped methods: {unmapped.tolist()}")

    # add best method flags
    best_phase2 = manifest["best_phase2_method_display"]
    best_lora = manifest["best_lora_method_display"]
    combined["is_best_phase2"] = combined["method"] == best_phase2
    combined["is_best_lora"] = combined["method"] == best_lora

    # sort by source_phase → method_short → split → label_id
    method_order = cfg["method_order"]
    # Phase 1 method ordering
    p1_order = list(cfg.get("phase1_method_short_to_display", {}).keys())
    full_order = p1_order + [m for m in method_order if m not in p1_order]
    order_map = {m: i for i, m in enumerate(full_order)}

    combined["method_rank"] = combined["method_short"].map(order_map).fillna(99)
    combined = combined.sort_values(
        by=["source_phase", "method_rank", "split", "label_id"],
        ascending=[True, True, True, True],
    ).drop(columns=["method_rank"])

    # add per-class ranking
    for split in combined["split"].unique():
        split_mask = combined["split"] == split
        for label_id in combined["label_id"].unique():
            mask = split_mask & (combined["label_id"] == label_id)
            if not mask.any():
                continue
            sub = combined.loc[mask].copy()
            sub = sub.sort_values("f1_mean", ascending=False)
            combined.loc[sub.index, "rank_in_split_class_by_f1_mean"] = range(1, len(sub) + 1)
            combined.loc[sub.index, "is_best_in_class"] = False
            combined.loc[sub.index[0], "is_best_in_class"] = True

    combined["rank_in_split_class_by_f1_mean"] = combined["rank_in_split_class_by_f1_mean"].astype(int)

    # write separate Phase 1 / Phase 2 files
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # source table columns + source_phase
    source_cols = [
        "source_phase", "method", "model", "split", "label_id", "label",
        "f1_mean", "f1_std", "precision_mean", "precision_std",
        "recall_mean", "recall_std",
    ]

    p1_out = combined[combined["source_phase"] == 1][source_cols].copy()
    p1_out_path = TABLES_DIR / "exp04_per_class_phase1_summary.csv"
    p1_out.to_csv(p1_out_path, index=False)
    print(f"  wrote: {p1_out_path.relative_to(EXP_ROOT)} ({len(p1_out)} rows)")

    p2_out = combined[combined["source_phase"] == 2][source_cols].copy()
    p2_out_path = TABLES_DIR / "exp04_per_class_phase2_summary.csv"
    p2_out.to_csv(p2_out_path, index=False)
    print(f"  wrote: {p2_out_path.relative_to(EXP_ROOT)} ({len(p2_out)} rows)")

    # write combined output
    output_path = TABLES_DIR / "exp04_per_class_all_methods.csv"
    combined.to_csv(output_path, index=False)
    print(f"  wrote: {output_path.relative_to(EXP_ROOT)}")
    print(f"  rows: {len(combined)}, methods: {combined['method'].nunique()}, "
          f"splits: {combined['split'].nunique()}")

    # summary stats
    print("-" * 60)
    print("  Per-class F1 summary (test split):")
    test_df = combined[combined["split"] == "test"]
    for method in test_df["method"].unique():
        m_df = test_df[test_df["method"] == method]
        mean_f1 = m_df["f1_mean"].mean()
        min_f1 = m_df["f1_mean"].min()
        min_class = m_df.loc[m_df["f1_mean"].idxmin(), "label"]
        marker = ""
        if method == best_phase2:
            marker = " ★best_phase2"
        elif method == best_lora:
            marker = " ★best_lora"
        print(f"    {method}: avg_f1={mean_f1:.4f}, "
              f"weakest={min_class}({min_f1:.4f}){marker}")

    print("=" * 60)
    print("build_exp04_per_class_analysis: done")


if __name__ == "__main__":
    main()
