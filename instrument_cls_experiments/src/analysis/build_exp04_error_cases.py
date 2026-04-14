"""Extract and rank high-confidence error cases from best Phase 2 method."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = PROJECT_ROOT / "instrument_cls_experiments"

TABLES_DIR = EXP_ROOT / "reports/tables"
ARTIFACTS_DIR = EXP_ROOT / "artifacts/exp04_analysis"


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def select_error_cases(
    predictions_df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Select top-K error cases, prioritizing distinct (true, pred) pairs."""
    errors = predictions_df[
        predictions_df["true_label"] != predictions_df["pred_label"]
    ].copy()

    if errors.empty:
        return errors

    errors = errors.sort_values("top1_score", ascending=False)
    errors["error_pair"] = errors["true_label"] + " → " + errors["pred_label"]

    # prioritize distinct error_pair coverage
    selected_indices = []
    seen_pairs = set()

    for idx, row in errors.iterrows():
        if len(selected_indices) >= top_k:
            break
        pair = row["error_pair"]
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            selected_indices.append(idx)

    # fill remaining slots by descending top1_score
    remaining = errors.drop(index=selected_indices)
    for idx in remaining.index:
        if len(selected_indices) >= top_k:
            break
        selected_indices.append(idx)

    selected = errors.loc[selected_indices].copy()
    selected["case_rank"] = range(1, len(selected) + 1)

    return selected


def enrich_with_metadata(
    cases_df: pd.DataFrame,
    segment_manifest_path: Path,
    split: str,
    dataset_cfg: dict,
    seed_name: str,
) -> pd.DataFrame:
    """Enrich error cases with segment manifest metadata."""
    if cases_df.empty:
        return cases_df

    seg_manifest = pd.read_csv(segment_manifest_path)
    seg_manifest = seg_manifest[seg_manifest["split"] == split]

    joined = cases_df.merge(
        seg_manifest[["segment_id", "segment_path", "source_dataset",
                       "start_time_sec", "end_time_sec"]],
        on="segment_id",
        how="left",
        suffixes=("", "_manifest"),
    )

    # check join success rate
    missing = joined["segment_path"].isna().sum()
    if missing > 0:
        print(f"  WARNING: {missing}/{len(joined)} cases failed to join segment_manifest")

    # rename source_dataset → dataset_name
    if "source_dataset" in joined.columns:
        joined["dataset_name"] = joined["source_dataset"]

    # compute segment_abs_path
    segment_root_template = dataset_cfg.get("segment_root_template", "")
    seed_index = {"seed0": 0, "seed1": 1, "seed2": 2}[seed_name]
    segment_root = segment_root_template.format(split_seed=seed_index)

    def _compute_abs_path(row):
        if pd.isna(row.get("segment_path")):
            return ""
        return str(PROJECT_ROOT / segment_root / row["segment_path"])

    joined["segment_abs_path"] = joined.apply(_compute_abs_path, axis=1)

    return joined


def main() -> None:
    print("build_exp04_error_cases: error case analysis")
    print("=" * 60)

    manifest = _load_json(ARTIFACTS_DIR / "phase4_selection_manifest.json")
    analysis_cfg = _load_yaml(EXP_ROOT / "configs/analysis/phase4_analysis_v1.yaml")
    dataset_cfg = _load_yaml(EXP_ROOT / "configs/data/dataset_v1.yaml")

    seed_name = manifest["qualitative_seed_name"]
    split = manifest["error_case_split"]
    top_k = analysis_cfg["error_case_top_k"]
    method_short = manifest["best_phase2_method_short"]
    method_display = manifest["best_phase2_method_display"]

    print(f"  Method: {method_display} ({method_short})")
    print(f"  Seed: {seed_name}, Split: {split}, Top-K: {top_k}")

    # load predictions
    pred_path = (EXP_ROOT / "runs/exp02_adaptation" / method_short /
                 seed_name / f"{split}_predictions.csv")
    if not pred_path.exists():
        print(f"  ERROR: {pred_path} not found")
        sys.exit(1)

    df = pd.read_csv(pred_path)
    print(f"  Total predictions: {len(df)}")

    # filter error predictions
    errors_all = df[df["true_label"] != df["pred_label"]].copy()
    print(f"  Errors: {len(errors_all)} ({len(errors_all)/len(df)*100:.1f}%)")

    if errors_all.empty:
        print("  No errors found — writing empty tables")
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(TABLES_DIR / "exp04_error_cases.csv", index=False)
        pd.DataFrame().to_csv(TABLES_DIR / "exp04_error_case_summary.csv", index=False)
        return

    # select top-K cases
    selected = select_error_cases(df, top_k)
    print(f"  Selected top-{top_k}: {len(selected)} cases")

    # enrich with metadata
    manifest_template = dataset_cfg["manifest_paths"]["segment_manifest_template"]
    seed_index = {"seed0": 0, "seed1": 1, "seed2": 2}[seed_name]
    seg_manifest_path = PROJECT_ROOT / manifest_template.format(seed_index=seed_index)

    if seg_manifest_path.exists():
        selected = enrich_with_metadata(
            selected, seg_manifest_path, split, dataset_cfg, seed_name,
        )
    else:
        print(f"  WARNING: segment_manifest not found: {seg_manifest_path}")
        selected["segment_abs_path"] = ""
        selected["dataset_name"] = ""

    # add metadata columns
    selected["method_short"] = method_short
    selected["method_display"] = method_display
    selected["source_phase"] = 2
    selected["seed_name"] = seed_name

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # output column order
    output_cols = [
        "case_rank",
        "source_phase",
        "method_short",
        "method_display",
        "seed_name",
        "split",
        "segment_id",
        "segment_abs_path",
        "dataset_name",
        "true_label",
        "pred_label",
        "top1_score",
        "error_pair",
    ]
    # keep only existing columns
    output_cols = [c for c in output_cols if c in selected.columns]
    selected[output_cols].to_csv(TABLES_DIR / "exp04_error_cases.csv", index=False)
    print(f"  wrote: exp04_error_cases.csv ({len(selected)} rows)")

    # group by (true_label, pred_label) using all errors, not just top-K
    errors_all["error_pair"] = errors_all["true_label"] + " → " + errors_all["pred_label"]
    summary = (
        errors_all.groupby(["true_label", "pred_label", "error_pair"])
        .agg(
            count=("segment_id", "count"),
            mean_score=("top1_score", "mean"),
            min_score=("top1_score", "min"),
            max_score=("top1_score", "max"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )
    summary["method_short"] = method_short
    summary["method_display"] = method_display
    summary["source_phase"] = 2
    summary["seed_name"] = seed_name
    summary["split"] = split

    summary.to_csv(TABLES_DIR / "exp04_error_case_summary.csv", index=False)
    print(f"  wrote: exp04_error_case_summary.csv ({len(summary)} rows)")

    # top-K most frequent confusion pairs
    print(f"\n  Top-{top_k} confusion pairs:")
    for _, row in summary.head(top_k).iterrows():
        print(f"    {row['true_label']} → {row['pred_label']}: "
              f"{row['count']} errors, mean_score={row['mean_score']:.4f}")

    print("=" * 60)
    print("build_exp04_error_cases: done")


if __name__ == "__main__":
    main()
