"""
Phase 3 train subset builder.
Builds 12 stratified train subsets (4 ratios x 3 seeds), writes files, and runs audit.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = PROJECT_ROOT / "instrument_cls_experiments"
sys.path.insert(0, str(EXP_ROOT))

from src.data.manifest_loader import load_seed_data

DATASET_CFG = EXP_ROOT / "configs/data/dataset_v1.yaml"
SUBSAMPLE_CFG = EXP_ROOT / "configs/data/phase3_subsample_v1.yaml"
SUBSETS_DIR = EXP_ROOT / "artifacts/exp03_train_subsets"

SEED_CONFIGS = [
    {"seed_name": "seed0", "seed_index": 0, "base_seed": 3407},
    {"seed_name": "seed1", "seed_index": 1, "base_seed": 2027},
    {"seed_name": "seed2", "seed_index": 2, "base_seed": 9413},
]

# Frozen class order (from dataset_v1.yaml), locale-independent
FROZEN_CLASS_ORDER = ["二胡", "琵琶", "中阮", "笛子", "唢呐", "笙"]


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_subset_for_ratio_seed(
    train_df: pd.DataFrame,
    ratio_name: str,
    ratio_value: float,
    subset_seed: int,
    seed_name: str,
    base_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build train subset for a single ratio x seed.

    Returns:
        (groups_df, segments_df, meta_dict)
    """
    # Stratified sampling by family_label
    group_rows = []
    kept_group_ids = set()
    class_group_counts = {}
    class_segment_counts = {}

    for class_index, family_label in enumerate(FROZEN_CLASS_ORDER):
        class_train = train_df[train_df["family_label"] == family_label]
        # Unique recording_group_ids in canonical sorted order
        unique_groups = sorted(class_train["recording_group_id"].unique())
        n_full = len(unique_groups)

        if n_full == 0:
            raise RuntimeError(
                f"blocking_error: {family_label} has no recording_group_id in train"
            )

        # Number to keep
        n_keep = max(1, math.ceil(n_full * ratio_value))

        # Independent RNG per class: subset_seed + class_index
        class_rng = random.Random(subset_seed + class_index)
        shuffled = list(unique_groups)
        class_rng.shuffle(shuffled)
        kept = shuffled[:n_keep]

        # Record selected groups
        for rank, gid in enumerate(kept):
            group_rows.append({
                "seed_name": seed_name,
                "base_seed": base_seed,
                "ratio_name": ratio_name,
                "ratio_value": ratio_value,
                "subset_seed": subset_seed,
                "family_label": family_label,
                "recording_group_id": gid,
                "selection_rank": rank,
            })
            kept_group_ids.add(gid)

        # Statistics
        class_segments = class_train[
            class_train["recording_group_id"].isin(set(kept))
        ]
        class_group_counts[family_label] = {
            "full": n_full,
            "subset": n_keep,
        }
        class_segment_counts[family_label] = {
            "full": len(class_train),
            "subset": len(class_segments),
        }

    # Build segments DataFrame
    # Filter train to kept groups, preserving original row order
    subset_train = train_df[
        train_df["recording_group_id"].isin(kept_group_ids)
    ].copy()

    segment_rows = []
    for _, row in subset_train.iterrows():
        segment_rows.append({
            "seed_name": seed_name,
            "base_seed": base_seed,
            "ratio_name": ratio_name,
            "ratio_value": ratio_value,
            "subset_seed": subset_seed,
            "segment_id": row["segment_id"],
            "sample_id": row["sample_id"],
            "record_id": row["record_id"],
            "recording_group_id": row["recording_group_id"],
            "family_label": row["family_label"],
            "split": "train",
        })

    groups_df = pd.DataFrame(group_rows)
    segments_df = pd.DataFrame(segment_rows)

    # Summary statistics
    full_train_groups = train_df["recording_group_id"].nunique()
    full_train_records = train_df["record_id"].nunique()
    full_train_segments = len(train_df)
    subset_train_groups = len(kept_group_ids)
    subset_train_records = subset_train["record_id"].nunique()
    subset_train_segments = len(subset_train)

    meta = {
        "seed_name": seed_name,
        "base_seed": base_seed,
        "ratio_name": ratio_name,
        "ratio_value": ratio_value,
        "subset_seed": subset_seed,
        "sampling_unit": "recording_group_id",
        "rounding_rule": "max(1, ceil(n_full * ratio))",
        "full_train_group_count": full_train_groups,
        "full_train_record_count": full_train_records,
        "full_train_segment_count": full_train_segments,
        "subset_train_group_count": subset_train_groups,
        "subset_train_record_count": subset_train_records,
        "subset_train_segment_count": subset_train_segments,
        "actual_group_ratio_overall": round(subset_train_groups / full_train_groups, 4)
        if full_train_groups > 0 else 0.0,
        "actual_segment_ratio_overall": round(subset_train_segments / full_train_segments, 4)
        if full_train_segments > 0 else 0.0,
        "per_class_group_counts": class_group_counts,
        "per_class_segment_counts": class_segment_counts,
    }

    return groups_df, segments_df, meta


def run_audit(
    subsample_cfg: dict,
    all_seed_train_dfs: dict[str, pd.DataFrame],
) -> dict:
    """Audit all 12 subsets.

    Returns:
        audit report dict
    """
    ratio_defs = subsample_cfg["ratio_definitions"]
    checks = []
    all_passed = True

    for ratio_name in sorted(ratio_defs.keys()):
        for seed_cfg in SEED_CONFIGS:
            seed_name = seed_cfg["seed_name"]
            subset_dir = SUBSETS_DIR / ratio_name / seed_name

            # Check 1: files exist
            groups_csv = subset_dir / "train_subset_groups.csv"
            segments_csv = subset_dir / "train_subset_segments.csv"
            meta_json = subset_dir / "subset_meta.json"

            files_exist = groups_csv.exists() and segments_csv.exists() and meta_json.exists()
            if not files_exist:
                checks.append({
                    "ratio": ratio_name, "seed": seed_name,
                    "check": "files_exist", "passed": False,
                    "detail": "Subset files missing",
                })
                all_passed = False
                continue

            seg_df = pd.read_csv(segments_csv, dtype=str)
            groups_df = pd.read_csv(groups_csv, dtype=str)
            with open(meta_json) as f:
                meta = json.load(f)

            train_df = all_seed_train_dfs[seed_name]

            # Check 2: train-only split
            only_train = set(seg_df["split"].unique()) == {"train"}
            checks.append({
                "ratio": ratio_name, "seed": seed_name,
                "check": "only_train", "passed": only_train,
                "detail": f"splits={seg_df['split'].unique().tolist()}",
            })
            if not only_train:
                all_passed = False

            # Check 3: all from full train
            full_train_ids = set(train_df["segment_id"])
            subset_ids = set(seg_df["segment_id"])
            all_from_train = subset_ids.issubset(full_train_ids)
            checks.append({
                "ratio": ratio_name, "seed": seed_name,
                "check": "all_from_train", "passed": all_from_train,
                "detail": f"missing={len(subset_ids - full_train_ids)}",
            })
            if not all_from_train:
                all_passed = False

            # Check 4: all 6 classes present
            actual_classes = set(seg_df["family_label"].unique())
            expected_classes = set(FROZEN_CLASS_ORDER)
            has_all_classes = actual_classes == expected_classes
            checks.append({
                "ratio": ratio_name, "seed": seed_name,
                "check": "has_6_classes", "passed": has_all_classes,
                "detail": f"actual={sorted(actual_classes)}, expected={sorted(expected_classes)}",
            })
            if not has_all_classes:
                all_passed = False

            # Check 5: train100 groups match full train
            if ratio_name == "train100":
                full_groups = set(train_df["recording_group_id"].unique())
                subset_groups = set(groups_df["recording_group_id"].unique())
                groups_match = full_groups == subset_groups
                checks.append({
                    "ratio": ratio_name, "seed": seed_name,
                    "check": "train100_groups_match", "passed": groups_match,
                    "detail": f"full={len(full_groups)} subset={len(subset_groups)}",
                })
                if not groups_match:
                    all_passed = False

            # Check 6: counts match meta
            meta_group_count = meta["subset_train_group_count"]
            meta_segment_count = meta["subset_train_segment_count"]
            actual_group_count = groups_df["recording_group_id"].nunique()
            actual_segment_count = len(seg_df)
            counts_match = (
                actual_group_count == meta_group_count
                and actual_segment_count == meta_segment_count
            )
            checks.append({
                "ratio": ratio_name, "seed": seed_name,
                "check": "counts_match_meta", "passed": counts_match,
                "detail": (
                    f"groups: actual={actual_group_count} meta={meta_group_count}, "
                    f"segments: actual={actual_segment_count} meta={meta_segment_count}"
                ),
            })
            if not counts_match:
                all_passed = False

    return {
        "passed": all_passed,
        "total_checks": len(checks),
        "failed_checks": sum(1 for c in checks if not c["passed"]),
        "checks": checks,
    }


def main():
    print("Phase 3 — Train Subset Builder")
    print("=" * 60)

    subsample_cfg = _load_yaml(SUBSAMPLE_CFG)
    ratio_defs = subsample_cfg["ratio_definitions"]
    ratio_offsets = subsample_cfg["ratio_offsets"]

    # Load all seed data
    all_unified = {}
    all_train = {}
    for seed_cfg in SEED_CONFIGS:
        seed_name = seed_cfg["seed_name"]
        seed_cfg_path = EXP_ROOT / f"configs/data/{seed_name}.yaml"
        unified_df = load_seed_data(DATASET_CFG, seed_cfg_path, PROJECT_ROOT)
        all_unified[seed_name] = unified_df
        all_train[seed_name] = unified_df[unified_df["split"] == "train"].copy()
        print(f"  {seed_name}: {len(unified_df)} total, {len(all_train[seed_name])} train")

    # Build 12 subsets
    summary_rows = []  # ratio × seed × class
    overview_rows = []  # ratio × seed

    for ratio_name in sorted(ratio_defs.keys()):
        ratio_value = ratio_defs[ratio_name]
        offset = ratio_offsets[ratio_name]

        for seed_cfg in SEED_CONFIGS:
            seed_name = seed_cfg["seed_name"]
            base_seed = seed_cfg["base_seed"]
            subset_seed = base_seed + offset

            print(f"\n  Building: {ratio_name} × {seed_name} (subset_seed={subset_seed})")

            train_df = all_train[seed_name]

            groups_df, segments_df, meta = build_subset_for_ratio_seed(
                train_df=train_df,
                ratio_name=ratio_name,
                ratio_value=ratio_value,
                subset_seed=subset_seed,
                seed_name=seed_name,
                base_seed=base_seed,
            )

            # Write outputs
            out_dir = SUBSETS_DIR / ratio_name / seed_name
            out_dir.mkdir(parents=True, exist_ok=True)

            groups_df.to_csv(out_dir / "train_subset_groups.csv", index=False)
            segments_df.to_csv(out_dir / "train_subset_segments.csv", index=False)
            with open(out_dir / "subset_meta.json", "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            print(f"    groups={meta['subset_train_group_count']}, "
                  f"segments={meta['subset_train_segment_count']}, "
                  f"actual_group_ratio={meta['actual_group_ratio_overall']:.3f}")

            # Summary row — per class
            for family_label in FROZEN_CLASS_ORDER:
                gc = meta["per_class_group_counts"][family_label]
                sc = meta["per_class_segment_counts"][family_label]
                summary_rows.append({
                    "ratio_name": ratio_name,
                    "ratio_value": ratio_value,
                    "seed_name": seed_name,
                    "family_label": family_label,
                    "full_group_count": gc["full"],
                    "subset_group_count": gc["subset"],
                    "actual_group_ratio": round(gc["subset"] / gc["full"], 4) if gc["full"] > 0 else 0.0,
                    "full_record_count": train_df[train_df["family_label"] == family_label]["record_id"].nunique(),
                    "subset_record_count": segments_df[segments_df["family_label"] == family_label]["record_id"].nunique()
                    if len(segments_df[segments_df["family_label"] == family_label]) > 0 else 0,
                    "full_segment_count": sc["full"],
                    "subset_segment_count": sc["subset"],
                    "actual_segment_ratio": round(sc["subset"] / sc["full"], 4) if sc["full"] > 0 else 0.0,
                })

            # Summary row — overview
            overview_rows.append({
                "ratio_name": ratio_name,
                "ratio_value": ratio_value,
                "seed_name": seed_name,
                "subset_seed": subset_seed,
                "full_train_group_count": meta["full_train_group_count"],
                "subset_train_group_count": meta["subset_train_group_count"],
                "actual_group_ratio_overall": meta["actual_group_ratio_overall"],
                "full_train_record_count": meta["full_train_record_count"],
                "subset_train_record_count": meta["subset_train_record_count"],
                "full_train_segment_count": meta["full_train_segment_count"],
                "subset_train_segment_count": meta["subset_train_segment_count"],
                "actual_segment_ratio_overall": meta["actual_segment_ratio_overall"],
            })

    # Write summary tables
    reports_dir = EXP_ROOT / "reports/tables"
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(reports_dir / "exp03_train_subset_summary.csv", index=False)
    print(f"\n  exp03_train_subset_summary.csv: {len(summary_df)} rows")

    overview_df = pd.DataFrame(overview_rows)
    overview_df.to_csv(reports_dir / "exp03_train_subset_overview.csv", index=False)
    print(f"  exp03_train_subset_overview.csv: {len(overview_df)} rows")

    # Audit
    print("\n  Running audit...")
    audit_report = run_audit(subsample_cfg, all_train)

    audit_path = SUBSETS_DIR / "subset_audit_report.json"
    with open(audit_path, "w") as f:
        json.dump(audit_report, f, indent=2, ensure_ascii=False)

    if audit_report["passed"]:
        print(f"  Audit: PASSED ({audit_report['total_checks']} checks)")
    else:
        print(f"  Audit: FAILED ({audit_report['failed_checks']}/{audit_report['total_checks']} checks failed)")
        for check in audit_report["checks"]:
            if not check["passed"]:
                print(f"    FAIL: {check['ratio']} × {check['seed']} — "
                      f"{check['check']}: {check['detail']}")
        sys.exit(1)

    print("\nPhase 3 Subset Build: DONE")


if __name__ == "__main__":
    main()
