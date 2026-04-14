"""
Manifest data contract audit runner.
Validates 3 seeds against schema, split, leakage, and file existence constraints.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

# Allow running from project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "instrument_cls_experiments"))

from src.data.label_map import load_label_map
from src.data.manifest_loader import (
    JOIN_KEYS,
    SEGMENT_MANIFEST_REQUIRED,
    SPLIT_MANIFEST_REQUIRED,
    load_dataset_config,
    load_manifests,
    load_seed_config,
    load_seed_data,
)
from src.eval.run_meta import write_config_snapshot, write_run_meta
from src.eval.registry_writer import update_registry

# Constants
DATASET_CFG_PATH = PROJECT_ROOT / "instrument_cls_experiments/configs/data/dataset_v1.yaml"
SEED_CFG_DIR = PROJECT_ROOT / "instrument_cls_experiments/configs/data"
OUTPUT_DIR = PROJECT_ROOT / "instrument_cls_experiments/runs/smoke/data_contract_audit"
REGISTRY_CSV = PROJECT_ROOT / "instrument_cls_experiments/registry/run_status.csv"

VALID_SPLITS = {"train", "val", "test", "external_test"}
VALID_SEED_INDICES = {0, 1, 2}
SOURCE_SPLIT_RULES = {
    "CCMusic": {"train", "val", "test"},
    "ChMusic": {"external_test"},
}
# Leakage check ID fields (excludes external_test source)
LEAKAGE_ID_FIELDS = ["sample_id", "record_id", "recording_group_id"]
PRIMARY_SPLITS = {"train", "val", "test"}


def audit_single_seed(
    seed_index: int,
    dataset_cfg: dict,
    label_map: dict,
) -> tuple[list[dict], pd.DataFrame]:
    """Audit a single seed. Return (errors_list, unified_df)."""
    errors: list[dict] = []
    seed_cfg_path = SEED_CFG_DIR / f"seed{seed_index}.yaml"
    seed_cfg = load_seed_config(seed_cfg_path)

    # Load raw manifests
    try:
        split_df, segment_df = load_manifests(seed_cfg, PROJECT_ROOT)
    except ValueError as e:
        errors.append({
            "seed": seed_index,
            "check": "manifest_load",
            "severity": "blocking_error",
            "detail": str(e),
        })
        return errors, pd.DataFrame()

    # Load unified table
    try:
        df = load_seed_data(DATASET_CFG_PATH, seed_cfg_path, PROJECT_ROOT)
    except ValueError as e:
        errors.append({
            "seed": seed_index,
            "check": "join_and_build",
            "severity": "blocking_error",
            "detail": str(e),
        })
        return errors, pd.DataFrame()

    # Seed mapping consistency check
    seeds_array = dataset_cfg.get("seeds", [])

    # seed_name should be seed{seed_index}
    expected_seed_name = f"seed{seed_index}"
    if seed_cfg.get("seed_name") != expected_seed_name:
        errors.append({
            "seed": seed_index,
            "check": "seed_mapping_consistency",
            "severity": "blocking_error",
            "detail": (
                f"seed_cfg.seed_name='{seed_cfg.get('seed_name')}' "
                f"!= expected '{expected_seed_name}'"
            ),
        })

    # seed_index should match argument
    cfg_seed_index = seed_cfg.get("seed_index")
    if cfg_seed_index != seed_index:
        errors.append({
            "seed": seed_index,
            "check": "seed_mapping_consistency",
            "severity": "blocking_error",
            "detail": (
                f"seed_cfg.seed_index={cfg_seed_index} "
                f"!= function arg seed_index={seed_index}"
            ),
        })

    # base_seed should match dataset_v1.yaml entry
    if seed_index < len(seeds_array):
        expected_base_seed = seeds_array[seed_index].get("base_seed")
        cfg_base_seed = seed_cfg.get("base_seed")
        if cfg_base_seed != expected_base_seed:
            errors.append({
                "seed": seed_index,
                "check": "seed_mapping_consistency",
                "severity": "blocking_error",
                "detail": (
                    f"seed_cfg.base_seed={cfg_base_seed} "
                    f"!= dataset_cfg.seeds[{seed_index}].base_seed={expected_base_seed}"
                ),
            })

    # Triplet uniqueness check
    triplet = (
        seed_cfg.get("seed_name"),
        seed_cfg.get("seed_index"),
        seed_cfg.get("base_seed"),
    )
    matching = [
        s for s in seeds_array
        if (s.get("seed_name"), s.get("seed_index"), s.get("base_seed")) == triplet
    ]
    if len(matching) != 1:
        errors.append({
            "seed": seed_index,
            "check": "seed_mapping_consistency",
            "severity": "blocking_error",
            "detail": (
                f"Triplet {triplet} matched {len(matching)} entries "
                f"in dataset_cfg.seeds (expected exactly 1)"
            ),
        })

    # Check valid splits
    invalid_splits = set(df["split"].unique()) - VALID_SPLITS
    if invalid_splits:
        errors.append({
            "seed": seed_index,
            "check": "valid_splits",
            "severity": "blocking_error",
            "detail": f"Invalid splits: {invalid_splits}",
        })

    # Check valid split_seed
    invalid_seeds = set(df["split_seed"].unique()) - VALID_SEED_INDICES
    if invalid_seeds:
        errors.append({
            "seed": seed_index,
            "check": "valid_split_seed",
            "severity": "blocking_error",
            "detail": f"Invalid split_seed values: {invalid_seeds}",
        })

    # Check source-split constraints
    for source, allowed_splits in SOURCE_SPLIT_RULES.items():
        source_mask = df["source_dataset"] == source
        if not source_mask.any():
            continue
        actual_splits = set(df.loc[source_mask, "split"].unique())
        forbidden = actual_splits - allowed_splits
        if forbidden:
            errors.append({
                "seed": seed_index,
                "check": "source_split_constraint",
                "severity": "blocking_error",
                "detail": f"{source} found in forbidden splits: {forbidden}",
            })

    # Check class set
    expected_classes = set(label_map["classes"])
    actual_classes = set(df["family_label"].unique())
    if actual_classes != expected_classes:
        extra = actual_classes - expected_classes
        missing = expected_classes - actual_classes
        errors.append({
            "seed": seed_index,
            "check": "class_set",
            "severity": "blocking_error",
            "detail": f"Extra: {extra}, Missing: {missing}",
        })

    # Check label_id mapping
    for _, row in df[["family_label", "label_id"]].drop_duplicates().iterrows():
        expected_id = label_map["label_to_id"].get(row["family_label"])
        if expected_id is None or int(row["label_id"]) != expected_id:
            errors.append({
                "seed": seed_index,
                "check": "label_id_consistency",
                "severity": "blocking_error",
                "detail": (
                    f"{row['family_label']}: expected label_id={expected_id}, "
                    f"got {row['label_id']}"
                ),
            })

    # Leakage check (primary splits)
    primary_df = df[df["split"].isin(PRIMARY_SPLITS)]
    for id_field in LEAKAGE_ID_FIELDS:
        if id_field not in primary_df.columns:
            continue
        id_split_map = (
            primary_df.groupby(id_field)["split"]
            .apply(lambda x: set(x.unique()))
            .reset_index()
        )
        leaked = id_split_map[id_split_map["split"].apply(len) > 1]
        if not leaked.empty:
            examples = leaked.head(5)
            errors.append({
                "seed": seed_index,
                "check": f"{id_field}_leakage",
                "severity": "blocking_error",
                "detail": (
                    f"{len(leaked)} {id_field}(s) appear in multiple primary splits. "
                    f"Examples: {examples[[id_field, 'split']].to_dict('records')}"
                ),
            })

    # Check segment_abs_path exists
    missing_files = []
    for abs_path in df["segment_abs_path"]:
        p = Path(abs_path)
        if not p.exists() or not p.is_file():
            missing_files.append(abs_path)
    if missing_files:
        errors.append({
            "seed": seed_index,
            "check": "segment_file_exists",
            "severity": "blocking_error",
            "detail": (
                f"{len(missing_files)} segment files missing. "
                f"First 5: {missing_files[:5]}"
            ),
        })

    # Check segment_id uniqueness
    dup_seg_ids = df[df.duplicated("segment_id", keep=False)]
    if not dup_seg_ids.empty:
        errors.append({
            "seed": seed_index,
            "check": "segment_id_unique",
            "severity": "blocking_error",
            "detail": (
                f"{len(dup_seg_ids)} duplicate segment_id entries. "
                f"Examples: {dup_seg_ids['segment_id'].unique()[:5].tolist()}"
            ),
        })

    return errors, df


def run_audit() -> dict:
    """Run audit for all 3 seeds."""
    dataset_cfg = load_dataset_config(DATASET_CFG_PATH)
    label_map = load_label_map(DATASET_CFG_PATH)

    all_errors: list[dict] = []
    seed_summaries = {}

    for seed_idx in [0, 1, 2]:
        errors, df = audit_single_seed(seed_idx, dataset_cfg, label_map)
        all_errors.extend(errors)

        if not df.empty:
            seed_summaries[f"seed{seed_idx}"] = {
                "total_segments": len(df),
                "splits": dict(df.groupby("split").size()),
                "classes": dict(df.groupby("family_label").size()),
                "sources": dict(df.groupby("source_dataset").size()),
            }

    blocking = [e for e in all_errors if e["severity"] == "blocking_error"]
    warnings = [e for e in all_errors if e["severity"] == "warning"]

    report = {
        "audit_type": "data_contract",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_config": str(DATASET_CFG_PATH),
        "num_seeds_audited": 3,
        "num_blocking_errors": len(blocking),
        "num_warnings": len(warnings),
        "passed": len(blocking) == 0,
        "seed_summaries": seed_summaries,
        "errors": all_errors,
    }

    return report


def main():
    started_at = datetime.now(timezone.utc).isoformat()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = run_audit()

    # Write JSON report
    report_path = OUTPUT_DIR / "data_contract_report.json"
    # Convert numpy int to Python int before serializing
    def _convert(obj):
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, set):
            return sorted(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=_convert)

    # Write errors CSV
    errors_path = OUTPUT_DIR / "data_contract_errors.csv"
    if report["errors"]:
        pd.DataFrame(report["errors"]).to_csv(errors_path, index=False)
    else:
        pd.DataFrame(columns=["seed", "check", "severity", "detail"]).to_csv(
            errors_path, index=False
        )

    # Write run_meta.json and registry
    finished_at = datetime.now(timezone.utc).isoformat()
    n_blocking = report["num_blocking_errors"]
    notes = f"3 seeds {'all passed' if n_blocking == 0 else f'{n_blocking} blocking errors'}"

    write_run_meta(
        run_id="run_audit_contract",
        seed_name="all",
        seed_index="all",
        base_seed="all",
        config_version="v1",
        started_at=started_at,
        finished_at=finished_at,
        output_path=OUTPUT_DIR / "run_meta.json",
        extra={"task_type": "audit", "model": "data_contract"},
    )
    update_registry(
        run_meta_path=OUTPUT_DIR / "run_meta.json",
        registry_csv_path=REGISTRY_CSV,
        experiment_id="smoke_audit",
        phase=0,
        task_type="audit",
        model="data_contract",
        notes=notes,
    )

    # Console summary
    print(f"Data contract audit: {'PASSED' if report['passed'] else 'FAILED'}")
    print(f"  Blocking errors: {report['num_blocking_errors']}")
    print(f"  Warnings: {report['num_warnings']}")
    print(f"  Report: {report_path}")

    if not report["passed"]:
        print("\nBlocking errors:")
        for e in report["errors"]:
            if e["severity"] == "blocking_error":
                print(f"  [seed{e['seed']}] {e['check']}: {e['detail']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
