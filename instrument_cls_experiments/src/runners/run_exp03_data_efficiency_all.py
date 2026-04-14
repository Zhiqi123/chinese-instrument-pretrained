"""
Phase 3 data efficiency experiment orchestrator.
Runs gate check, subset build, pilot, remaining seeds/ratios, aggregation, and plots.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = PROJECT_ROOT / "instrument_cls_experiments"
sys.path.insert(0, str(EXP_ROOT))

from src.eval.registry_writer import REGISTRY_COLUMNS

REGISTRY_CSV = EXP_ROOT / "registry/run_status.csv"

# Method definitions
METHODS = [
    {
        "name": "mert_linear",
        "runner": "src/runners/run_exp03_mert_linear.py",
        "config": "configs/experiments/exp03_data_efficiency/mert_linear_probe_exp03_v1.yaml",
    },
    {
        "name": "mert_lora_r4",
        "runner": "src/runners/run_exp03_mert_lora.py",
        "config": "configs/experiments/exp03_data_efficiency/mert_lora_r4_exp03_v1.yaml",
    },
    {
        "name": "mert_lora_r8",
        "runner": "src/runners/run_exp03_mert_lora.py",
        "config": "configs/experiments/exp03_data_efficiency/mert_lora_r8_exp03_v1.yaml",
    },
    {
        "name": "mert_full_ft",
        "runner": "src/runners/run_exp03_mert_full_ft.py",
        "config": "configs/experiments/exp03_data_efficiency/mert_full_finetune_exp03_v1.yaml",
    },
]

RATIO_NAMES = ["train10", "train25", "train50", "train100"]
SEED_NAMES = ["seed0", "seed1", "seed2"]
SEED_MAP = {"seed0": 3407, "seed1": 2027, "seed2": 9413}

METHOD_META = {
    "mert_linear": {"experiment_id": "exp03_mert_linear", "task_type": "linear_probe", "model": "mert_v1_95m"},
    "mert_lora_r4": {"experiment_id": "exp03_mert_lora_r4", "task_type": "lora", "model": "mert_v1_95m"},
    "mert_lora_r8": {"experiment_id": "exp03_mert_lora_r8", "task_type": "lora", "model": "mert_v1_95m"},
    "mert_full_ft": {"experiment_id": "exp03_mert_full_ft", "task_type": "full_finetune", "model": "mert_v1_95m"},
}


def run_single(runner: str, config: str, seed_name: str, ratio_name: str) -> bool:
    """Run a single method+seed+ratio. Return True on success."""
    cmd = [
        sys.executable,
        str(EXP_ROOT / runner),
        "--config", str(EXP_ROOT / config),
        "--seed-name", seed_name,
        "--ratio-name", ratio_name,
    ]
    print(f"\n{'='*60}")
    print(f"Running: {runner} --seed-name {seed_name} --ratio-name {ratio_name}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def write_registry_status(method_name: str, seed_name: str, ratio_name: str,
                          status: str, notes: str = "") -> None:
    """Write registry entry for a failed/blocked run."""
    meta = METHOD_META[method_name]
    run_id = f"run_exp03_{method_name}_{ratio_name}_{seed_name}"
    seed_index = int(seed_name[-1])
    base_seed = SEED_MAP[seed_name]
    now = datetime.now(timezone.utc).isoformat()

    row = {
        "run_id": run_id,
        "experiment_id": meta["experiment_id"],
        "phase": "3",
        "task_type": meta["task_type"],
        "model": meta["model"],
        "seed_name": seed_name,
        "seed_index": str(seed_index),
        "base_seed": str(base_seed),
        "status": status,
        "output_dir": f"runs/exp03_data_efficiency/{method_name}/{ratio_name}/{seed_name}",
        "started_at": now,
        "finished_at": now,
        "notes": notes,
    }

    if REGISTRY_CSV.exists():
        df = pd.read_csv(REGISTRY_CSV, dtype=str)
    else:
        df = pd.DataFrame(columns=REGISTRY_COLUMNS)

    mask = df["run_id"] == run_id
    if mask.any():
        idx = df.index[mask][0]
        for col in REGISTRY_COLUMNS:
            df.at[idx, col] = str(row[col])
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df[REGISTRY_COLUMNS].to_csv(REGISTRY_CSV, index=False)


def validate_pilot(seed_name: str, ratio_name: str) -> bool:
    """Validate pilot outputs: file completeness + row-count consistency."""
    import json

    # Required output files per run
    required_files = [
        "run_meta.json",
        "cost_metrics.json",
        "config_snapshot.yaml",
        "val_predictions_best.csv",
        "val_metrics_best.json",
        "val_per_class_metrics_best.csv",
        "val_confusion_matrix_best.csv",
        "test_predictions.csv",
        "test_metrics.json",
        "test_per_class_metrics.csv",
        "test_confusion_matrix.csv",
        "external_test_predictions.csv",
        "external_test_metrics.json",
        "external_test_per_class_metrics.csv",
        "external_test_confusion_matrix.csv",
        "artifacts/train_subset_segments.csv",
        "artifacts/train_subset_groups.csv",
    ]

    # Row-count consistency: predictions CSV rows == metrics num_samples
    consistency_checks = [
        ("val_predictions_best.csv", "val_metrics_best.json"),
        ("test_predictions.csv", "test_metrics.json"),
        ("external_test_predictions.csv", "external_test_metrics.json"),
    ]

    all_ok = True
    for method in METHODS:
        run_dir = (EXP_ROOT / "runs/exp03_data_efficiency" /
                   method["name"] / ratio_name / seed_name)

        # File existence check
        for f in required_files:
            if not (run_dir / f).exists():
                print(f"  Pilot FAIL: {method['name']} missing {f}")
                all_ok = False

        if not all_ok:
            continue

        # Row-count consistency check
        for pred_csv, metrics_json in consistency_checks:
            pred_path = run_dir / pred_csv
            metrics_path = run_dir / metrics_json

            pred_df = pd.read_csv(pred_path)
            with open(metrics_path) as f:
                metrics = json.load(f)

            pred_rows = len(pred_df)
            expected_n = metrics["num_samples"]
            if pred_rows != expected_n:
                print(f"  Pilot FAIL: {method['name']} {pred_csv} "
                      f"rows {pred_rows} != {metrics_json} num_samples {expected_n}")
                all_ok = False

    if all_ok:
        print(f"  Pilot validation PASSED: {seed_name} × {ratio_name}")
    return all_ok


def _derive_experiment_status(method_name: str) -> str:
    """Derive experiment status from registry."""
    if not REGISTRY_CSV.exists():
        return "partial"

    df = pd.read_csv(REGISTRY_CSV, dtype=str)
    prefix = f"run_exp03_{method_name}_"
    method_runs = df[df["run_id"].str.startswith(prefix)]

    if len(method_runs) == 0:
        return "partial"

    statuses = set(method_runs["status"].values)
    if statuses == {"completed"} and len(method_runs) == 12:
        return "completed"
    if "failed" in statuses or "blocked" in statuses:
        return "failed"
    return "partial"


def update_experiment_matrix():
    """Update experiment_matrix.csv with Phase 3 entries."""
    matrix_path = EXP_ROOT / "registry/experiment_matrix.csv"
    df = pd.read_csv(matrix_path, dtype=str)

    base_entries = [
        {"experiment_id": "exp03_mert_linear", "phase": "3", "task_type": "linear_probe",
         "model": "mert_v1_95m", "seed_mode": "3_seeds_4_ratios",
         "config_version": "mert_linear_probe_exp03_v1",
         "notes": "Phase 3 MERT linear probe data efficiency",
         "method_name": "mert_linear"},
        {"experiment_id": "exp03_mert_lora_r4", "phase": "3", "task_type": "lora",
         "model": "mert_v1_95m", "seed_mode": "3_seeds_4_ratios",
         "config_version": "mert_lora_r4_exp03_v1",
         "notes": "Phase 3 MERT + LoRA (rank=4) data efficiency",
         "method_name": "mert_lora_r4"},
        {"experiment_id": "exp03_mert_lora_r8", "phase": "3", "task_type": "lora",
         "model": "mert_v1_95m", "seed_mode": "3_seeds_4_ratios",
         "config_version": "mert_lora_r8_exp03_v1",
         "notes": "Phase 3 MERT + LoRA (rank=8) data efficiency",
         "method_name": "mert_lora_r8"},
        {"experiment_id": "exp03_mert_full_ft", "phase": "3", "task_type": "full_finetune",
         "model": "mert_v1_95m", "seed_mode": "3_seeds_4_ratios",
         "config_version": "mert_full_finetune_exp03_v1",
         "notes": "Phase 3 MERT full fine-tuning data efficiency",
         "method_name": "mert_full_ft"},
    ]

    for base in base_entries:
        method_name = base.pop("method_name")
        entry = {**base, "status": _derive_experiment_status(method_name)}

        mask = df["experiment_id"] == entry["experiment_id"]
        if mask.any():
            idx = df.index[mask][0]
            for k, v in entry.items():
                df.at[idx, k] = v
        else:
            df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)

    df.to_csv(matrix_path, index=False)
    print(f"  experiment_matrix.csv: {len(df)} rows")


def main():
    print("Phase 3 — Data Efficiency Experiment")
    print("=" * 60)

    # Gate check
    print("\n[Step 0] Gate Check...")
    gate_cmd = [sys.executable, str(EXP_ROOT / "src/runners/run_phase3_gate_check.py")]
    gate_ok = subprocess.run(gate_cmd).returncode == 0
    if not gate_ok:
        print("Gate check FAILED. Aborting Phase 3.")
        sys.exit(1)

    # Subset build
    print("\n[Step 1] Subset Build...")
    build_cmd = [sys.executable, str(EXP_ROOT / "src/runners/run_phase3_subset_build.py")]
    build_ok = subprocess.run(build_cmd).returncode == 0
    if not build_ok:
        print("Subset build FAILED. Aborting Phase 3.")
        sys.exit(1)

    # Pilot: seed1 x train25
    print("\n[Step 2] Pilot: seed1 × train25...")
    any_failed = False
    run_results = {}  # (method_name, ratio_name, seed_name) → bool
    pilot_failed = False

    for method in METHODS:
        ok = run_single(method["runner"], method["config"], "seed1", "train25")
        run_results[(method["name"], "train25", "seed1")] = ok
        if not ok:
            pilot_failed = True
            any_failed = True
            print(f"  PILOT FAIL: {method['name']} seed1 × train25 FAILED")
            write_registry_status(method["name"], "seed1", "train25", "failed",
                                  "pilot run failed")

    # Any pilot failure → abort
    if pilot_failed:
        print("Pilot run(s) FAILED. Aborting Phase 3.")
        sys.exit(1)

    # Validate pilot output completeness
    if not validate_pilot("seed1", "train25"):
        print("Pilot validation FAILED. Aborting Phase 3.")
        sys.exit(1)

    # seed1 remaining ratios
    print("\n[Step 3] seed1 remaining ratios...")
    for ratio_name in ["train10", "train50", "train100"]:
        for method in METHODS:
            ok = run_single(method["runner"], method["config"], "seed1", ratio_name)
            run_results[(method["name"], ratio_name, "seed1")] = ok
            if not ok:
                any_failed = True
                print(f"  WARNING: {method['name']} seed1 × {ratio_name} FAILED")
                write_registry_status(method["name"], "seed1", ratio_name, "failed",
                                      f"seed1 × {ratio_name} failed")

    # seed0 all ratios
    print("\n[Step 4] seed0 all ratios...")
    for ratio_name in RATIO_NAMES:
        for method in METHODS:
            ok = run_single(method["runner"], method["config"], "seed0", ratio_name)
            run_results[(method["name"], ratio_name, "seed0")] = ok
            if not ok:
                any_failed = True
                print(f"  WARNING: {method['name']} seed0 × {ratio_name} FAILED")
                write_registry_status(method["name"], "seed0", ratio_name, "failed",
                                      f"seed0 × {ratio_name} failed")

    # seed2 all ratios
    print("\n[Step 5] seed2 all ratios...")
    for ratio_name in RATIO_NAMES:
        for method in METHODS:
            ok = run_single(method["runner"], method["config"], "seed2", ratio_name)
            run_results[(method["name"], ratio_name, "seed2")] = ok
            if not ok:
                any_failed = True
                print(f"  WARNING: {method['name']} seed2 × {ratio_name} FAILED")
                write_registry_status(method["name"], "seed2", ratio_name, "failed",
                                      f"seed2 × {ratio_name} failed")

    # Aggregation
    print("\n[Step 6] Aggregation...")
    from src.eval.summary_aggregator import run_exp03_aggregation
    runs_base = EXP_ROOT / "runs/exp03_data_efficiency"
    reports_dir = EXP_ROOT / "reports/tables"

    try:
        run_exp03_aggregation(runs_base, reports_dir)
    except RuntimeError as e:
        print(f"  Aggregation FAILED: {e}")
        any_failed = True

    # Plotting
    print("\n[Step 7] Plotting...")
    plot_cmd = [sys.executable, str(EXP_ROOT / "src/analysis/plot_exp03_learning_curves.py")]
    plot_ok = subprocess.run(plot_cmd).returncode == 0
    if not plot_ok:
        print("  Plotting FAILED")
        any_failed = True

    # Update experiment_matrix
    print("\n[Step 8] Updating experiment_matrix...")
    update_experiment_matrix()

    # Summary
    print("\n" + "=" * 60)
    print("Phase 3 Summary:")
    total_runs = len(run_results)
    passed_runs = sum(1 for ok in run_results.values() if ok)
    failed_runs = total_runs - passed_runs
    print(f"  Total runs: {total_runs}")
    print(f"  Passed: {passed_runs}")
    print(f"  Failed: {failed_runs}")

    if failed_runs > 0:
        print("\n  Failed runs:")
        for (method, ratio, seed), ok in sorted(run_results.items()):
            if not ok:
                print(f"    {method} × {ratio} × {seed}")

    print("=" * 60)

    if not any_failed:
        print("Phase 3: ALL 48 RUNS COMPLETED")
    else:
        print("Phase 3: SOME RUNS FAILED — check logs above")

    sys.exit(0 if not any_failed else 1)


if __name__ == "__main__":
    main()
