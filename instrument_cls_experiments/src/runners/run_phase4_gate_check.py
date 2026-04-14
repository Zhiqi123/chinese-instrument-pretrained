"""
Phase 4 prerequisite checks and selection manifest generation.
Verifies Phase 1/2/3 completion, resolves best methods, writes manifest.
Any check failure exits with code 1.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = PROJECT_ROOT / "instrument_cls_experiments"

# Fixed method sets

PHASE1_METHODS = ["mfcc_svm", "clap_zeroshot", "clap_linear", "mert_linear"]
PHASE2_METHODS = ["mert_linear", "mert_lora_r4", "mert_lora_r8", "mert_full_ft"]
SEED_NAMES = ["seed0", "seed1", "seed2"]
CM_SPLITS = ["test", "external_test"]

# Check functions

def _check_runs_complete(
    registry_df: pd.DataFrame,
    experiment_prefix: str,
    expected_run_ids: list[str],
) -> tuple[bool, str]:
    """Check specified run_ids are all completed."""
    missing = []
    not_completed = []
    for run_id in expected_run_ids:
        mask = registry_df["run_id"] == run_id
        if not mask.any():
            missing.append(run_id)
            continue
        status = registry_df.loc[mask, "status"].iloc[0]
        if status != "completed":
            not_completed.append(f"{run_id}(status={status})")

    if missing or not_completed:
        parts = []
        if missing:
            parts.append(f"missing: {missing[:5]}")
        if not_completed:
            parts.append(f"not completed: {not_completed[:5]}")
        return False, f"{experiment_prefix} check failed — {'; '.join(parts)}"
    return True, f"{experiment_prefix} {len(expected_run_ids)} runs all completed"


def check_phase1_runs_complete(registry_df: pd.DataFrame) -> tuple[bool, str]:
    expected = [
        f"run_exp01_{method}_{seed}"
        for method in PHASE1_METHODS
        for seed in SEED_NAMES
    ]
    return _check_runs_complete(registry_df, "Phase 1 (exp01)", expected)


def check_phase2_runs_complete(registry_df: pd.DataFrame) -> tuple[bool, str]:
    expected = [
        f"run_exp02_{method}_{seed}"
        for method in PHASE2_METHODS
        for seed in SEED_NAMES
    ]
    return _check_runs_complete(registry_df, "Phase 2 (exp02)", expected)


def check_phase3_runs_complete(registry_df: pd.DataFrame) -> tuple[bool, str]:
    ratios = ["train10", "train25", "train50", "train100"]
    expected = [
        f"run_exp03_{method}_{ratio}_{seed}"
        for method in PHASE2_METHODS
        for ratio in ratios
        for seed in SEED_NAMES
    ]
    return _check_runs_complete(registry_df, "Phase 3 (exp03)", expected)


def check_summary_tables_exist() -> tuple[bool, str]:
    """Check 9 required summary tables exist."""
    tables = [
        "exp01_transfer_run_metrics.csv",
        "exp01_transfer_summary_mean_std.csv",
        "exp01_transfer_per_class_summary.csv",
        "exp02_adaptation_run_metrics.csv",
        "exp02_adaptation_summary_mean_std.csv",
        "exp02_adaptation_per_class_summary.csv",
        "exp02_adaptation_run_costs.csv",
        "exp02_adaptation_cost_summary_mean_std.csv",
        "exp03_learning_curve_points.csv",
    ]
    tables_dir = EXP_ROOT / "reports/tables"
    missing = [t for t in tables if not (tables_dir / t).exists()]
    if missing:
        return False, f"Missing {len(missing)} summary tables: {missing[:5]}"
    return True, f"9 summary tables all exist"


def check_confusion_matrices_exist() -> tuple[bool, str]:
    """Check Phase 1/2 confusion matrix files exist."""
    missing = []

    # Phase 1
    for method in PHASE1_METHODS:
        for seed in SEED_NAMES:
            for split in CM_SPLITS:
                path = EXP_ROOT / f"runs/exp01_transfer/{method}/{seed}/{split}_confusion_matrix.csv"
                if not path.exists():
                    missing.append(str(path.relative_to(EXP_ROOT)))

    # Phase 2
    for method in PHASE2_METHODS:
        for seed in SEED_NAMES:
            for split in CM_SPLITS:
                path = EXP_ROOT / f"runs/exp02_adaptation/{method}/{seed}/{split}_confusion_matrix.csv"
                if not path.exists():
                    missing.append(str(path.relative_to(EXP_ROOT)))

    if missing:
        return False, f"Missing {len(missing)} confusion matrices: {missing[:5]}"
    total = (len(PHASE1_METHODS) + len(PHASE2_METHODS)) * len(SEED_NAMES) * len(CM_SPLITS)
    return True, f"Phase 1/2 confusion matrices all exist ({total} total)"


def check_phase2_run_dirs_readable() -> tuple[bool, str]:
    """Check Phase 2 seed1 run dirs are readable."""
    missing = []
    for method in PHASE2_METHODS:
        run_dir = EXP_ROOT / f"runs/exp02_adaptation/{method}/seed1"
        if not run_dir.is_dir():
            missing.append(method)
    if missing:
        return False, f"Phase 2 seed1 dirs missing: {missing}"
    return True, "Phase 2 all method seed1 dirs readable"


def check_model_cache() -> tuple[bool, str]:
    model_dir = PROJECT_ROOT / "model_cache/huggingface/models--m-a-p--MERT-v1-95M"
    if model_dir.is_dir():
        return True, "MERT model cache exists"
    return False, f"MERT model cache missing: {model_dir}"


def check_config_readable(config_path: Path) -> tuple[bool, str]:
    if not config_path.exists():
        return False, f"Config missing: {config_path.relative_to(EXP_ROOT)}"
    try:
        with open(config_path) as f:
            yaml.safe_load(f)
        return True, f"Config readable: {config_path.name}"
    except Exception as e:
        return False, f"Config parse failed: {config_path.name} — {e}"


def check_directory_writable(dir_path: Path) -> tuple[bool, str]:
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        test_file = dir_path / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        return True, f"Dir writable: {dir_path.relative_to(EXP_ROOT)}"
    except Exception as e:
        return False, f"Dir not writable: {dir_path} — {e}"


def check_umap_importable() -> tuple[bool, str]:
    """Check umap importability in subprocess.

    Uses same import path as production code to ensure consistency.
    """
    import subprocess
    try:
        # Subprocess with same safe import wrapper
        check_script = (
            "import sys, os; "
            f"sys.path.insert(0, {str(EXP_ROOT)!r}); "
            "from src.utils.umap_import import get_umap_class; "
            "UMAP = get_umap_class(); "
            "import umap; print(umap.__version__)"
        )
        result = subprocess.run(
            [sys.executable, "-c", check_script],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, f"umap importable (version={version})"
        else:
            stderr = result.stderr.strip()[:300]
            return False, f"umap import failed (returncode={result.returncode}): {stderr}"
    except subprocess.TimeoutExpired:
        return False, "umap import timeout (60s)"
    except Exception as e:
        return False, f"umap check error: {e}"


def check_peft_importable() -> tuple[bool, str]:
    try:
        import peft  # noqa: F401
        return True, f"peft importable (version={peft.__version__})"
    except ImportError as e:
        return False, f"peft not importable: {e}"

# Selection resolution

def resolve_best_methods(analysis_cfg: dict) -> dict:
    """Resolve best_phase2_method and best_lora_method from frozen rules.

    Returns selection dict for manifest.
    """
    perf_path = EXP_ROOT / "reports/tables/exp02_adaptation_summary_mean_std.csv"
    cost_path = EXP_ROOT / "reports/tables/exp02_adaptation_cost_summary_mean_std.csv"

    perf_df = pd.read_csv(perf_path)
    cost_df = pd.read_csv(cost_path)

    # Join key: method + model
    joined = perf_df.merge(cost_df, on=["method", "model"], how="inner", suffixes=("", "_cost"))

    # Join quality check
    if len(joined) == 0:
        raise RuntimeError("perf/cost table join result is empty")

    # Verify all methods in mapping
    display_to_short = analysis_cfg["phase2_method_display_to_short"]
    for method in joined["method"].unique():
        if method not in display_to_short:
            raise RuntimeError(f"method '{method}' not in phase2_method_display_to_short mapping")

    # Only test split
    test_df = joined[joined["split"] == "test"].copy()
    if len(test_df) == 0:
        raise RuntimeError("No split=test rows in joined result")

    # Check (method, model) uniqueness
    if test_df.duplicated(subset=["method", "model"]).any():
        raise RuntimeError("Duplicate (method, model) rows in test split")

    # Check all 4 Phase 2 methods present
    expected_displays = set(display_to_short.keys())
    actual_displays = set(test_df["method"].unique())
    missing = expected_displays - actual_displays
    if missing:
        raise RuntimeError(f"Missing methods in test split: {missing}")

    # Check required fields
    for col in ["macro_f1_mean", "trainable_params_mean"]:
        if test_df[col].isna().any():
            raise RuntimeError(f"NaN values in {col}")

    # Method ordering for tie-breaking
    method_order = analysis_cfg["method_order"]
    short_to_order = {m: i for i, m in enumerate(method_order)}
    test_df["method_short"] = test_df["method"].map(display_to_short)
    test_df["method_rank"] = test_df["method_short"].map(short_to_order)

    # Ensure method_order covers all methods
    if test_df["method_rank"].isna().any():
        unmapped = test_df.loc[test_df["method_rank"].isna(), "method_short"].tolist()
        raise RuntimeError(f"Methods missing from method_order: {unmapped}")

    # Resolve best_phase2_method
    # Sort: macro_f1_mean desc, trainable_params_mean asc, method_rank asc
    sorted_all = test_df.sort_values(
        by=["macro_f1_mean", "trainable_params_mean", "method_rank"],
        ascending=[False, True, True],
    )
    best_row = sorted_all.iloc[0]

    # Resolve best_lora_method
    lora_displays = analysis_cfg["lora_method_displays"]
    lora_df = test_df[test_df["method"].isin(lora_displays)].copy()
    if len(lora_df) == 0:
        raise RuntimeError("No LoRA methods in joined result")
    sorted_lora = lora_df.sort_values(
        by=["macro_f1_mean", "trainable_params_mean", "method_rank"],
        ascending=[False, True, True],
    )
    best_lora_row = sorted_lora.iloc[0]

    selection = {
        "phase": 4,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualitative_seed_name": analysis_cfg["qualitative_seed_name"],
        "umap_split": analysis_cfg["umap_split"],
        "error_case_split": analysis_cfg["error_case_split"],
        "error_case_top_k": analysis_cfg["error_case_top_k"],
        "selection_source_perf_table": "reports/tables/exp02_adaptation_summary_mean_std.csv",
        "selection_source_cost_table": "reports/tables/exp02_adaptation_cost_summary_mean_std.csv",
        "selection_metric": analysis_cfg["selection_metric"],
        "phase2_method_display_to_short": display_to_short,
        "best_phase2_method_short": best_row["method_short"],
        "best_phase2_method_display": best_row["method"],
        "best_phase2_test_macro_f1_mean": float(best_row["macro_f1_mean"]),
        "best_phase2_trainable_params_mean": float(best_row["trainable_params_mean"]),
        "best_lora_method_short": best_lora_row["method_short"],
        "best_lora_method_display": best_lora_row["method"],
        "best_lora_test_macro_f1_mean": float(best_lora_row["macro_f1_mean"]),
        "best_lora_trainable_params_mean": float(best_lora_row["trainable_params_mean"]),
        "tie_break_rules": [
            "1. max test macro_f1_mean",
            "2. min trainable_params_mean",
            "3. method_order: mert_linear → mert_lora_r4 → mert_lora_r8 → mert_full_ft",
        ],
    }

    return selection


def write_selection_manifest(selection: dict, output_path: Path) -> None:
    """Write phase4_selection_manifest.json."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(selection, f, indent=2, ensure_ascii=False)

# Main entry

def run_gate_check() -> bool:
    """Run all Phase 4 gate checks. Return True if all passed."""
    checks: list[tuple[bool, str]] = []

    # 0. Load registry
    registry_path = EXP_ROOT / "registry/run_status.csv"
    if not registry_path.exists():
        print("Phase 4 Gate Check")
        print("=" * 60)
        print("  [FAIL] registry/run_status.csv not found")
        print("=" * 60)
        print("Gate check: FAILED")
        return False

    registry_df = pd.read_csv(registry_path, dtype=str)

    # 1. Phase 1/2/3 complete
    checks.append(check_phase1_runs_complete(registry_df))
    checks.append(check_phase2_runs_complete(registry_df))
    checks.append(check_phase3_runs_complete(registry_df))

    # 2. Summary tables
    checks.append(check_summary_tables_exist())

    # 3. Confusion matrix
    checks.append(check_confusion_matrices_exist())

    # 4. Phase 2 run dirs
    checks.append(check_phase2_run_dirs_readable())

    # 5. Model cache
    checks.append(check_model_cache())

    # 6. Config files
    config_files = [
        EXP_ROOT / "configs/analysis/phase4_analysis_v1.yaml",
        EXP_ROOT / "configs/experiments/exp04_layer_probe/mert_layer_probe_v1.yaml",
        EXP_ROOT / "configs/data/dataset_v1.yaml",
        EXP_ROOT / "configs/data/seed0.yaml",
        EXP_ROOT / "configs/data/seed1.yaml",
        EXP_ROOT / "configs/data/seed2.yaml",
        EXP_ROOT / "configs/train/default_supervised.yaml",
    ]
    for cfg in config_files:
        checks.append(check_config_readable(cfg))

    # 7. Writable output dirs
    writable_dirs = [
        EXP_ROOT / "runs/exp04_layer_probe",
        EXP_ROOT / "artifacts/exp04_analysis",
        EXP_ROOT / "reports/tables",
        EXP_ROOT / "reports/figures",
        EXP_ROOT / "registry",
    ]
    for d in writable_dirs:
        checks.append(check_directory_writable(d))

    # 8. Dependencies
    checks.append(check_peft_importable())
    checks.append(check_umap_importable())

    # Report
    print("Phase 4 Gate Check")
    print("=" * 60)
    all_ok = True
    for ok, msg in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {msg}")
        if not ok:
            all_ok = False

    # Selection resolution
    if all_ok:
        print("-" * 60)
        print("  Resolving best_phase2_method / best_lora_method ...")
        try:
            analysis_cfg_path = EXP_ROOT / "configs/analysis/phase4_analysis_v1.yaml"
            with open(analysis_cfg_path) as f:
                analysis_cfg = yaml.safe_load(f)

            selection = resolve_best_methods(analysis_cfg)

            manifest_path = EXP_ROOT / "artifacts/exp04_analysis/phase4_selection_manifest.json"
            write_selection_manifest(selection, manifest_path)

            print(f"  [PASS] best_phase2_method = {selection['best_phase2_method_display']} "
                  f"({selection['best_phase2_method_short']}), "
                  f"test_macro_f1={selection['best_phase2_test_macro_f1_mean']:.4f}")
            print(f"  [PASS] best_lora_method = {selection['best_lora_method_display']} "
                  f"({selection['best_lora_method_short']}), "
                  f"test_macro_f1={selection['best_lora_test_macro_f1_mean']:.4f}")
            print(f"  [PASS] phase4_selection_manifest.json written: {manifest_path.relative_to(EXP_ROOT)}")
        except Exception as e:
            print(f"  [FAIL] Selection resolution failed: {e}")
            all_ok = False

    print("=" * 60)
    if all_ok:
        print("Gate check: ALL PASSED — Phase 4 ready")
    else:
        print("Gate check: FAILED — cannot enter Phase 4")

    return all_ok


if __name__ == "__main__":
    ok = run_gate_check()
    sys.exit(0 if ok else 1)
