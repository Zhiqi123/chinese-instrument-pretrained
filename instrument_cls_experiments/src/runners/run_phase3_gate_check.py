"""
Phase 3 prerequisite checks: exp02 runs, summary tables, MERT weights,
configs, peft, writable dirs. Any check failure exits with code 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = PROJECT_ROOT / "instrument_cls_experiments"


def check_exp02_runs_complete() -> tuple[bool, str]:
    """Check all 12 Phase 2 exp02 runs are completed."""
    registry_path = EXP_ROOT / "registry/run_status.csv"
    if not registry_path.exists():
        return False, "registry/run_status.csv not found"

    df = pd.read_csv(registry_path, dtype=str)

    methods = ["mert_linear", "mert_lora_r4", "mert_lora_r8", "mert_full_ft"]
    seeds = ["seed0", "seed1", "seed2"]
    missing = []
    not_completed = []

    for method in methods:
        for seed in seeds:
            run_id = f"run_exp02_{method}_{seed}"
            mask = df["run_id"] == run_id
            if not mask.any():
                missing.append(run_id)
            else:
                status = df.loc[mask, "status"].iloc[0]
                if status != "completed":
                    not_completed.append(f"{run_id} (status={status})")

    if missing:
        return False, f"Missing exp02 runs: {missing}"
    if not_completed:
        return False, f"exp02 runs not completed: {not_completed}"
    return True, "exp02 all 12 runs completed"


def check_exp02_summary_tables() -> tuple[bool, str]:
    """Check exp02 summary tables exist."""
    tables = [
        "exp02_adaptation_run_metrics.csv",
        "exp02_adaptation_summary_mean_std.csv",
        "exp02_adaptation_run_costs.csv",
        "exp02_adaptation_cost_summary_mean_std.csv",
    ]
    missing = []
    for t in tables:
        if not (EXP_ROOT / "reports/tables" / t).exists():
            missing.append(t)

    if missing:
        return False, f"Missing exp02 summary tables: {missing}"
    return True, "exp02 4 summary tables exist"


def check_model_cache() -> tuple[bool, str]:
    """Check MERT weights directory exists."""
    model_dir = PROJECT_ROOT / "model_cache/huggingface/models--m-a-p--MERT-v1-95M"
    if model_dir.is_dir():
        return True, "MERT model cache exists"
    return False, f"MERT model cache missing: {model_dir}"


def check_config_readable(config_path: Path) -> tuple[bool, str]:
    """Check YAML config is readable and parseable."""
    if not config_path.exists():
        return False, f"Config missing: {config_path}"
    try:
        with open(config_path) as f:
            yaml.safe_load(f)
        return True, f"Config readable: {config_path.name}"
    except Exception as e:
        return False, f"Config parse failed: {config_path.name} — {e}"


def check_peft_importable() -> tuple[bool, str]:
    """Check peft is importable."""
    try:
        import peft  # noqa: F401
        return True, f"peft importable (version={peft.__version__})"
    except ImportError as e:
        return False, f"peft not importable: {e}"


def check_directory_writable(dir_path: Path) -> tuple[bool, str]:
    """Check directory is writable (create if needed)."""
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        test_file = dir_path / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        return True, f"Dir writable: {dir_path.relative_to(EXP_ROOT)}"
    except Exception as e:
        return False, f"Dir not writable: {dir_path} — {e}"


def run_gate_check() -> bool:
    """Run all Phase 3 gate checks. Return True if all passed."""
    checks: list[tuple[bool, str]] = []

    # 1. Phase 2 complete
    checks.append(check_exp02_runs_complete())

    # 2. exp02 summary tables
    checks.append(check_exp02_summary_tables())

    # 3. Model cache
    checks.append(check_model_cache())

    # 4. Data configs
    data_configs = [
        EXP_ROOT / "configs/data/dataset_v1.yaml",
        EXP_ROOT / "configs/data/seed0.yaml",
        EXP_ROOT / "configs/data/seed1.yaml",
        EXP_ROOT / "configs/data/seed2.yaml",
    ]
    for cfg in data_configs:
        checks.append(check_config_readable(cfg))

    # 5. Phase 3 configs
    phase3_configs = [
        EXP_ROOT / "configs/data/phase3_subsample_v1.yaml",
        EXP_ROOT / "configs/experiments/exp03_data_efficiency/mert_linear_probe_exp03_v1.yaml",
        EXP_ROOT / "configs/experiments/exp03_data_efficiency/mert_lora_r4_exp03_v1.yaml",
        EXP_ROOT / "configs/experiments/exp03_data_efficiency/mert_lora_r8_exp03_v1.yaml",
        EXP_ROOT / "configs/experiments/exp03_data_efficiency/mert_full_finetune_exp03_v1.yaml",
    ]
    for cfg in phase3_configs:
        checks.append(check_config_readable(cfg))

    # 6. peft
    checks.append(check_peft_importable())

    # 7. Writable output dirs
    checks.append(check_directory_writable(EXP_ROOT / "runs/exp03_data_efficiency"))
    checks.append(check_directory_writable(EXP_ROOT / "reports/tables"))
    checks.append(check_directory_writable(EXP_ROOT / "reports/figures"))
    checks.append(check_directory_writable(EXP_ROOT / "registry"))

    # Report
    print("Phase 3 Gate Check")
    print("=" * 50)
    all_ok = True
    for ok, msg in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {msg}")
        if not ok:
            all_ok = False

    print("=" * 50)
    if all_ok:
        print("Gate check: ALL PASSED — Phase 3 ready")
    else:
        print("Gate check: FAILED — cannot enter Phase 3")

    return all_ok


if __name__ == "__main__":
    ok = run_gate_check()
    sys.exit(0 if ok else 1)
