"""
Phase 2 prerequisite checks: Phase 1 completion, MERT weights, configs, peft, writable dirs.
Any check failure exits with code 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = PROJECT_ROOT / "instrument_cls_experiments"


def check_phase1_mert_linear_complete() -> tuple[bool, str]:
    """Check Phase 1 exp01_mert_linear 3 seeds all completed."""
    registry_path = EXP_ROOT / "registry/run_status.csv"
    if not registry_path.exists():
        return False, "registry/run_status.csv not found"

    df = pd.read_csv(registry_path, dtype=str)
    required_runs = [
        "run_exp01_mert_linear_seed0",
        "run_exp01_mert_linear_seed1",
        "run_exp01_mert_linear_seed2",
    ]
    for run_id in required_runs:
        mask = df["run_id"] == run_id
        if not mask.any():
            return False, f"Missing {run_id}"
        status = df.loc[mask, "status"].iloc[0]
        if status != "completed":
            return False, f"{run_id} status={status} (expected completed)"

    return True, "exp01_mert_linear 3 seeds all completed"


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
    """Run all Phase 2 gate checks. Return True if all passed."""
    checks: list[tuple[bool, str]] = []

    # 1. Phase 1 complete
    checks.append(check_phase1_mert_linear_complete())

    # 2. Model cache
    checks.append(check_model_cache())

    # 3. Config files
    config_files = [
        EXP_ROOT / "configs/data/dataset_v1.yaml",
        EXP_ROOT / "configs/data/seed0.yaml",
        EXP_ROOT / "configs/data/seed1.yaml",
        EXP_ROOT / "configs/data/seed2.yaml",
    ]
    for cfg in config_files:
        checks.append(check_config_readable(cfg))

    # 4. peft
    checks.append(check_peft_importable())

    # 5. Writable output dirs
    checks.append(check_directory_writable(EXP_ROOT / "runs/exp02_adaptation"))
    checks.append(check_directory_writable(EXP_ROOT / "reports/tables"))
    checks.append(check_directory_writable(EXP_ROOT / "registry"))

    # Report
    print("Phase 2 Gate Check")
    print("=" * 50)
    all_ok = True
    for ok, msg in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {msg}")
        if not ok:
            all_ok = False

    print("=" * 50)
    if all_ok:
        print("Gate check: ALL PASSED")
    else:
        print("Gate check: FAILED — cannot enter Phase 2")

    return all_ok


if __name__ == "__main__":
    ok = run_gate_check()
    sys.exit(0 if ok else 1)
