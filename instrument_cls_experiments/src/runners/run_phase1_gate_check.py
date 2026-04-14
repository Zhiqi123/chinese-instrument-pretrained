"""
Phase 1 prerequisite checks: audit reports, model cache, config readability.
Any check failure exits with code 1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = PROJECT_ROOT / "instrument_cls_experiments"


def check_audit_report(report_path: Path, audit_name: str) -> tuple[bool, str]:
    """Check audit report exists and passed=true."""
    if not report_path.exists():
        return False, f"{audit_name}: file not found — {report_path}"
    try:
        with open(report_path) as f:
            data = json.load(f)
        if data.get("passed") is not True:
            return False, f"{audit_name}: passed={data.get('passed')} (expected true)"
        return True, f"{audit_name}: passed=true"
    except Exception as e:
        return False, f"{audit_name}: read failed — {e}"


def check_model_cache(model_dir: Path, model_name: str) -> tuple[bool, str]:
    """Check model cache directory exists."""
    if model_dir.is_dir():
        return True, f"Model cache exists: {model_name}"
    return False, f"Model cache missing: {model_dir}"


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


def run_gate_check() -> bool:
    """Run all gate checks. Return True if all passed."""
    checks: list[tuple[bool, str]] = []

    # 1. Phase 0 audit reports
    checks.append(check_audit_report(
        EXP_ROOT / "runs/smoke/data_contract_audit/data_contract_report.json",
        "Data Contract Audit",
    ))
    checks.append(check_audit_report(
        EXP_ROOT / "runs/smoke/audio_io_audit/audio_io_audit.json",
        "Audio I/O Audit",
    ))

    # 2. Model cache
    model_cache = PROJECT_ROOT / "model_cache/huggingface"
    checks.append(check_model_cache(
        model_cache / "models--laion--clap-htsat-unfused",
        "laion/clap-htsat-unfused",
    ))
    checks.append(check_model_cache(
        model_cache / "models--m-a-p--MERT-v1-95M",
        "m-a-p/MERT-v1-95M",
    ))

    # 3. Config files
    config_files = [
        EXP_ROOT / "configs/data/dataset_v1.yaml",
        EXP_ROOT / "configs/data/seed0.yaml",
        EXP_ROOT / "configs/data/seed1.yaml",
        EXP_ROOT / "configs/data/seed2.yaml",
        EXP_ROOT / "configs/prompts/clap_zeroshot_v1.yaml",
    ]
    for cfg in config_files:
        checks.append(check_config_readable(cfg))

    # Report
    print("Phase 1 Gate Check")
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
        print("Gate check: FAILED — cannot enter Phase 1")

    return all_ok


if __name__ == "__main__":
    ok = run_gate_check()
    sys.exit(0 if ok else 1)
