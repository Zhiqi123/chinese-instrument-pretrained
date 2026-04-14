"""
Phase 4 full pipeline: gate check, layer probe training, analysis scripts.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = PROJECT_ROOT / "instrument_cls_experiments"
sys.path.insert(0, str(EXP_ROOT))


def _run(script: str, label: str) -> bool:
    """Run script, return True on success."""
    print(f"\n{'=' * 60}")
    print(f"[{label}]")
    print(f"{'=' * 60}")
    cmd = [sys.executable, str(EXP_ROOT / script)]
    result = subprocess.run(cmd)
    ok = result.returncode == 0
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}")
    return ok


def _run_blocking(script: str, label: str) -> None:
    """Run script; exit(1) on failure."""
    if not _run(script, label):
        print(f"\n  BLOCKING FAILURE: {label} — aborting Phase 4")
        sys.exit(1)


def main():
    print("Phase 4 — Full Pipeline")
    print("=" * 60)

    results = {}

    # 1. Gate check (blocking)
    _run_blocking("src/runners/run_phase4_gate_check.py", "Gate Check")
    results["gate_check"] = True

    # 2. Layer probe training (blocking)
    _run_blocking("src/runners/run_exp04_layer_probe_all.py", "Layer Probe Training")
    results["layer_probe"] = True

    # 3. Analysis scripts
    # Blocking steps
    blocking_analysis = [
        ("src/analysis/build_exp04_umap_points.py", "UMAP Points"),
        ("src/analysis/plot_exp04_umap.py", "UMAP Visualization"),
    ]
    for script, label in blocking_analysis:
        script_path = EXP_ROOT / script
        if not script_path.exists():
            print(f"\n  [SKIP] {label}: script not found")
            results[label] = None
            continue
        _run_blocking(script, label)
        results[label] = True

    # Non-blocking steps
    nonblocking_analysis = [
        ("src/analysis/build_exp04_confusion_summary.py", "Confusion Summary"),
        ("src/analysis/plot_exp04_confusion_matrix.py", "Confusion Matrix Plots"),
        ("src/analysis/build_exp04_per_class_analysis.py", "Per-class Analysis"),
        ("src/analysis/build_exp04_layer_probe_table.py", "Layer Probe Table"),
        ("src/analysis/plot_exp04_layer_probe.py", "Layer Probe Plots"),
        ("src/analysis/build_exp04_error_cases.py", "Error Cases"),
    ]

    for script, label in nonblocking_analysis:
        script_path = EXP_ROOT / script
        if not script_path.exists():
            print(f"\n  [SKIP] {label}: script not found")
            results[label] = None
            continue
        ok = _run(script, label)
        results[label] = ok

    # Summary
    print("\n" + "=" * 60)
    print("Phase 4 Summary")
    print("=" * 60)
    all_ok = True
    for name, ok in results.items():
        if ok is None:
            status = "SKIP"
        elif ok:
            status = "PASS"
        else:
            status = "FAIL"
            all_ok = False
        print(f"  [{status}] {name}")

    print("=" * 60)
    if all_ok:
        print("Phase 4: ALL STEPS COMPLETED")
    else:
        print("Phase 4: SOME NON-BLOCKING STEPS FAILED — check logs above")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
