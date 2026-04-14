"""
Audio I/O audit runner.
Checks sample rate, channels, and duration for sampled audio segments across 3 seeds.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "instrument_cls_experiments"))

from src.data.manifest_loader import load_seed_data
from src.eval.run_meta import write_run_meta
from src.eval.registry_writer import update_registry

DATASET_CFG_PATH = PROJECT_ROOT / "instrument_cls_experiments/configs/data/dataset_v1.yaml"
OUTPUT_DIR = PROJECT_ROOT / "instrument_cls_experiments/runs/smoke/audio_io_audit"
REGISTRY_CSV = PROJECT_ROOT / "instrument_cls_experiments/registry/run_status.csv"

SAMPLE_LIMITS = {"train": 12, "val": 6, "test": 6, "external_test": 6}
EXPECTED_SR = 24000
EXPECTED_CHANNELS = 1
EXPECTED_DURATION = 5.0
DURATION_TOLERANCE_SAMPLES = 1  # ±1 sample


def audit_seed(seed_index: int) -> tuple[list[dict], list[dict]]:
    """Audit audio files for a single seed. Return (errors, checked_records)."""
    seed_cfg_path = (
        PROJECT_ROOT / f"instrument_cls_experiments/configs/data/seed{seed_index}.yaml"
    )
    df = load_seed_data(DATASET_CFG_PATH, seed_cfg_path, PROJECT_ROOT)

    errors: list[dict] = []
    checked: list[dict] = []

    for split, limit in SAMPLE_LIMITS.items():
        split_df = df[df["split"] == split].sort_values("segment_id").reset_index(drop=True)
        n = min(limit, len(split_df))

        for i in range(n):
            row = split_df.iloc[i]
            abs_path = row["segment_abs_path"]
            record = {
                "seed": seed_index,
                "split": split,
                "segment_id": row["segment_id"],
                "segment_abs_path": abs_path,
                "status": "ok",
                "detail": "",
            }

            try:
                info = sf.info(abs_path)

                # Sample rate check
                if info.samplerate != EXPECTED_SR:
                    record["status"] = "blocking_error"
                    record["detail"] = f"sample_rate={info.samplerate}, expected {EXPECTED_SR}"
                    errors.append(record.copy())
                    checked.append(record)
                    continue

                # Channel check
                if info.channels != EXPECTED_CHANNELS:
                    record["status"] = "blocking_error"
                    record["detail"] = f"channels={info.channels}, expected {EXPECTED_CHANNELS}"
                    errors.append(record.copy())
                    checked.append(record)
                    continue

                # Duration check
                tolerance_sec = DURATION_TOLERANCE_SAMPLES / EXPECTED_SR
                actual_duration = info.frames / info.samplerate
                if abs(actual_duration - EXPECTED_DURATION) > tolerance_sec:
                    record["status"] = "blocking_error"
                    record["detail"] = (
                        f"duration={actual_duration:.6f}s, "
                        f"expected {EXPECTED_DURATION}s ±{tolerance_sec:.6f}s"
                    )
                    errors.append(record.copy())
                    checked.append(record)
                    continue

                # Actual read verification
                data, sr = sf.read(abs_path, dtype="float32")
                if data.ndim > 1:
                    record["status"] = "blocking_error"
                    record["detail"] = f"read shape={data.shape}, expected 1D"
                    errors.append(record.copy())

            except Exception as e:
                record["status"] = "blocking_error"
                record["detail"] = f"decode_error: {type(e).__name__}: {e}"
                errors.append(record.copy())

            checked.append(record)

    return errors, checked


def main():
    started_at = datetime.now(timezone.utc).isoformat()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_errors: list[dict] = []
    all_checked: list[dict] = []

    for seed_idx in [0, 1, 2]:
        errors, checked = audit_seed(seed_idx)
        all_errors.extend(errors)
        all_checked.extend(checked)

    passed = len(all_errors) == 0
    report = {
        "audit_type": "audio_io",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_seeds_audited": 3,
        "total_files_checked": len(all_checked),
        "num_blocking_errors": len(all_errors),
        "passed": passed,
        "expected_spec": {
            "sample_rate": EXPECTED_SR,
            "channels": EXPECTED_CHANNELS,
            "duration_sec": EXPECTED_DURATION,
            "tolerance_samples": DURATION_TOLERANCE_SAMPLES,
        },
    }

    with open(OUTPUT_DIR / "audio_io_audit.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    failures_df = pd.DataFrame(all_errors) if all_errors else pd.DataFrame(
        columns=["seed", "split", "segment_id", "segment_abs_path", "status", "detail"]
    )
    failures_df.to_csv(OUTPUT_DIR / "audio_io_failures.csv", index=False)

    # Write run_meta.json and registry
    finished_at = datetime.now(timezone.utc).isoformat()
    notes = f"{len(all_checked)} files checked"

    write_run_meta(
        run_id="run_audit_audio_io",
        seed_name="all",
        seed_index="all",
        base_seed="all",
        config_version="v1",
        started_at=started_at,
        finished_at=finished_at,
        output_path=OUTPUT_DIR / "run_meta.json",
        extra={"task_type": "audit", "model": "audio_io"},
    )
    update_registry(
        run_meta_path=OUTPUT_DIR / "run_meta.json",
        registry_csv_path=REGISTRY_CSV,
        experiment_id="smoke_audio_io",
        phase=0,
        task_type="audit",
        model="audio_io",
        notes=notes,
    )

    print(f"Audio I/O audit: {'PASSED' if passed else 'FAILED'}")
    print(f"  Files checked: {len(all_checked)}")
    print(f"  Errors: {len(all_errors)}")

    if not passed:
        print("\nErrors:")
        for e in all_errors:
            print(f"  [seed{e['seed']}/{e['split']}] {e['segment_id']}: {e['detail']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
