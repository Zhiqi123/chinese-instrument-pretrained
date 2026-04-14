# Phase 2 adaptation comparison orchestrator.


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

METHODS = [
    {
        "name": "mert_linear",
        "runner": "src/runners/run_exp02_mert_linear.py",
        "config": "configs/experiments/exp02_adaptation/mert_linear_probe_v1.yaml",
    },
    {
        "name": "mert_lora_r4",
        "runner": "src/runners/run_exp02_mert_lora.py",
        "config": "configs/experiments/exp02_adaptation/mert_lora_r4_v1.yaml",
    },
    {
        "name": "mert_lora_r8",
        "runner": "src/runners/run_exp02_mert_lora.py",
        "config": "configs/experiments/exp02_adaptation/mert_lora_r8_v1.yaml",
    },
    {
        "name": "mert_full_ft",
        "runner": "src/runners/run_exp02_mert_full_ft.py",
        "config": "configs/experiments/exp02_adaptation/mert_full_finetune_v1.yaml",
    },
]

SEED_NAMES = ["seed0", "seed1", "seed2"]

METHOD_META = {
    "mert_linear": {"experiment_id": "exp02_mert_linear", "task_type": "linear_probe", "model": "mert_v1_95m"},
    "mert_lora_r4": {"experiment_id": "exp02_mert_lora_r4", "task_type": "lora", "model": "mert_v1_95m"},
    "mert_lora_r8": {"experiment_id": "exp02_mert_lora_r8", "task_type": "lora", "model": "mert_v1_95m"},
    "mert_full_ft": {"experiment_id": "exp02_mert_full_ft", "task_type": "full_finetune", "model": "mert_v1_95m"},
}

SEED_MAP = {"seed0": 3407, "seed1": 2027, "seed2": 9413}


def run_single(runner: str, config: str, seed_name: str) -> bool:
    """Run a single method+seed. Returns True on success."""
    cmd = [
        sys.executable,
        str(EXP_ROOT / runner),
        "--config", str(EXP_ROOT / config),
        "--seed-name", seed_name,
    ]
    print(f"\n{'='*60}")
    print(f"Running: {runner} --seed-name {seed_name}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def write_registry_status(method_name: str, seed_name: str, status: str, notes: str = "") -> None:
    """Write registry entry for failed/blocked runs."""
    meta = METHOD_META[method_name]
    run_id = f"run_exp02_{method_name}_{seed_name}"
    seed_index = int(seed_name[-1])
    base_seed = SEED_MAP[seed_name]
    now = datetime.now(timezone.utc).isoformat()

    row = {
        "run_id": run_id,
        "experiment_id": meta["experiment_id"],
        "phase": "2",
        "task_type": meta["task_type"],
        "model": meta["model"],
        "seed_name": seed_name,
        "seed_index": str(seed_index),
        "base_seed": str(base_seed),
        "status": status,
        "output_dir": f"runs/exp02_adaptation/{method_name}/{seed_name}",
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


def _derive_experiment_status(method_name: str) -> str:
    """Derive experiment status from registry."""
    if not REGISTRY_CSV.exists():
        return "partial"

    df = pd.read_csv(REGISTRY_CSV, dtype=str)
    prefix = f"run_exp02_{method_name}_seed"
    method_runs = df[df["run_id"].str.startswith(prefix)]

    if len(method_runs) == 0:
        return "partial"

    statuses = set(method_runs["status"].values)
    if statuses == {"completed"} and len(method_runs) == 3:
        return "completed"
    if "failed" in statuses or "blocked" in statuses:
        return "failed"
    return "partial"


def update_experiment_matrix():
    """Update experiment_matrix.csv with Phase 2 entries."""
    matrix_path = EXP_ROOT / "registry/experiment_matrix.csv"
    df = pd.read_csv(matrix_path, dtype=str)

    base_entries = [
        {"experiment_id": "exp02_mert_linear", "phase": "2", "task_type": "linear_probe",
         "model": "mert_v1_95m", "seed_mode": "3_seeds",
         "config_version": "mert_linear_probe_v1", "notes": "Phase 2 MERT linear probe baseline",
         "method_name": "mert_linear"},
        {"experiment_id": "exp02_mert_lora_r4", "phase": "2", "task_type": "lora",
         "model": "mert_v1_95m", "seed_mode": "3_seeds",
         "config_version": "mert_lora_r4_v1", "notes": "MERT + LoRA (rank=4)",
         "method_name": "mert_lora_r4"},
        {"experiment_id": "exp02_mert_lora_r8", "phase": "2", "task_type": "lora",
         "model": "mert_v1_95m", "seed_mode": "3_seeds",
         "config_version": "mert_lora_r8_v1", "notes": "MERT + LoRA (rank=8)",
         "method_name": "mert_lora_r8"},
        {"experiment_id": "exp02_mert_full_ft", "phase": "2", "task_type": "full_finetune",
         "model": "mert_v1_95m", "seed_mode": "3_seeds",
         "config_version": "mert_full_finetune_v1", "notes": "MERT full fine-tuning",
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
    print("Phase 2 — Adaptation Method Comparison")
    print("=" * 60)

    # 1. Gate check
    print("\n[Step 0] Gate Check...")
    gate_cmd = [sys.executable, str(EXP_ROOT / "src/runners/run_phase2_gate_check.py")]
    gate_ok = subprocess.run(gate_cmd).returncode == 0
    if not gate_ok:
        print("Gate check FAILED. Aborting Phase 2.")
        sys.exit(1)

    # 2. seed1 validation round
    print("\n[Step 1] seed1 Validation Round...")
    method_status = {}
    for method in METHODS:
        ok = run_single(method["runner"], method["config"], "seed1")
        method_status[method["name"]] = ok
        if not ok:
            print(f"  ⚠ {method['name']} seed1 FAILED")
            write_registry_status(method["name"], "seed1", "failed", "seed1 validation failed")
            for blocked_seed in ["seed0", "seed2"]:
                write_registry_status(method["name"], blocked_seed, "blocked",
                                       "blocked: seed1 failed")

    # 3. Run seed0 + seed2
    print("\n[Step 2] Supplementary Seeds (seed0, seed2)...")
    for method in METHODS:
        if not method_status.get(method["name"], False):
            print(f"  SKIP {method['name']}: seed1 failed, blocking seed0/seed2")
            continue

        for seed_name in ["seed0", "seed2"]:
            ok = run_single(method["runner"], method["config"], seed_name)
            if not ok:
                print(f"  ⚠ {method['name']} {seed_name} FAILED")
                write_registry_status(method["name"], seed_name, "failed",
                                       f"{seed_name} failed")
                method_status[method["name"]] = False
                if seed_name == "seed0":
                    write_registry_status(method["name"], "seed2", "blocked",
                                           "blocked: seed0 failed")
                break

    # 4. Aggregation
    print("\n[Step 3] Aggregation...")
    from src.eval.summary_aggregator import run_exp02_aggregation
    runs_base = EXP_ROOT / "runs/exp02_adaptation"
    reports_dir = EXP_ROOT / "reports/tables"

    try:
        run_exp02_aggregation(runs_base, reports_dir)
    except RuntimeError as e:
        print(f"  Aggregation FAILED: {e}")
        print("  Aborting — will NOT update experiment_matrix.")
        sys.exit(1)

    # 5. Update experiment_matrix
    print("\n[Step 4] Updating experiment_matrix...")
    update_experiment_matrix()

    # Summary
    print("\n" + "=" * 60)
    print("Phase 2 Summary:")
    for name, ok in method_status.items():
        status = "PASSED" if ok else "FAILED"
        print(f"  {name}: {status}")
    print("=" * 60)

    all_ok = all(method_status.values())
    if all_ok:
        print("Phase 2: ALL 12 RUNS COMPLETED")
    else:
        print("Phase 2: SOME RUNS FAILED — check logs above")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
