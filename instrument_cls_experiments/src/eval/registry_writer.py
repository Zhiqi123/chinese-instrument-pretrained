"""Auto-update registry/run_status.csv from run_meta.json."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REGISTRY_COLUMNS = [
    "run_id",
    "experiment_id",
    "phase",
    "task_type",
    "model",
    "seed_name",
    "seed_index",
    "base_seed",
    "status",
    "output_dir",
    "started_at",
    "finished_at",
    "notes",
]


def update_registry(
    run_meta_path: Path,
    registry_csv_path: Path,
    experiment_id: str,
    phase: int,
    task_type: str,
    model: str,
    notes: str = "",
) -> None:
    """Read run_meta.json and upsert into registry/run_status.csv."""
    with open(run_meta_path) as f:
        meta = json.load(f)

    # Compute relative output_dir
    experiment_root = registry_csv_path.parent.parent
    output_dir = str(run_meta_path.parent.relative_to(experiment_root))

    row = {
        "run_id": meta["run_id"],
        "experiment_id": experiment_id,
        "phase": phase,
        "task_type": task_type,
        "model": model,
        "seed_name": meta["seed_name"],
        "seed_index": meta["seed_index"],
        "base_seed": meta["base_seed"],
        "status": "completed",
        "output_dir": output_dir,
        "started_at": meta["started_at"],
        "finished_at": meta["finished_at"],
        "notes": notes,
    }

    # Read existing CSV or create empty
    if registry_csv_path.exists():
        df = pd.read_csv(registry_csv_path, dtype=str)
    else:
        df = pd.DataFrame(columns=REGISTRY_COLUMNS)

    # Upsert by run_id
    run_id = row["run_id"]
    mask = df["run_id"] == run_id
    if mask.any():
        idx = df.index[mask][0]
        for col in REGISTRY_COLUMNS:
            df.at[idx, col] = str(row[col])
    else:
        new_row = {col: str(row[col]) for col in REGISTRY_COLUMNS}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df[REGISTRY_COLUMNS].to_csv(registry_csv_path, index=False)
