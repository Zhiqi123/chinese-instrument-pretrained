"""Config snapshot and run metadata writers."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


def get_code_version() -> str:
    """Get git commit hash, or 'git_unavailable'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "git_unavailable"


def write_config_snapshot(
    resolved_config: dict,
    output_path: str | Path,
) -> None:
    """Write resolved config as a YAML snapshot."""
    with open(output_path, "w") as f:
        yaml.dump(resolved_config, f, allow_unicode=True, default_flow_style=False)


def write_run_meta(
    run_id: str,
    seed_name: str,
    seed_index: int | str,
    base_seed: int | str,
    config_version: str,
    started_at: str,
    finished_at: str | None,
    output_path: str | Path,
    extra: dict | None = None,
) -> None:
    """Write run_meta.json."""
    meta = {
        "run_id": run_id,
        "seed_name": seed_name,
        "seed_index": seed_index,
        "base_seed": base_seed,
        "config_version": config_version,
        "code_version": get_code_version(),
        "started_at": started_at,
        "finished_at": finished_at or datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        meta.update(extra)

    with open(output_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
