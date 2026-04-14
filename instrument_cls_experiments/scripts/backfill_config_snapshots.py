"""One-time backfill of config_snapshot.yaml for 12 Phase 1 runs.
Replaces old filename-based format with resolved full config format.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = PROJECT_ROOT / "instrument_cls_experiments"
sys.path.insert(0, str(EXP_ROOT))

from src.eval.run_meta import write_config_snapshot


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


DATASET_CFG_PATH = EXP_ROOT / "configs/data/dataset_v1.yaml"
SUPERVISED_CFG_PATH = EXP_ROOT / "configs/train/default_supervised.yaml"
INFERENCE_CFG_PATH = EXP_ROOT / "configs/train/default_inference.yaml"

SEED_NAMES = ["seed0", "seed1", "seed2"]


def backfill_mfcc_svm():
    method_cfg = _load_yaml(EXP_ROOT / "configs/experiments/exp01_transfer/mfcc_svm_v1.yaml")
    dataset_cfg = _load_yaml(DATASET_CFG_PATH)

    for seed_name in SEED_NAMES:
        seed_cfg = _load_yaml(EXP_ROOT / f"configs/data/{seed_name}.yaml")
        output_dir = EXP_ROOT / f"runs/exp01_transfer/mfcc_svm/{seed_name}"
        if not output_dir.exists():
            print(f"  SKIP {output_dir} (not found)")
            continue

        resolved = {
            "dataset": dataset_cfg,
            "seed": seed_cfg,
            "method": method_cfg["method"],
            "model": method_cfg["model"],
            "experiment_id": method_cfg["experiment_id"],
            "version": method_cfg["version"],
            "feature_extraction": method_cfg["feature_extraction"],
            "classifier": method_cfg["classifier"],
            "output": method_cfg["output"],
        }
        write_config_snapshot(resolved, output_dir / "config_snapshot.yaml")
        print(f"  OK {output_dir / 'config_snapshot.yaml'}")


def backfill_clap_zeroshot():
    method_cfg = _load_yaml(EXP_ROOT / "configs/experiments/exp01_transfer/clap_zeroshot_eval_v1.yaml")
    dataset_cfg = _load_yaml(DATASET_CFG_PATH)
    prompt_cfg = _load_yaml(EXP_ROOT / method_cfg["model_config"]["prompt_config"])
    inf_default = _load_yaml(INFERENCE_CFG_PATH)
    model_cfg = method_cfg["model_config"]

    for seed_name in SEED_NAMES:
        seed_cfg = _load_yaml(EXP_ROOT / f"configs/data/{seed_name}.yaml")
        output_dir = EXP_ROOT / f"runs/exp01_transfer/clap_zeroshot/{seed_name}"
        if not output_dir.exists():
            print(f"  SKIP {output_dir} (not found)")
            continue

        resolved = {
            "dataset": dataset_cfg,
            "seed": seed_cfg,
            "method": method_cfg["method"],
            "model": method_cfg["model"],
            "experiment_id": method_cfg["experiment_id"],
            "version": method_cfg["version"],
            "model_config": model_cfg,
            "inference": {
                **inf_default.get("inference", {}),
                **method_cfg["inference"],
            },
            "prompt_config": prompt_cfg,
            "eval_splits": method_cfg["eval_splits"],
            "output": method_cfg["output"],
        }
        write_config_snapshot(resolved, output_dir / "config_snapshot.yaml")
        print(f"  OK {output_dir / 'config_snapshot.yaml'}")


def backfill_clap_linear():
    method_cfg = _load_yaml(EXP_ROOT / "configs/experiments/exp01_transfer/clap_linear_probe_v1.yaml")
    dataset_cfg = _load_yaml(DATASET_CFG_PATH)
    sup_cfg = _load_yaml(SUPERVISED_CFG_PATH)
    model_cfg = method_cfg["model_config"]

    for seed_name in SEED_NAMES:
        seed_cfg = _load_yaml(EXP_ROOT / f"configs/data/{seed_name}.yaml")
        output_dir = EXP_ROOT / f"runs/exp01_transfer/clap_linear/{seed_name}"
        if not output_dir.exists():
            print(f"  SKIP {output_dir} (not found)")
            continue

        resolved = {
            "dataset": dataset_cfg,
            "seed": seed_cfg,
            "method": method_cfg["method"],
            "model": method_cfg["model"],
            "experiment_id": method_cfg["experiment_id"],
            "version": method_cfg["version"],
            "model_config": model_cfg,
            "training": sup_cfg["training"],
            "evaluation": sup_cfg["evaluation"],
            "device": sup_cfg.get("device", "auto"),
            "output": method_cfg["output"],
        }
        write_config_snapshot(resolved, output_dir / "config_snapshot.yaml")
        print(f"  OK {output_dir / 'config_snapshot.yaml'}")


def backfill_mert_linear():
    method_cfg = _load_yaml(EXP_ROOT / "configs/experiments/exp01_transfer/mert_linear_probe_v1.yaml")
    dataset_cfg = _load_yaml(DATASET_CFG_PATH)
    sup_cfg = _load_yaml(SUPERVISED_CFG_PATH)
    model_cfg = method_cfg["model_config"]

    for seed_name in SEED_NAMES:
        seed_cfg = _load_yaml(EXP_ROOT / f"configs/data/{seed_name}.yaml")
        output_dir = EXP_ROOT / f"runs/exp01_transfer/mert_linear/{seed_name}"
        if not output_dir.exists():
            print(f"  SKIP {output_dir} (not found)")
            continue

        resolved = {
            "dataset": dataset_cfg,
            "seed": seed_cfg,
            "method": method_cfg["method"],
            "model": method_cfg["model"],
            "experiment_id": method_cfg["experiment_id"],
            "version": method_cfg["version"],
            "model_config": model_cfg,
            "training": sup_cfg["training"],
            "evaluation": sup_cfg["evaluation"],
            "device": sup_cfg.get("device", "auto"),
            "output": method_cfg["output"],
        }
        write_config_snapshot(resolved, output_dir / "config_snapshot.yaml")
        print(f"  OK {output_dir / 'config_snapshot.yaml'}")


def main():
    print("Backfilling config_snapshot.yaml for 12 Phase 1 runs...")
    print("\n[mfcc_svm]")
    backfill_mfcc_svm()
    print("\n[clap_zeroshot]")
    backfill_clap_zeroshot()
    print("\n[clap_linear]")
    backfill_clap_linear()
    print("\n[mert_linear]")
    backfill_mert_linear()
    print("\nDone.")


if __name__ == "__main__":
    main()
