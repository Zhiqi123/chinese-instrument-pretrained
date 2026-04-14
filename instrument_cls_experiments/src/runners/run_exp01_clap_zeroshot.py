# Exp 1: CLAP zero-shot single-seed evaluation.

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "instrument_cls_experiments"))

from src.data.dataloaders import build_dataloader
from src.data.label_map import load_label_map
from src.data.manifest_loader import load_seed_data
from src.eval.confusion import compute_confusion_matrix, write_confusion_matrix
from src.eval.metrics import compute_metrics, compute_per_class_metrics
from src.eval.prediction_writer import write_per_class_metrics, write_predictions, write_metrics
from src.eval.registry_writer import update_registry
from src.eval.run_meta import write_config_snapshot, write_run_meta
from src.models.clap_zeroshot import ClapZeroShotClassifier
from src.utils.seed import get_device, set_global_seed


EXP_ROOT = PROJECT_ROOT / "instrument_cls_experiments"
DATASET_CFG = EXP_ROOT / "configs/data/dataset_v1.yaml"
REGISTRY_CSV = EXP_ROOT / "registry/run_status.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="CLAP zero-shot single-seed runner")
    parser.add_argument("--config", required=True, help="Method config YAML")
    parser.add_argument("--seed-name", required=True, choices=["seed0", "seed1", "seed2"])
    return parser.parse_args()


def resolve_seed_config(seed_name: str) -> Path:
    return EXP_ROOT / f"configs/data/{seed_name}.yaml"


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    started_at = datetime.now(timezone.utc).isoformat()

    # Load configs
    method_cfg = _load_yaml(Path(args.config))
    seed_cfg_path = resolve_seed_config(args.seed_name)
    seed_cfg = _load_yaml(seed_cfg_path)

    seed_name = seed_cfg["seed_name"]
    seed_index = seed_cfg["seed_index"]
    base_seed = seed_cfg["base_seed"]
    method_short = "clap_zeroshot"
    run_id = f"run_exp01_{method_short}_{seed_name}"

    print(f"=== CLAP Zero-Shot — {seed_name} (seed={base_seed}) ===")

    # Set global seed
    set_global_seed(base_seed)

    # Output directory
    output_dir = EXP_ROOT / method_cfg["output"]["base_dir"] / seed_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data and model
    label_map = load_label_map(DATASET_CFG)
    unified_df = load_seed_data(DATASET_CFG, seed_cfg_path, PROJECT_ROOT)

    model_cfg = method_cfg["model_config"]
    inf_cfg = method_cfg["inference"]
    device = get_device("auto")

    print(f"  Device: {device}")
    print("  Loading CLAP model...")
    prompt_cfg_path = EXP_ROOT / model_cfg["prompt_config"]
    cache_dir = PROJECT_ROOT / model_cfg["model_cache_dir"]

    classifier = ClapZeroShotClassifier(
        model_name=model_cfg["model_name"],
        prompt_config_path=prompt_cfg_path,
        label_map=label_map,
        device=str(device),
        cache_dir=cache_dir,
    )

    # Evaluate
    eval_splits = method_cfg["eval_splits"]
    batch_size = inf_cfg["batch_size"]
    num_workers = inf_cfg["num_workers"]

    print("  Evaluating...")
    for split in eval_splits:
        loader = build_dataloader(
            unified_df, split, batch_size=batch_size, num_workers=num_workers,
        )

        all_preds = []
        all_true = []
        predictions = []

        for batch in loader:
            wf = batch["waveform"]
            lid = batch["label_id"]
            meta = batch["metadata"]

            pred_ids, top1_scores = classifier.predict_batch(wf, source_sr=24000)

            for i in range(len(pred_ids)):
                true_lid = int(lid[i])
                pred_lid = int(pred_ids[i])
                all_true.append(true_lid)
                all_preds.append(pred_lid)
                predictions.append({
                    "segment_id": meta[i]["segment_id"],
                    "sample_id": meta[i]["sample_id"],
                    "record_id": meta[i]["record_id"],
                    "split": split,
                    "source_dataset": meta[i]["source_dataset"],
                    "true_label": label_map["id_to_label"][true_lid],
                    "true_label_id": true_lid,
                    "pred_label": label_map["id_to_label"][pred_lid],
                    "pred_label_id": pred_lid,
                    "top1_score": float(top1_scores[i]),
                })

        # Metrics
        metrics = compute_metrics(all_true, all_preds, split)
        per_class = compute_per_class_metrics(all_true, all_preds, label_map)
        cm_df = compute_confusion_matrix(all_true, all_preds, label_map)

        # Write outputs
        write_predictions(predictions, output_dir / f"{split}_predictions.csv")
        write_metrics(metrics, output_dir / f"{split}_metrics.json")
        write_per_class_metrics(per_class, output_dir / f"{split}_per_class_metrics.csv")
        write_confusion_matrix(cm_df, output_dir / f"{split}_confusion_matrix.csv")

        print(f"    {split}: n={metrics['num_samples']}, "
              f"acc={metrics['accuracy']:.3f}, macro_f1={metrics['macro_f1']:.3f}")

    # Write config snapshot, run meta, registry
    prompt_cfg_data = _load_yaml(EXP_ROOT / model_cfg["prompt_config"])
    inf_default = _load_yaml(EXP_ROOT / "configs/train/default_inference.yaml")
    resolved_config = {
        "dataset": _load_yaml(DATASET_CFG),
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
        "prompt_config": prompt_cfg_data,
        "eval_splits": method_cfg["eval_splits"],
        "output": method_cfg["output"],
    }
    write_config_snapshot(resolved_config, output_dir / "config_snapshot.yaml")

    finished_at = datetime.now(timezone.utc).isoformat()
    write_run_meta(
        run_id=run_id,
        seed_name=seed_name,
        seed_index=seed_index,
        base_seed=base_seed,
        config_version=method_cfg["version"],
        started_at=started_at,
        finished_at=finished_at,
        output_path=output_dir / "run_meta.json",
        extra={
            "task_type": "zero_shot",
            "model": "clap_htsat_unfused",
            "model_name": model_cfg["model_name"],
            "model_cache_dir": model_cfg["model_cache_dir"],
        },
    )

    update_registry(
        run_meta_path=output_dir / "run_meta.json",
        registry_csv_path=REGISTRY_CSV,
        experiment_id=method_cfg["experiment_id"],
        phase=1,
        task_type="zero_shot",
        model="clap_htsat_unfused",
        notes=f"seed={base_seed}",
    )

    print(f"  CLAP zero-shot {seed_name}: DONE")


if __name__ == "__main__":
    main()
