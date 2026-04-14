# Exp 2: MERT full fine-tuning single-seed run.

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "instrument_cls_experiments"))

from src.data.dataloaders import build_dataloader
from src.data.label_map import load_label_map
from src.data.manifest_loader import load_seed_data
from src.eval.confusion import compute_confusion_matrix, write_confusion_matrix
from src.eval.cost_metrics import (
    build_cost_metrics,
    count_parameters,
    read_peak_memory,
    reset_peak_memory,
    write_cost_metrics,
)
from src.eval.metrics import compute_metrics, compute_per_class_metrics
from src.eval.prediction_writer import write_metrics, write_per_class_metrics, write_predictions
from src.eval.registry_writer import update_registry
from src.eval.run_meta import write_config_snapshot, write_run_meta
from src.utils.seed import (
    build_train_generator,
    build_worker_init_fn,
    get_device,
    set_global_seed,
)


EXP_ROOT = PROJECT_ROOT / "instrument_cls_experiments"
DATASET_CFG = EXP_ROOT / "configs/data/dataset_v1.yaml"
ADAPTATION_CFG = EXP_ROOT / "configs/train/default_adaptation.yaml"
REGISTRY_CSV = EXP_ROOT / "registry/run_status.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="MERT full fine-tuning single-seed runner")
    parser.add_argument("--config", required=True, help="Method config YAML")
    parser.add_argument("--seed-name", required=True, choices=["seed0", "seed1", "seed2"])
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _merge_training_config(adapt_cfg: dict, method_cfg: dict) -> dict:
    """Merge default_adaptation.yaml with method-specific training overrides."""
    merged = dict(adapt_cfg)
    method_training = method_cfg.get("training", {})
    if "optimizer" in method_training:
        merged_opt = dict(merged["training"]["optimizer"])
        merged_opt.update(method_training["optimizer"])
        merged["training"] = dict(merged["training"])
        merged["training"]["optimizer"] = merged_opt
    return merged


def main():
    args = parse_args()
    started_at = datetime.now(timezone.utc).isoformat()

    # Load configs
    method_cfg = _load_yaml(Path(args.config))
    seed_cfg_path = EXP_ROOT / f"configs/data/{args.seed_name}.yaml"
    seed_cfg = _load_yaml(seed_cfg_path)
    adapt_cfg = _load_yaml(ADAPTATION_CFG)

    merged_cfg = _merge_training_config(adapt_cfg, method_cfg)

    seed_name = seed_cfg["seed_name"]
    seed_index = seed_cfg["seed_index"]
    base_seed = seed_cfg["base_seed"]
    method_short = method_cfg["method_short"]
    run_id = f"run_exp02_{method_short}_{seed_name}"
    model_cfg = method_cfg["model_config"]

    print(f"=== [Phase 2] MERT Full Fine-Tuning — {seed_name} (seed={base_seed}) ===")

    # Set global seed
    set_global_seed(base_seed)

    # Output directory
    output_dir = EXP_ROOT / method_cfg["output"]["base_dir"] / seed_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(merged_cfg.get("device", "auto"))
    print(f"  Device: {device}")

    # Load data
    label_map = load_label_map(DATASET_CFG)
    unified_df = load_seed_data(DATASET_CFG, seed_cfg_path, PROJECT_ROOT)
    print(f"  Total segments: {len(unified_df)}")

    # Create model
    from src.models.mert_adaptation import MERTForAdaptation

    cache_dir = PROJECT_ROOT / model_cfg["model_cache_dir"]

    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)

    model = MERTForAdaptation(
        num_classes=label_map["num_classes"],
        model_name=model_cfg["model_name"],
        cache_dir=cache_dir,
        adaptation_mode="full_ft",
    )
    model = model.to(device)

    param_stats = count_parameters(model)
    print(f"  Parameters: total={param_stats['total_params']}, "
          f"trainable={param_stats['trainable_params']}, "
          f"ratio={param_stats['trainable_ratio']:.6f}")

    # Build DataLoaders
    train_batch = merged_cfg["training"]["batch_size"]
    eval_batch = merged_cfg["evaluation"]["batch_size"]
    num_workers = merged_cfg["training"]["num_workers"]

    train_loader = build_dataloader(
        unified_df, "train",
        batch_size=train_batch,
        num_workers=num_workers,
        generator=build_train_generator(base_seed),
        worker_init_fn=build_worker_init_fn(base_seed),
    )
    val_loader = build_dataloader(
        unified_df, "val",
        batch_size=eval_batch,
        num_workers=merged_cfg["evaluation"]["num_workers"],
    )

    # Train
    from src.train.supervised_audio_trainer import SupervisedAudioTrainer

    trainer = SupervisedAudioTrainer(
        model=model,
        train_cfg=merged_cfg,
        device=device,
        base_seed=base_seed,
        output_dir=output_dir,
    )

    reset_peak_memory()
    t_train_start = time.perf_counter()
    trainer.train(train_loader, val_loader)
    train_wall_time_sec = time.perf_counter() - t_train_start
    peak_memory = read_peak_memory()

    # Evaluate
    print("  Evaluating...")
    seg_to_source = dict(zip(unified_df["segment_id"], unified_df["source_dataset"]))

    for split in ["val", "test", "external_test"]:
        loader = build_dataloader(
            unified_df, split,
            batch_size=eval_batch,
            num_workers=merged_cfg["evaluation"]["num_workers"],
        )
        pred_ids, top1_scores = trainer.predict(loader)

        # Collect metadata and true labels
        split_loader = build_dataloader(unified_df, split, batch_size=eval_batch, num_workers=0)
        meta_list = []
        true_ids = []
        for batch in split_loader:
            for i, meta in enumerate(batch["metadata"]):
                meta_list.append(meta)
                true_ids.append(batch["label_id"][i].item())

        assert len(meta_list) == len(pred_ids), f"meta count {len(meta_list)} != pred count {len(pred_ids)}"

        predictions = []
        for i in range(len(pred_ids)):
            meta = meta_list[i]
            predictions.append({
                "segment_id": meta["segment_id"],
                "sample_id": meta["sample_id"],
                "record_id": meta["record_id"],
                "split": split,
                "source_dataset": meta["source_dataset"],
                "true_label": label_map["id_to_label"][true_ids[i]],
                "true_label_id": true_ids[i],
                "pred_label": label_map["id_to_label"][pred_ids[i]],
                "pred_label_id": pred_ids[i],
                "top1_score": top1_scores[i],
            })

        metrics = compute_metrics(true_ids, pred_ids, split)
        per_class = compute_per_class_metrics(true_ids, pred_ids, label_map)
        cm_df = compute_confusion_matrix(true_ids, pred_ids, label_map)

        if split == "val":
            write_predictions(predictions, output_dir / "val_predictions_best.csv")
            write_metrics(metrics, output_dir / "val_metrics_best.json")
            write_per_class_metrics(per_class, output_dir / "val_per_class_metrics_best.csv")
            write_confusion_matrix(cm_df, output_dir / "val_confusion_matrix_best.csv")
        else:
            write_predictions(predictions, output_dir / f"{split}_predictions.csv")
            write_metrics(metrics, output_dir / f"{split}_metrics.json")
            write_per_class_metrics(per_class, output_dir / f"{split}_per_class_metrics.csv")
            write_confusion_matrix(cm_df, output_dir / f"{split}_confusion_matrix.csv")

        print(f"    {split}: n={metrics['num_samples']}, "
              f"acc={metrics['accuracy']:.3f}, macro_f1={metrics['macro_f1']:.3f}")

    # Cost metrics
    cost_dict = build_cost_metrics(
        method=method_short,
        model_name=method_cfg["model"],
        seed_name=seed_name,
        device_type=str(device.type),
        param_stats=param_stats,
        feature_prep_time_sec=0.0,
        train_wall_time_sec=train_wall_time_sec,
        epoch_times=trainer.epoch_times,
        step_times_ms=trainer.step_times_ms,
        peak_memory=peak_memory,
        best_epoch=trainer.best_epoch,
        stopped_epoch=trainer.stopped_epoch,
    )
    write_cost_metrics(cost_dict, output_dir / "cost_metrics.json")

    # Config snapshot, run meta, registry
    dataset_cfg = _load_yaml(DATASET_CFG)
    resolved_config = {
        "dataset": dataset_cfg,
        "seed": seed_cfg,
        "method": method_cfg["method"],
        "method_short": method_short,
        "model": method_cfg["model"],
        "experiment_id": method_cfg["experiment_id"],
        "version": method_cfg["version"],
        "model_config": model_cfg,
        "training": merged_cfg["training"],
        "evaluation": merged_cfg["evaluation"],
        "device": merged_cfg.get("device", "auto"),
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
            "task_type": "full_finetune",
            "model_name": model_cfg["model_name"],
            "model_cache_dir": model_cfg["model_cache_dir"],
            "adaptation_mode": "full_ft",
            "lora_rank": None,
            "total_params": param_stats["total_params"],
            "trainable_params": param_stats["trainable_params"],
            "trainable_ratio": param_stats["trainable_ratio"],
        },
    )

    update_registry(
        run_meta_path=output_dir / "run_meta.json",
        registry_csv_path=REGISTRY_CSV,
        experiment_id=method_cfg["experiment_id"],
        phase=2,
        task_type="full_finetune",
        model="mert_v1_95m",
        notes=f"Phase 2; seed={base_seed}; full fine-tuning",
    )

    print(f"  [Phase 2] MERT full fine-tuning {seed_name}: DONE")


if __name__ == "__main__":
    main()
