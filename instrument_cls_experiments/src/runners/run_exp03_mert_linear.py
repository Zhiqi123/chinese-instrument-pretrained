"""
Phase 3 MERT linear probe single seed x ratio runner.
"""

from __future__ import annotations

import argparse
import gc
import shutil
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
from src.data.train_subset_loader import load_train_subset
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
from src.train.linear_probe_trainer import LinearProbeTrainer, build_embedding_dataloader
from src.utils.seed import (
    build_train_generator,
    build_worker_init_fn,
    get_device,
    set_global_seed,
)


EXP_ROOT = PROJECT_ROOT / "instrument_cls_experiments"
DATASET_CFG = EXP_ROOT / "configs/data/dataset_v1.yaml"
SUPERVISED_CFG = EXP_ROOT / "configs/train/default_supervised.yaml"
REGISTRY_CSV = EXP_ROOT / "registry/run_status.csv"
SUBSETS_DIR = EXP_ROOT / "artifacts/exp03_train_subsets"


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3 MERT linear probe single run")
    parser.add_argument("--config", required=True, help="Method config YAML")
    parser.add_argument("--seed-name", required=True, choices=["seed0", "seed1", "seed2"])
    parser.add_argument("--ratio-name", required=True,
                        choices=["train10", "train25", "train50", "train100"])
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    started_at = datetime.now(timezone.utc).isoformat()

    # Load configs

    seed_name = seed_cfg["seed_name"]
    seed_index = seed_cfg["seed_index"]
    base_seed = seed_cfg["base_seed"]
    method_short = method_cfg["method_short"]
    ratio_name = args.ratio_name
    ratio_value = subsample_cfg["ratio_definitions"][ratio_name]
    subset_seed = base_seed + subsample_cfg["ratio_offsets"][ratio_name]

    run_id = f"run_exp03_{method_short}_{ratio_name}_{seed_name}"

    print(f"=== [Phase 3] MERT Linear Probe — {ratio_name} × {seed_name} "
          f"(seed={base_seed}, subset_seed={subset_seed}) ===")

    # Set global seed
    set_global_seed(base_seed)

    # Output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(sup_cfg.get("device", "auto"))
    print(f"  Device: {device}")

    # Load data
    label_map = load_label_map(DATASET_CFG)
    unified_df = load_seed_data(DATASET_CFG, seed_cfg_path, PROJECT_ROOT)
    print(f"  Total segments (full): {len(unified_df)}")

    # Apply train subset
    subset_segments_csv = SUBSETS_DIR / ratio_name / seed_name / "train_subset_segments.csv"
    subset_meta_json = SUBSETS_DIR / ratio_name / seed_name / "subset_meta.json"

    filtered_df, subset_meta = load_train_subset(
        unified_df, subset_segments_csv, subset_meta_json,
    )
    n_train_subset = (filtered_df["split"] == "train").sum()
    print(f"  Train subset: {n_train_subset} segments "
          f"(ratio={ratio_value}, groups={subset_meta['subset_train_group_count']})")

    # Copy frozen subset files to run artifacts
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(subset_segments_csv, artifacts_dir / "train_subset_segments.csv")
    shutil.copy2(
        SUBSETS_DIR / ratio_name / seed_name / "train_subset_groups.csv",
        artifacts_dir / "train_subset_groups.csv",
    )

    # Stage 1: Extract embeddings (subset train + full val/test/external_test)
    print("  Phase 1: Extracting MERT embeddings...")
    from src.models.mert_linear import MERTLinearProbe

    model_cfg = method_cfg["model_config"]
    cache_dir = PROJECT_ROOT / model_cfg["model_cache_dir"]
    model = MERTLinearProbe(
        num_classes=label_map["num_classes"],
        model_name=model_cfg["model_name"],
        freeze_encoder=model_cfg["freeze_encoder"],
        cache_dir=cache_dir,
    )
    model = model.to(device)
    embedding_dim = model.encoder.config.hidden_size
    print(f"    embedding_dim={embedding_dim}")

    inf_batch_size = sup_cfg["evaluation"]["batch_size"]
    inf_num_workers = sup_cfg["evaluation"]["num_workers"]

    t_feature_start = time.perf_counter()
    for split in ["train", "val", "test", "external_test"]:
        loader = build_dataloader(
            filtered_df, split, batch_size=inf_batch_size, num_workers=inf_num_workers,
        )
        emb_data = model.extract_from_dataloader(loader)
        torch.save(emb_data, emb_dir / f"{split}.pt")
        print(f"    {split}: {emb_data['embeddings'].shape[0]} samples, "
              f"dim={emb_data['embeddings'].shape[1]}")
    feature_prep_time_sec = time.perf_counter() - t_feature_start

    param_stats = count_parameters(model)

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Stage 2: Train linear head
    print("  Phase 2: Training linear head...")

    train_emb = torch.load(emb_dir / "train.pt", weights_only=False)
    val_emb = torch.load(emb_dir / "val.pt", weights_only=False)

    train_loader = build_embedding_dataloader(
        train_emb,
        batch_size=sup_cfg["training"]["batch_size"],
        shuffle=True,
        generator=build_train_generator(base_seed),
        worker_init_fn=build_worker_init_fn(base_seed),
    )
    val_loader = build_embedding_dataloader(
        val_emb,
        batch_size=sup_cfg["evaluation"]["batch_size"],
        shuffle=False,
    )

    trainer = LinearProbeTrainer(
        embedding_dim=embedding_dim,
        num_classes=label_map["num_classes"],
        train_cfg=sup_cfg,
        device=device,
        base_seed=base_seed,
        output_dir=output_dir,
    )

    reset_peak_memory()

    t_train_start = time.perf_counter()
    trainer.train(train_loader, val_loader)
    train_wall_time_sec = time.perf_counter() - t_train_start

    peak_memory = read_peak_memory()

    # Stage 3: Evaluate
    print("  Phase 3: Evaluating...")

    seg_to_source = dict(zip(unified_df["segment_id"], unified_df["source_dataset"]))

    for split in ["val", "test", "external_test"]:
        emb_data = torch.load(emb_dir / f"{split}.pt", weights_only=False)
        eval_loader = build_embedding_dataloader(
            emb_data, batch_size=sup_cfg["evaluation"]["batch_size"], shuffle=False,
        )
        pred_ids, top1_scores = trainer.predict(eval_loader)
        true_ids = emb_data["label_ids"].tolist()

        predictions = []
        for i in range(len(pred_ids)):
            true_lid = true_ids[i]
            pred_lid = pred_ids[i]
            seg_id = emb_data["segment_ids"][i]
            predictions.append({
                "segment_id": seg_id,
                "sample_id": emb_data["sample_ids"][i],
                "record_id": emb_data["record_ids"][i],
                "split": split,
                "source_dataset": seg_to_source[seg_id],
                "true_label": label_map["id_to_label"][true_lid],
                "true_label_id": true_lid,
                "pred_label": label_map["id_to_label"][pred_lid],
                "pred_label_id": pred_lid,
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

    # cost_metrics
    cost_dict = build_cost_metrics(
        method=method_short,
        model_name=method_cfg["model"],
        seed_name=seed_name,
        device_type=str(device.type),
        param_stats=param_stats,
        feature_prep_time_sec=feature_prep_time_sec,
        train_wall_time_sec=train_wall_time_sec,
        epoch_times=trainer.epoch_times,
        step_times_ms=trainer.step_times_ms,
        peak_memory=peak_memory,
        best_epoch=trainer.best_epoch,
        stopped_epoch=trainer.stopped_epoch,
    )
    write_cost_metrics(cost_dict, output_dir / "cost_metrics.json")

    # config_snapshot + run_meta + registry
    dataset_cfg = _load_yaml(DATASET_CFG)
    resolved_config = {
        "dataset": dataset_cfg,
        "seed": seed_cfg,
        "subsample": subsample_cfg,
        "ratio_name": ratio_name,
        "ratio_value": ratio_value,
        "subset_seed": subset_seed,
        "method": method_cfg["method"],
        "method_short": method_short,
        "model": method_cfg["model"],
        "experiment_id": method_cfg["experiment_id"],
        "version": method_cfg["version"],
        "phase": method_cfg["phase"],
        "model_config": model_cfg,
        "training": sup_cfg["training"],
        "evaluation": sup_cfg["evaluation"],
        "device": sup_cfg.get("device", "auto"),
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
            "task_type": "linear_probe",
            "model_name": model_cfg["model_name"],
            "model_cache_dir": model_cfg["model_cache_dir"],
            "adaptation_mode": "linear_probe",
            "lora_rank": None,
            "total_params": param_stats["total_params"],
            "trainable_params": param_stats["trainable_params"],
            "trainable_ratio": param_stats["trainable_ratio"],
            "embedding_dim": embedding_dim,
            "train_ratio_name": ratio_name,
            "train_ratio_value": ratio_value,
            "subset_seed": subset_seed,
            "subset_sampling_unit": "recording_group_id",
            "full_train_group_count": subset_meta["full_train_group_count"],
            "full_train_record_count": subset_meta["full_train_record_count"],
            "full_train_segment_count": subset_meta["full_train_segment_count"],
            "subset_train_group_count": subset_meta["subset_train_group_count"],
            "subset_train_record_count": subset_meta["subset_train_record_count"],
            "subset_train_segment_count": subset_meta["subset_train_segment_count"],
        },
    )

    update_registry(
        run_meta_path=output_dir / "run_meta.json",
        registry_csv_path=REGISTRY_CSV,
        experiment_id=method_cfg["experiment_id"],
        phase=3,
        task_type="linear_probe",
        model="mert_v1_95m",
        notes=f"Phase 3; ratio={ratio_name}; seed={base_seed}; emb_dim={embedding_dim}",
    )

    print(f"  [Phase 3] MERT linear probe {ratio_name} × {seed_name}: DONE")


if __name__ == "__main__":
    main()
