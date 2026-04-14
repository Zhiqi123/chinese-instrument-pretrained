"""
Single layer x seed MERT layer probe runner.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "instrument_cls_experiments"))

from src.data.label_map import load_label_map
from src.data.manifest_loader import load_seed_data
from src.eval.confusion import compute_confusion_matrix, write_confusion_matrix
from src.eval.cost_metrics import write_cost_metrics
from src.eval.metrics import compute_metrics, compute_per_class_metrics
from src.eval.prediction_writer import write_per_class_metrics, write_predictions, write_metrics
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


def parse_args():
    parser = argparse.ArgumentParser(description="MERT layer probe single layer×seed runner")
    parser.add_argument("--config", required=True, help="Method config YAML")
    parser.add_argument("--seed-name", required=True, choices=["seed0", "seed1", "seed2"])
    parser.add_argument("--layer-index", required=True, type=int, help="Layer index (0..12)")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_layer_embeddings(
    cache_dir: Path,
    seed_name: str,
    split: str,
    layer_index: int,
) -> dict:
    """Load embeddings for a given layer from NPZ cache."""
    npz_path = cache_dir / seed_name / f"{split}_all_layers.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Layer cache not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    layer_key = f"layer_{layer_index:02d}"
    if layer_key not in data:
        raise KeyError(f"{layer_key} not in NPZ, available keys: {list(data.keys())}")

    return {
        "embeddings": torch.from_numpy(data[layer_key]),
        "label_ids": torch.from_numpy(data["label_id"]),
        "segment_ids": data["segment_id"].tolist(),
    }


def main():
    args = parse_args()
    started_at = datetime.now(timezone.utc).isoformat()

    # Load configs
    method_cfg = _load_yaml(Path(args.config))
    seed_cfg_path = EXP_ROOT / f"configs/data/{args.seed_name}.yaml"
    seed_cfg = _load_yaml(seed_cfg_path)
    sup_cfg = _load_yaml(SUPERVISED_CFG)

    seed_name = seed_cfg["seed_name"]
    seed_index = seed_cfg["seed_index"]
    base_seed = seed_cfg["base_seed"]
    layer_idx = args.layer_index
    method_short = "mert_layer_probe"
    layer_name = "embedding_output" if layer_idx == 0 else f"transformer_layer_{layer_idx}"
    run_id = f"run_exp04_{method_short}_layer{layer_idx:02d}_{seed_name}"

    print(f"=== MERT Layer Probe — layer {layer_idx} ({layer_name}), {seed_name} ===")

    set_global_seed(base_seed)

    # Output directory
    output_dir = EXP_ROOT / method_cfg["output"]["base_dir"] / f"layer_{layer_idx:02d}" / seed_name
    cache_dir = EXP_ROOT / method_cfg["output"]["cache_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(sup_cfg.get("device", "auto"))
    print(f"  Device: {device}")

    label_map = load_label_map(DATASET_CFG)
    num_classes = label_map["num_classes"]

    # Detect hidden_size and num_layers from NPZ
    probe_npz = np.load(
        cache_dir / seed_name / "train_all_layers.npz", allow_pickle=True,
    )
    layer_keys = sorted([k for k in probe_npz.keys() if k.startswith("layer_")])
    num_layers = len(layer_keys)
    hidden_size = probe_npz[layer_keys[0]].shape[1]
    probe_npz.close()

    if layer_idx < 0 or layer_idx >= num_layers:
        print(f"  ERROR: layer_index={layer_idx} out of range [0, {num_layers - 1}]")
        sys.exit(1)

    print(f"  num_layers={num_layers}, hidden_size={hidden_size}")

    # Load layer embeddings
    train_emb = _load_layer_embeddings(cache_dir, seed_name, "train", layer_idx)
    val_emb = _load_layer_embeddings(cache_dir, seed_name, "val", layer_idx)

    # Build DataLoaders
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

    # Train
    t_train_start = time.perf_counter()
    trainer = LinearProbeTrainer(
        embedding_dim=hidden_size,
        num_classes=num_classes,
        train_cfg=sup_cfg,
        device=device,
        base_seed=base_seed,
        output_dir=output_dir,
    )
    trainer.train(train_loader, val_loader)
    train_wall_time = time.perf_counter() - t_train_start

    # cost_metrics.json
    layer_cost = {
        "method": method_short,
        "model": "mert_v1_95m",
        "seed_name": seed_name,
        "layer_index": layer_idx,
        "layer_name": layer_name,
        "device_type": str(device.type),
        "trainable_params": hidden_size * num_classes + num_classes,
        "feature_prep_time_sec": 0.0,
        "train_wall_time_sec": round(train_wall_time, 3),
        "best_epoch": trainer.best_epoch,
        "stopped_epoch": trainer.stopped_epoch,
    }
    write_cost_metrics(layer_cost, output_dir / "cost_metrics.json")

    # Load unified_df for segment metadata
    unified_df = load_seed_data(DATASET_CFG, seed_cfg_path, PROJECT_ROOT)
    seg_to_source = dict(zip(unified_df["segment_id"], unified_df["source_dataset"]))
    seg_to_sample = dict(zip(unified_df["segment_id"], unified_df["sample_id"]))
    seg_to_record = dict(zip(unified_df["segment_id"], unified_df["record_id"]))

    # Evaluate val / test / external_test
    for split in ["val", "test", "external_test"]:
        split_emb = _load_layer_embeddings(cache_dir, seed_name, split, layer_idx)
        eval_loader = build_embedding_dataloader(
            split_emb,
            batch_size=sup_cfg["evaluation"]["batch_size"],
            shuffle=False,
        )
        pred_ids, top1_scores = trainer.predict(eval_loader)
        true_ids = split_emb["label_ids"].tolist()
        segment_ids = split_emb["segment_ids"]

        predictions = []
        for i in range(len(pred_ids)):
            seg_id = segment_ids[i]
            predictions.append({
                "segment_id": seg_id,
                "sample_id": seg_to_sample.get(seg_id, "unknown"),
                "record_id": seg_to_record.get(seg_id, "unknown"),
                "split": split,
                "source_dataset": seg_to_source.get(seg_id, "unknown"),
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

        print(f"  {split}: acc={metrics['accuracy']:.3f}, macro_f1={metrics['macro_f1']:.3f}")

    # config_snapshot + run_meta + registry
    resolved_config = {
        "dataset": _load_yaml(DATASET_CFG),
        "seed": seed_cfg,
        "method": method_cfg["method"],
        "model": method_cfg["model"],
        "experiment_id": method_cfg["experiment_id"],
        "version": method_cfg["version"],
        "model_config": method_cfg["model_config"],
        "training": sup_cfg["training"],
        "evaluation": sup_cfg["evaluation"],
        "device": sup_cfg.get("device", "auto"),
        "output": method_cfg["output"],
        "layer_index": layer_idx,
        "layer_name": layer_name,
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
            "phase": 4,
            "experiment_id": method_cfg["experiment_id"],
            "task_type": "layer_probe",
            "model": "mert_v1_95m",
            "method_short": method_short,
            "layer_index": layer_idx,
            "layer_name": layer_name,
            "num_encoder_layers": num_layers - 1,
            "pooling": "mean",
            "label_map_version": "v1",
            "source_embedding_cache_dir": str(
                (cache_dir / seed_name).relative_to(EXP_ROOT)
            ),
        },
    )

    update_registry(
        run_meta_path=output_dir / "run_meta.json",
        registry_csv_path=REGISTRY_CSV,
        experiment_id=method_cfg["experiment_id"],
        phase=4,
        task_type="layer_probe",
        model="mert_v1_95m",
        notes=f"Phase 4; method_short={method_short}; layer_index={layer_idx}",
    )

    print(f"\n  Layer probe layer{layer_idx:02d}/{seed_name}: DONE (run_id={run_id})")


if __name__ == "__main__":
    main()
