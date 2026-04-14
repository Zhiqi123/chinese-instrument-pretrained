# Exp 1: MFCC+SVM single-seed run.

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "instrument_cls_experiments"))

from src.baselines.mfcc_svm import MFCCSVMPipeline, extract_features_from_dataloader
from src.data.dataloaders import build_dataloader
from src.data.label_map import load_label_map
from src.data.manifest_loader import load_seed_data
from src.eval.confusion import compute_confusion_matrix, write_confusion_matrix
from src.eval.metrics import compute_metrics, compute_per_class_metrics
from src.eval.prediction_writer import write_per_class_metrics, write_predictions, write_metrics
from src.eval.registry_writer import update_registry
from src.eval.run_meta import write_config_snapshot, write_run_meta
from src.utils.seed import set_global_seed



EXP_ROOT = PROJECT_ROOT / "instrument_cls_experiments"
DATASET_CFG = EXP_ROOT / "configs/data/dataset_v1.yaml"
REGISTRY_CSV = EXP_ROOT / "registry/run_status.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="MFCC+SVM single-seed runner")
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
    dataset_cfg = _load_yaml(DATASET_CFG)

    seed_name = seed_cfg["seed_name"]
    seed_index = seed_cfg["seed_index"]
    base_seed = seed_cfg["base_seed"]
    method_short = "mfcc_svm"
    run_id = f"run_exp01_{method_short}_{seed_name}"

    print(f"=== MFCC+SVM — {seed_name} (seed={base_seed}) ===")

    # Set global seed
    set_global_seed(base_seed)

    # Output directory
    output_dir = EXP_ROOT / method_cfg["output"]["base_dir"] / seed_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    label_map = load_label_map(DATASET_CFG)
    unified_df = load_seed_data(DATASET_CFG, seed_cfg_path, PROJECT_ROOT)
    print(f"  Total segments: {len(unified_df)}")

    # Feature extraction
    feat_cfg = method_cfg["feature_extraction"]
    sr = feat_cfg["sample_rate"]
    n_mfcc = feat_cfg["n_mfcc"]
    n_fft = feat_cfg["n_fft"]
    hop_length = feat_cfg["hop_length"]

    print("  Extracting MFCC features...")
    split_data = {}
    for split in ["train", "val", "test", "external_test"]:
        loader = build_dataloader(unified_df, split, batch_size=32,
                                   num_workers=0, use_balanced_sampler=False)
        features, label_ids, metadata = extract_features_from_dataloader(
            loader, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length,
        )
        split_data[split] = (features, label_ids, metadata)
        print(f"    {split}: {features.shape[0]} samples, features={features.shape[1]}-dim")

    # Fit SVM
    cls_cfg = method_cfg["classifier"]
    pipeline = MFCCSVMPipeline(
        n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length,
        kernel=cls_cfg["kernel"], C=cls_cfg["C"], gamma=cls_cfg["gamma"],
        base_seed=base_seed,
    )

    train_features, train_labels, _ = split_data["train"]
    print(f"  Fitting SVM on {len(train_labels)} train samples...")
    pipeline.fit(train_features, train_labels)

    # Evaluate
    print("  Evaluating...")
    for split in ["val", "test", "external_test"]:
        features, true_label_ids, metadata = split_data[split]
        pred_ids, top1_scores = pipeline.predict(features)

        # Build predictions
        predictions = []
        for i in range(len(pred_ids)):
            true_lid = int(true_label_ids[i])
            pred_lid = int(pred_ids[i])
            predictions.append({
                "segment_id": metadata[i]["segment_id"],
                "sample_id": metadata[i]["sample_id"],
                "record_id": metadata[i]["record_id"],
                "split": split,
                "source_dataset": metadata[i]["source_dataset"],
                "true_label": label_map["id_to_label"][true_lid],
                "true_label_id": true_lid,
                "pred_label": label_map["id_to_label"][pred_lid],
                "pred_label_id": pred_lid,
                "top1_score": float(top1_scores[i]),
            })

        true_list = [int(x) for x in true_label_ids]
        pred_list = [int(x) for x in pred_ids]

        # Metrics
        metrics = compute_metrics(true_list, pred_list, split)
        per_class = compute_per_class_metrics(true_list, pred_list, label_map)
        cm_df = compute_confusion_matrix(true_list, pred_list, label_map)

        # Write outputs
        write_predictions(predictions, output_dir / f"{split}_predictions.csv")
        write_metrics(metrics, output_dir / f"{split}_metrics.json")
        write_per_class_metrics(per_class, output_dir / f"{split}_per_class_metrics.csv")
        write_confusion_matrix(cm_df, output_dir / f"{split}_confusion_matrix.csv")

        print(f"    {split}: n={metrics['num_samples']}, "
              f"acc={metrics['accuracy']:.3f}, macro_f1={metrics['macro_f1']:.3f}")

    # Save model
    pipeline.save(output_dir)

    # Config snapshot, run meta, registry
    resolved_config = {
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
        extra={"task_type": "instrument_classification", "model": "mfcc_svm"},
    )

    update_registry(
        run_meta_path=output_dir / "run_meta.json",
        registry_csv_path=REGISTRY_CSV,
        experiment_id=method_cfg["experiment_id"],
        phase=1,
        task_type="instrument_classification",
        model="mfcc_svm",
        notes=f"seed={base_seed}",
    )

    print(f"  MFCC+SVM {seed_name}: DONE")


if __name__ == "__main__":
    main()
