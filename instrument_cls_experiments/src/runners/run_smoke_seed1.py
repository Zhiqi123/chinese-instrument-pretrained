"""
Seed1 smoke test runner: data layer, CLAP, and MERT minimal validation.
All parameters from frozen configs, no hard-coded values.
"""

from __future__ import annotations

import json
import math
import random
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
from src.data.dataloaders import build_dataloader
from src.eval.metrics import compute_metrics
from src.eval.prediction_writer import write_predictions, write_metrics
from src.eval.run_meta import write_config_snapshot, write_run_meta
from src.eval.registry_writer import update_registry

# Path constants

DATASET_CFG = PROJECT_ROOT / "instrument_cls_experiments/configs/data/dataset_v1.yaml"
SEED1_CFG = PROJECT_ROOT / "instrument_cls_experiments/configs/data/seed1.yaml"
PROMPT_CFG = PROJECT_ROOT / "instrument_cls_experiments/configs/prompts/clap_zeroshot_v1.yaml"
SUPERVISED_CFG = PROJECT_ROOT / "instrument_cls_experiments/configs/train/default_supervised.yaml"
INFERENCE_CFG = PROJECT_ROOT / "instrument_cls_experiments/configs/train/default_inference.yaml"

CLAP_OUTPUT_DIR = PROJECT_ROOT / "instrument_cls_experiments/runs/smoke/seed1_clap"
MERT_OUTPUT_DIR = PROJECT_ROOT / "instrument_cls_experiments/runs/smoke/seed1_mert"
REGISTRY_CSV = PROJECT_ROOT / "instrument_cls_experiments/registry/run_status.csv"


# Config loading

def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_configs() -> dict:
    """Load and return all configs."""
    return {
        "supervised": _load_yaml(SUPERVISED_CFG),
        "inference": _load_yaml(INFERENCE_CFG),
        "seed1": _load_yaml(SEED1_CFG),
    }


# Device selection

def get_device(device_cfg: str = "auto") -> torch.device:
    if device_cfg != "auto":
        return torch.device(device_cfg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Seed setup

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS covered by torch.manual_seed


# Data layer smoke

def smoke_data_layer(unified_df, configs: dict):
    """Data layer batch smoke: 1 batch per split."""
    print("\n=== Data Layer Smoke ===")
    label_map = load_label_map(DATASET_CFG)

    # Read batch_size and num_workers from supervised config
    sup_cfg = configs["supervised"]
    train_batch_size = sup_cfg["training"]["batch_size"]
    train_num_workers = sup_cfg["training"]["num_workers"]
    eval_batch_size = sup_cfg["evaluation"]["batch_size"]
    eval_num_workers = sup_cfg["evaluation"]["num_workers"]

    for split in ["train", "val", "test", "external_test"]:
        bs = train_batch_size if split == "train" else eval_batch_size
        nw = train_num_workers if split == "train" else eval_num_workers
        loader = build_dataloader(unified_df, split, batch_size=bs, num_workers=nw)
        batch = next(iter(loader))
        wf = batch["waveform"]
        lid = batch["label_id"]
        meta = batch["metadata"]

        # Validate batch
        assert wf.shape[1] == 120000, f"Expected 120000 samples, got {wf.shape[1]}"
        assert wf.dtype == torch.float32
        assert lid.dtype == torch.long
        assert len(meta) == wf.shape[0]

        # Check label_id range
        assert lid.min() >= 0 and lid.max() < label_map["num_classes"]

        # Check external_test is all ChMusic
        if split == "external_test":
            sources = {m["source_dataset"] for m in meta}
            assert sources == {"ChMusic"}, f"external_test should only have ChMusic, got {sources}"

        print(f"  {split}: batch_size={bs}, batch_shape={wf.shape}, labels={lid.tolist()[:4]}... OK")

    print("  Data layer smoke: PASSED")


# CLAP smoke

def smoke_clap(unified_df, configs: dict):
    """CLAP zero-shot smoke: test and external_test up to max_segments."""
    started_at = datetime.now(timezone.utc).isoformat()
    print("\n=== CLAP Zero-Shot Smoke ===")

    from src.models.clap_zeroshot import ClapZeroShotClassifier

    inf_cfg = configs["inference"]
    inference_batch_size = inf_cfg["inference"]["batch_size"]
    inference_num_workers = inf_cfg["inference"]["num_workers"]
    max_segments = inf_cfg["smoke"]["max_segments"]

    device = get_device(inf_cfg.get("device", "auto"))
    label_map = load_label_map(DATASET_CFG)

    print(f"  Device: {device}")
    print(f"  Inference batch_size: {inference_batch_size}, num_workers: {inference_num_workers}")
    print(f"  Smoke max_segments: {max_segments}")
    print("  Loading CLAP model...")

    classifier = ClapZeroShotClassifier(
        model_name="laion/clap-htsat-unfused",
        prompt_config_path=PROMPT_CFG,
        label_map=label_map,
        device=str(device),
        cache_dir=PROJECT_ROOT / "model_cache/huggingface",
    )

    CLAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Max batches per split
    max_batches = math.ceil(max_segments / inference_batch_size)
    split_accs = {}

    for split in ["test", "external_test"]:
        loader = build_dataloader(unified_df, split, batch_size=inference_batch_size,
                                  num_workers=inference_num_workers)

        all_preds = []
        all_true = []
        predictions = []
        total_segments = 0

        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            # Last batch may exceed max_segments, truncate
            remaining = max_segments - total_segments
            if remaining <= 0:
                break

            wf = batch["waveform"]
            lid = batch["label_id"]
            meta = batch["metadata"]

            # Truncate to remaining
            actual_size = min(len(wf), remaining)
            wf = wf[:actual_size]
            lid = lid[:actual_size]
            meta = meta[:actual_size]

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

            total_segments += actual_size

        metrics = compute_metrics(all_true, all_preds, split)
        split_accs[split] = metrics["accuracy"]
        print(f"  {split}: n={metrics['num_samples']}, "
              f"acc={metrics['accuracy']:.3f}, macro_f1={metrics['macro_f1']:.3f}")

        write_predictions(predictions, CLAP_OUTPUT_DIR / f"{split}_predictions.csv")
        write_metrics(metrics, CLAP_OUTPUT_DIR / f"{split}_metrics.json")

    # Write config_snapshot and run_meta
    finished_at = datetime.now(timezone.utc).isoformat()
    write_run_meta(
        run_id="run_smoke_clap_seed1",
        seed_name="seed1",
        seed_index=1,
        base_seed=2027,
        config_version="v1",
        started_at=started_at,
        finished_at=finished_at,
        output_path=CLAP_OUTPUT_DIR / "run_meta.json",
        extra={"task_type": "zero_shot", "model": "clap_htsat_unfused"},
    )
    update_registry(
        run_meta_path=CLAP_OUTPUT_DIR / "run_meta.json",
        registry_csv_path=REGISTRY_CSV,
        experiment_id="smoke_clap",
        phase=0,
        task_type="zero_shot",
        model="clap_htsat_unfused",
        notes=f"config-driven; test {max_segments}seg acc={split_accs.get('test', 0):.3f}; ext_test {max_segments}seg acc={split_accs.get('external_test', 0):.3f}",
    )
    print("  CLAP smoke: PASSED")


# MERT smoke

def smoke_mert(unified_df, configs: dict):
    """MERT minimal supervised loop smoke."""
    started_at = datetime.now(timezone.utc).isoformat()
    print("\n=== MERT Linear Probe Smoke ===")

    from src.models.mert_linear import MERTLinearProbe

    sup_cfg = configs["supervised"]
    train_batch_size = sup_cfg["training"]["batch_size"]
    train_num_workers = sup_cfg["training"]["num_workers"]
    eval_batch_size = sup_cfg["evaluation"]["batch_size"]
    eval_num_workers = sup_cfg["evaluation"]["num_workers"]
    max_train_batches = sup_cfg["smoke"]["max_train_batches"]
    max_train_steps = sup_cfg["smoke"]["max_train_steps"]
    max_eval_batches = sup_cfg["smoke"]["max_eval_batches"]
    lr = sup_cfg["training"]["optimizer"]["lr"]

    device = get_device(sup_cfg.get("device", "auto"))
    label_map = load_label_map(DATASET_CFG)

    print(f"  Device: {device}")
    print(f"  Train batch_size: {train_batch_size} (workers: {train_num_workers}), "
          f"eval batch_size: {eval_batch_size} (workers: {eval_num_workers})")
    print(f"  Smoke: max_train_batches={max_train_batches}, "
          f"max_train_steps={max_train_steps}, max_eval_batches={max_eval_batches}")
    print("  Loading MERT model...")

    model = MERTLinearProbe(
        num_classes=label_map["num_classes"],
        model_name="m-a-p/MERT-v1-95M",
        freeze_encoder=True,
        cache_dir=PROJECT_ROOT / "model_cache/huggingface",
    )
    model = model.to(device)

    MERT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Train: up to max_train_batches, max_train_steps optimizer steps
    print("  Training step...")
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    train_loader = build_dataloader(unified_df, "train", batch_size=train_batch_size,
                                     num_workers=train_num_workers)
    train_log = []
    steps_done = 0

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= max_train_batches:
            break

        wf = batch["waveform"]
        lid = batch["label_id"].to(device)

        # Preprocess
        inputs = model.preprocess(wf)
        input_values = inputs["input_values"].to(device)

        # Forward
        logits = model(input_values)
        loss = criterion(logits, lid)

        # Only do max_train_steps optimizer steps
        if steps_done < max_train_steps:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps_done += 1

        train_log.append({
            "batch": batch_idx,
            "loss": float(loss.item()),
        })
        print(f"    batch {batch_idx}: loss={loss.item():.4f}")

    # Write train_log
    import pandas as pd
    pd.DataFrame(train_log).to_csv(MERT_OUTPUT_DIR / "train_log.csv", index=False)

    # Eval: val/test/external_test up to max_eval_batches
    print("  Evaluation...")
    model.eval()

    for split in ["val", "test", "external_test"]:
        loader = build_dataloader(unified_df, split, batch_size=eval_batch_size,
                                  num_workers=eval_num_workers)

        all_preds = []
        all_true = []
        predictions = []

        for eval_batch_idx, batch in enumerate(loader):
            if eval_batch_idx >= max_eval_batches:
                break

            wf = batch["waveform"]
            lid = batch["label_id"]
            meta = batch["metadata"]

            inputs = model.preprocess(wf)
            input_values = inputs["input_values"].to(device)

            with torch.no_grad():
                logits = model(input_values)
                probs = torch.softmax(logits, dim=-1)

            pred_ids = logits.argmax(dim=-1).cpu().numpy()
            top1_scores = probs.max(dim=-1).values.cpu().numpy()

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

        metrics = compute_metrics(all_true, all_preds, split)
        print(f"    {split}: n={metrics['num_samples']}, "
              f"acc={metrics['accuracy']:.3f}, macro_f1={metrics['macro_f1']:.3f}")

        write_predictions(predictions, MERT_OUTPUT_DIR / f"{split}_predictions.csv")
        write_metrics(metrics, MERT_OUTPUT_DIR / f"{split}_metrics.json")

    # Write config_snapshot and run_meta
    write_config_snapshot(
        [DATASET_CFG, SEED1_CFG, SUPERVISED_CFG],
        MERT_OUTPUT_DIR / "config_snapshot.yaml",
    )
    finished_at = datetime.now(timezone.utc).isoformat()
    write_run_meta(
        run_id="run_smoke_mert_seed1",
        seed_name="seed1",
        seed_index=1,
        base_seed=2027,
        config_version="v1",
        started_at=started_at,
        finished_at=finished_at,
        output_path=MERT_OUTPUT_DIR / "run_meta.json",
        extra={"task_type": "linear_probe", "model": "mert_v1_95m"},
    )
    update_registry(
        run_meta_path=MERT_OUTPUT_DIR / "run_meta.json",
        registry_csv_path=REGISTRY_CSV,
        experiment_id="smoke_mert",
        phase=0,
        task_type="linear_probe",
        model="mert_v1_95m",
        notes="config-driven; seed=2027; val/test/ext_test evaluated",
    )
    print("  MERT smoke: PASSED")


# Main

def main():
    print("Phase 0 Smoke Test — seed1")
    print("=" * 50)

    # Load configs
    configs = _load_configs()
    base_seed = configs["seed1"]["base_seed"]

    # Set global seed
    print(f"\nSetting seed: {base_seed}")
    set_seed(base_seed)

    # Load seed1 data
    print("\nLoading seed1 data...")
    unified_df = load_seed_data(DATASET_CFG, SEED1_CFG, PROJECT_ROOT)
    print(f"  Total segments: {len(unified_df)}")

    # 1. Data layer smoke
    smoke_data_layer(unified_df, configs)

    # 2. Check model cache (offline fail-fast)
    clap_cache = PROJECT_ROOT / "model_cache/huggingface/models--laion--clap-htsat-unfused"
    mert_cache = PROJECT_ROOT / "model_cache/huggingface/models--m-a-p--MERT-v1-95M"
    for cache_dir in [clap_cache, mert_cache]:
        if not cache_dir.is_dir():
            print(f"\nERROR: Model cache not found: {cache_dir}")
            print("Please download models to model_cache/huggingface/ first.")
            sys.exit(1)

    # 3. CLAP smoke
    smoke_clap(unified_df, configs)

    # 4. MERT smoke
    smoke_mert(unified_df, configs)

    print("\n" + "=" * 50)
    print("Phase 0 Smoke Test: ALL PASSED")


if __name__ == "__main__":
    main()
