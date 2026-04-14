"""
MERT all-layer embedding cache builder.
Extracts and caches per-layer embeddings as NPZ files for all splits.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "instrument_cls_experiments"))

from src.data.dataloaders import build_dataloader
from src.data.label_map import load_label_map
from src.data.manifest_loader import load_seed_data
from src.utils.seed import get_device, set_global_seed

EXP_ROOT = PROJECT_ROOT / "instrument_cls_experiments"
DATASET_CFG = EXP_ROOT / "configs/data/dataset_v1.yaml"
SUPERVISED_CFG = EXP_ROOT / "configs/train/default_supervised.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="MERT all-layer embedding cache builder")
    parser.add_argument("--config", required=True, help="Layer probe method config YAML")
    parser.add_argument("--seed-name", required=True, choices=["seed0", "seed1", "seed2"])
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    method_cfg = _load_yaml(Path(args.config))
    seed_cfg_path = EXP_ROOT / f"configs/data/{args.seed_name}.yaml"
    seed_cfg = _load_yaml(seed_cfg_path)
    sup_cfg = _load_yaml(SUPERVISED_CFG)

    seed_name = seed_cfg["seed_name"]
    base_seed = seed_cfg["base_seed"]

    print(f"=== Layer Cache Build — {seed_name} (seed={base_seed}) ===")

    set_global_seed(base_seed)

    cache_dir = EXP_ROOT / method_cfg["output"]["cache_dir"] / seed_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(sup_cfg.get("device", "auto"))
    print(f"  Device: {device}")

    label_map = load_label_map(DATASET_CFG)
    unified_df = load_seed_data(DATASET_CFG, seed_cfg_path, PROJECT_ROOT)
    num_classes = label_map["num_classes"]
    print(f"  Total segments: {len(unified_df)}")

    # Load MERT model
    from src.models.mert_layer_probe import MERTLayerProbe

    model_cfg = method_cfg["model_config"]
    model_cache = PROJECT_ROOT / model_cfg["model_cache_dir"]
    model = MERTLayerProbe(
        num_classes=num_classes,
        model_name=model_cfg["model_name"],
        cache_dir=model_cache,
    )
    model = model.to(device)
    num_layers = model.num_layers
    hidden_size = model.hidden_size
    print(f"  num_layers={num_layers}, hidden_size={hidden_size}")

    inf_batch_size = sup_cfg["evaluation"]["batch_size"]
    inf_num_workers = sup_cfg["evaluation"]["num_workers"]

    for split in ["train", "val", "test", "external_test"]:
        cache_path = cache_dir / f"{split}_all_layers.npz"
        if cache_path.exists():
            print(f"  {split}: cached (.npz exists), skipping")
            continue

        # All splits sorted by segment_id
        split_df = unified_df[unified_df["split"] == split].copy()
        split_df = split_df.sort_values("segment_id").reset_index(drop=True)

        loader = build_dataloader(
            split_df, split, batch_size=inf_batch_size, num_workers=inf_num_workers,
        )
        emb_data = model.extract_all_layers_from_dataloader(loader)
        n_samples = emb_data["label_ids"].shape[0]

        # Build NPZ dict
        npz_dict = {
            "segment_id": np.array(emb_data["segment_ids"]),
            "label_id": emb_data["label_ids"].numpy(),
            "split": np.array([split] * n_samples),
            "seed_name": np.array([seed_name] * n_samples),
        }
        for layer_idx in range(num_layers):
            key = f"layer_{layer_idx:02d}"
            npz_dict[key] = emb_data["layer_embeddings"][layer_idx].numpy()

        np.savez_compressed(cache_path, **npz_dict)
        print(f"  {split}: {n_samples} samples × {num_layers} layers, "
              f"dim={hidden_size} → {cache_path.name}")

    # Release MERT model
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"\n  Layer cache build {seed_name}: DONE")


if __name__ == "__main__":
    main()
