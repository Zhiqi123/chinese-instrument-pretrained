"""Generate UMAP 2D coordinates from three representation spaces."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = PROJECT_ROOT / "instrument_cls_experiments"
sys.path.insert(0, str(EXP_ROOT))

ARTIFACTS_DIR = EXP_ROOT / "artifacts/exp04_analysis"
UMAP_EMB_DIR = ARTIFACTS_DIR / "umap_embeddings"
TABLES_DIR = EXP_ROOT / "reports/tables"


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _get_frozen_mert_embeddings(seed_name: str, split: str) -> dict:
    """Load frozen MERT last-layer embedding from layer cache NPZ."""
    cache_path = (EXP_ROOT / "artifacts/exp04_analysis/layer_embeddings"
                  / seed_name / f"{split}_all_layers.npz")
    if not cache_path.exists():
        raise FileNotFoundError(f"Layer cache NPZ not found: {cache_path}")

    data = np.load(cache_path, allow_pickle=True)
    layer_keys = sorted([k for k in data.keys() if k.startswith("layer_")])
    last_layer_key = layer_keys[-1]

    result = {
        "embeddings": torch.from_numpy(data[last_layer_key]),
        "label_ids": torch.from_numpy(data["label_id"]),
        "segment_ids": data["segment_id"].tolist(),
    }
    data.close()
    return result


def _extract_adaptation_embeddings(
    method_short: str,
    seed_name: str,
    split: str,
    adaptation_mode: str,
    lora_config: dict | None = None,
) -> dict:
    """Extract mean-pooled last_hidden_state from a Phase 2 adaptation checkpoint."""
    from src.data.dataloaders import build_dataloader
    from src.data.label_map import load_label_map
    from src.data.manifest_loader import load_seed_data
    from src.models.mert_adaptation import MERTForAdaptation
    from src.utils.seed import get_device

    run_dir = EXP_ROOT / "runs/exp02_adaptation" / method_short / seed_name

    cfg = _load_yaml(run_dir / "config_snapshot.yaml")
    model_name = cfg["model_config"]["model_name"]
    model_cache = PROJECT_ROOT / cfg["model_config"]["model_cache_dir"]

    label_map = load_label_map(EXP_ROOT / "configs/data/dataset_v1.yaml")
    seed_cfg_path = EXP_ROOT / f"configs/data/{seed_name}.yaml"
    unified_df = load_seed_data(
        EXP_ROOT / "configs/data/dataset_v1.yaml", seed_cfg_path, PROJECT_ROOT,
    )

    split_df = unified_df[unified_df["split"] == split].copy()
    split_df = split_df.sort_values("segment_id").reset_index(drop=True)

    model = MERTForAdaptation.load_for_eval(
        num_classes=label_map["num_classes"],
        model_name=model_name,
        cache_dir=model_cache,
        adaptation_mode=adaptation_mode,
        checkpoint_dir=run_dir,
        lora_config=lora_config,
    )

    device = get_device("auto")
    model = model.to(device)
    model.eval()

    loader = build_dataloader(split_df, split, batch_size=32, num_workers=4)

    all_embeddings = []
    all_label_ids = []
    all_segment_ids = []

    with torch.no_grad():
        for batch in loader:
            wf = batch["waveform"]
            lid = batch["label_id"]
            meta = batch["metadata"]

            inputs = model.preprocess(wf)
            input_values = inputs["input_values"].to(device)

            outputs = model.encoder(
                input_values, attention_mask=None, output_hidden_states=False,
            )
            hidden = outputs.last_hidden_state
            pooled = hidden.mean(dim=1).cpu()

            all_embeddings.append(pooled)
            all_label_ids.append(lid)
            for m in meta:
                all_segment_ids.append(m["segment_id"])

    return {
        "embeddings": torch.cat(all_embeddings, dim=0),
        "label_ids": torch.cat(all_label_ids, dim=0),
        "segment_ids": all_segment_ids,
    }


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def main() -> None:
    print("build_exp04_umap_points: generating UMAP coordinates")
    print("=" * 60)

    analysis_cfg = _load_yaml(EXP_ROOT / "configs/analysis/phase4_analysis_v1.yaml")
    manifest = _load_json(ARTIFACTS_DIR / "phase4_selection_manifest.json")

    seed_name = analysis_cfg["qualitative_seed_name"]
    split = analysis_cfg["umap_split"]
    umap_params = analysis_cfg["umap_params"]

    best_lora_short = manifest["best_lora_method_short"]
    best_lora_display = manifest["best_lora_method_display"]

    UMAP_EMB_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # build repr_specs from config (single source of truth)
    repr_specs = []
    for rep_cfg in analysis_cfg["umap_representations"]:
        spec = {"key": rep_cfg["key"]}

        if rep_cfg["key"] == "best_lora":
            # best_lora method_short/display filled dynamically from manifest
            spec["method_short"] = best_lora_short
            spec["method_display"] = best_lora_display
            spec["source_run_id"] = rep_cfg["source_run_id_template"].format(
                method_short=best_lora_short, seed_name=seed_name,
            )
        else:
            spec["method_short"] = rep_cfg["method_short"]
            spec["method_display"] = rep_cfg["method_display"]
            spec["source_run_id"] = rep_cfg["source_run_id_template"].format(
                seed_name=seed_name,
            )

        repr_specs.append(spec)

    representations = {}

    # 1. Frozen MERT
    print(f"  [1/3] Frozen MERT (from layer cache NPZ)...")
    frozen_data = _get_frozen_mert_embeddings(seed_name, split)
    representations["frozen_mert"] = frozen_data
    print(f"    {frozen_data['embeddings'].shape[0]} samples, "
          f"dim={frozen_data['embeddings'].shape[1]}")

    # 2. Best LoRA
    print(f"  [2/3] {best_lora_display} ({best_lora_short})...")
    lora_yaml_path = EXP_ROOT / f"configs/experiments/exp02_adaptation/{best_lora_short}_v1.yaml"
    lora_method_cfg = _load_yaml(lora_yaml_path)
    lora_config = lora_method_cfg["model_config"]["lora"]
    lora_data = _extract_adaptation_embeddings(
        best_lora_short, seed_name, split,
        adaptation_mode="lora", lora_config=lora_config,
    )
    representations["best_lora"] = lora_data
    print(f"    {lora_data['embeddings'].shape[0]} samples")

    # 3. MERT Full FT
    print(f"  [3/3] MERT full fine-tuning...")
    fullft_data = _extract_adaptation_embeddings(
        "mert_full_ft", seed_name, split,
        adaptation_mode="full_ft",
    )
    representations["mert_full_ft"] = fullft_data
    print(f"    {fullft_data['embeddings'].shape[0]} samples")

    # verify segment_id alignment (blocking check)
    ref_ids = frozen_data["segment_ids"]
    for spec in repr_specs[1:]:
        key = spec["key"]
        other_ids = representations[key]["segment_ids"]
        if other_ids != ref_ids:
            raise RuntimeError(
                f"segment_id alignment failed: {key} vs frozen_mert. "
                f"frozen_mert[0:3]={ref_ids[:3]}, {key}[0:3]={other_ids[:3]}. "
                f"len(frozen)={len(ref_ids)}, len({key})={len(other_ids)}"
            )
    print("  segment_id alignment: PASSED")

    # L2 normalize + UMAP
    from src.utils.umap_import import get_umap_class
    UMAP = get_umap_class()

    from src.data.label_map import load_label_map
    label_map = load_label_map(EXP_ROOT / "configs/data/dataset_v1.yaml")

    all_points = []

    for spec in repr_specs:
        key = spec["key"]
        data = representations[key]
        emb_np = data["embeddings"].numpy()
        emb_norm = _l2_normalize(emb_np)

        print(f"  UMAP fitting: {key}...")
        reducer = UMAP(
            n_components=2,
            n_neighbors=umap_params["n_neighbors"],
            min_dist=umap_params["min_dist"],
            metric=umap_params["metric"],
            random_state=umap_params["random_state"],
        )
        coords = reducer.fit_transform(emb_norm)

        # save NPZ
        npz_path = UMAP_EMB_DIR / f"{key}_{split}.npz"
        np.savez_compressed(
            npz_path,
            umap_x=coords[:, 0],
            umap_y=coords[:, 1],
            label_ids=data["label_ids"].numpy(),
            segment_ids=np.array(data["segment_ids"]),
        )
        print(f"    Saved: {npz_path.relative_to(EXP_ROOT)}")

        # collect CSV rows
        label_ids_np = data["label_ids"].numpy()
        for i in range(len(coords)):
            lid = int(label_ids_np[i])
            all_points.append({
                "segment_id": data["segment_ids"][i],
                "label_id": lid,
                "label": label_map["id_to_label"][lid],
                "method_short": spec["method_short"],
                "method_display": spec["method_display"],
                "seed_name": seed_name,
                "split": split,
                "umap_x": float(coords[i, 0]),
                "umap_y": float(coords[i, 1]),
                "source_run_id": spec["source_run_id"],
            })

    # write CSV
    import pandas as pd
    points_df = pd.DataFrame(all_points)
    csv_path = TABLES_DIR / "exp04_umap_points.csv"
    points_df.to_csv(csv_path, index=False)
    print(f"\n  wrote: {csv_path.relative_to(EXP_ROOT)} ({len(points_df)} rows)")

    print("=" * 60)
    print("build_exp04_umap_points: done")


if __name__ == "__main__":
    main()
