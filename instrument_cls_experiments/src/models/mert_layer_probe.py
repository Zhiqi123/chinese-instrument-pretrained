"""MERT per-layer probing model.

Architecture: MERT encoder (frozen, output_hidden_states=True)
→ select layer → mean pooling → Linear(hidden_size, num_classes).
MERT-v1-95M: 13 hidden states (0=embedding, 1-12=transformer).
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModel


class MERTLayerProbe(nn.Module):
    """MERT encoder + per-layer linear probe.

    Args:
        num_classes: Number of classes.
        model_name: HuggingFace model name.
        cache_dir: Model cache directory.
    """

    def __init__(
        self,
        num_classes: int = 6,
        model_name: str = "m-a-p/MERT-v1-95M",
        cache_dir: str | Path | None = None,
    ):
        super().__init__()

        cache_dir_str = str(cache_dir) if cache_dir is not None else None
        self.encoder = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir_str,
            local_files_only=True,
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir_str,
            local_files_only=True,
        )

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.hidden_size = self.encoder.config.hidden_size
        self.num_layers = self.encoder.config.num_hidden_layers + 1  # +1 for embedding layer
        self.expected_sr = self.feature_extractor.sampling_rate

    def preprocess(self, waveforms: torch.Tensor) -> dict[str, torch.Tensor]:
        """Preprocess waveforms with feature_extractor."""
        waveform_list = [w.numpy() for w in waveforms]
        inputs = self.feature_extractor(
            waveform_list,
            sampling_rate=self.expected_sr,
            return_tensors="pt",
            padding=True,
        )
        return inputs

    @torch.no_grad()
    def extract_all_layers(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Extract mean-pooled embeddings from all layers.

        Returns:
            list of Tensor[B, hidden_size], length = num_layers.
        """
        outputs = self.encoder(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # hidden_states: tuple of (num_layers+1) tensors, each [B, seq_len, hidden_size]
        # layer 0 = embedding output, layer 1..12 = transformer layers
        all_hidden = outputs.hidden_states

        pooled_layers = []
        for hs in all_hidden:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = hs.mean(dim=1)
            pooled_layers.append(pooled.cpu())

        return pooled_layers

    def extract_all_layers_from_dataloader(
        self,
        dataloader,
    ) -> dict:
        """Extract all-layer embeddings from a dataloader.

        Returns:
            dict: layer_embeddings (list of Tensor[N, H]), label_ids, segment_ids,
            sample_ids, record_ids.
        """
        all_layer_embs: list[list[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        all_label_ids = []
        all_segment_ids = []
        all_sample_ids = []
        all_record_ids = []

        device = next(self.parameters()).device
        self.eval()

        for batch in dataloader:
            wf = batch["waveform"]
            lid = batch["label_id"]
            meta = batch["metadata"]

            inputs = self.preprocess(wf)
            input_values = inputs["input_values"].to(device)

            layer_embs = self.extract_all_layers(input_values, attention_mask=None)

            for layer_idx, emb in enumerate(layer_embs):
                all_layer_embs[layer_idx].append(emb)

            all_label_ids.append(lid)
            for m in meta:
                all_segment_ids.append(m["segment_id"])
                all_sample_ids.append(m["sample_id"])
                all_record_ids.append(m["record_id"])

        return {
            "layer_embeddings": [torch.cat(le, dim=0) for le in all_layer_embs],
            "label_ids": torch.cat(all_label_ids, dim=0),
            "segment_ids": all_segment_ids,
            "sample_ids": all_sample_ids,
            "record_ids": all_record_ids,
        }
