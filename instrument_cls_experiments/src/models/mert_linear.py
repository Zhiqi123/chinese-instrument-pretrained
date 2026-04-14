"""MERT encoder + linear classification head.

Architecture: MERT encoder (frozen) → mean pooling → Linear(hidden_size, num_classes).
Model: m-a-p/MERT-v1-95M (HuBERT-based, 24kHz, hidden_size=768, 12 layers).
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


class MERTLinearProbe(nn.Module):
    """MERT encoder + linear probe.

    Args:
        model_name: HuggingFace model name.
        num_classes: Number of classes.
        freeze_encoder: Whether to freeze MERT parameters.
    """

    def __init__(
        self,
        num_classes: int = 6,
        model_name: str = "m-a-p/MERT-v1-95M",
        freeze_encoder: bool = True,
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

        hidden_size = self.encoder.config.hidden_size

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(hidden_size, num_classes)

        self.expected_sr = self.feature_extractor.sampling_rate

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_values: [B, num_samples] float32 (after feature_extractor).
            attention_mask: Optional [B, num_samples].

        Returns:
            logits: [B, num_classes].
        """
        outputs = self.encoder(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )

        # Last hidden states: [B, seq_len, H]
        hidden_states = outputs.last_hidden_state

        # Mean pooling over time: [B, H]
        if attention_mask is not None:
            # Masked mean pooling
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)

        logits = self.classifier(pooled)
        return logits

    def preprocess(
        self,
        waveforms: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Preprocess waveforms with feature_extractor.

        Args:
            waveforms: [B, num_samples] float32 tensor.

        Returns:
            dict with 'input_values' key.
        """
        waveform_list = [w.numpy() for w in waveforms]
        inputs = self.feature_extractor(
            waveform_list,
            sampling_rate=self.expected_sr,
            return_tensors="pt",
            padding=True,
        )
        return inputs

    @torch.no_grad()
    def extract_embeddings(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract mean-pooled embeddings (no classifier head).

        Returns:
            pooled: [B, hidden_size] float32 (CPU).
        """
        outputs = self.encoder(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        hidden_states = outputs.last_hidden_state

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)

        return pooled.cpu()

    def extract_from_dataloader(
        self,
        dataloader,
    ) -> dict:
        """Extract all embeddings from a dataloader.

        Returns:
            dict: embeddings [N, H], label_ids [N], segment_ids, sample_ids, record_ids.
        """
        all_embeddings = []
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
            # Fixed 5s segments, no attention mask needed
            emb = self.extract_embeddings(input_values, attention_mask=None)
            all_embeddings.append(emb)
            all_label_ids.append(lid)

            for m in meta:
                all_segment_ids.append(m["segment_id"])
                all_sample_ids.append(m["sample_id"])
                all_record_ids.append(m["record_id"])

        return {
            "embeddings": torch.cat(all_embeddings, dim=0),
            "label_ids": torch.cat(all_label_ids, dim=0),
            "segment_ids": all_segment_ids,
            "sample_ids": all_sample_ids,
            "record_ids": all_record_ids,
        }
