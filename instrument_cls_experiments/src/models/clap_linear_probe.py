"""CLAP audio embedding extractor for linear probing.

Pipeline: 24kHz segment → resample to 48kHz → ClapModel.get_audio_features().
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from pathlib import Path

import torch
import torchaudio
from transformers import AutoProcessor, ClapModel


class ClapEmbeddingExtractor:
    """Frozen CLAP audio embedding extractor.

    Args:
        model_name: HuggingFace model ID.
        device: Torch device string.
    """

    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        device: str = "cpu",
        cache_dir: str | Path | None = None,
    ):
        self.device = torch.device(device)
        cache_dir_str = str(cache_dir) if cache_dir is not None else None
        self.model = ClapModel.from_pretrained(
            model_name,
            cache_dir=cache_dir_str,
            local_files_only=True,
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir_str,
            local_files_only=True,
        )
        self.model.eval()

        self.target_sr = self.processor.feature_extractor.sampling_rate  # 48000
        self.embedding_dim = self.model.config.projection_dim

    @torch.no_grad()
    def extract_batch(
        self,
        waveforms: torch.Tensor,
        source_sr: int = 24000,
    ) -> torch.Tensor:
        """Extract embeddings for a batch of waveforms.

        Args:
            waveforms: [batch_size, num_samples] float32 at source_sr.
            source_sr: Input sample rate.

        Returns:
            embeddings: [batch_size, embedding_dim] float32.
        """
        # Resample to 48kHz
        if source_sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(source_sr, self.target_sr)
            waveforms = resampler(waveforms)

        waveforms_np = waveforms.cpu().numpy()
        audio_list = [waveforms_np[i] for i in range(len(waveforms_np))]

        inputs = self.processor(
            audio=audio_list,
            return_tensors="pt",
            padding=True,
            sampling_rate=self.target_sr,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get projected audio embedding
        audio_output = self.model.get_audio_features(**inputs)
        audio_features = audio_output.pooler_output
        return audio_features.cpu()

    def extract_from_dataloader(
        self,
        dataloader,
        source_sr: int = 24000,
    ) -> dict:
        """Extract all embeddings from a dataloader.

        Returns:
            dict: embeddings [N, D], label_ids [N], segment_ids, sample_ids, record_ids.
        """
        all_embeddings = []
        all_label_ids = []
        all_segment_ids = []
        all_sample_ids = []
        all_record_ids = []

        for batch in dataloader:
            wf = batch["waveform"]
            lid = batch["label_id"]
            meta = batch["metadata"]

            emb = self.extract_batch(wf, source_sr=source_sr)
            all_embeddings.append(emb)
            all_label_ids.append(lid)

            for m in meta:
                all_segment_ids.append(m["segment_id"])
                all_sample_ids.append(m["sample_id"])
                all_record_ids.append(m["record_id"])

        return {
            "embeddings": torch.cat(all_embeddings, dim=0),  # [N, D] float32
            "label_ids": torch.cat(all_label_ids, dim=0),     # [N] int64
            "segment_ids": all_segment_ids,
            "sample_ids": all_sample_ids,
            "record_ids": all_record_ids,
        }
