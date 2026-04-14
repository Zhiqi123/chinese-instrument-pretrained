"""CLAP zero-shot inference.

Pipeline: load ClapModel → build text prompts from templates → compute
audio-text similarity → average across templates → argmax.
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
import yaml
from transformers import AutoProcessor, ClapModel


def load_clap_prompt_config(prompt_config_path: str | Path) -> dict:
    """Load CLAP prompt config YAML."""
    with open(prompt_config_path) as f:
        return yaml.safe_load(f)


class ClapZeroShotClassifier:
    """CLAP zero-shot classifier."""

    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        prompt_config_path: str | Path | None = None,
        label_map: dict[str, Any] | None = None,
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

        # Target sample rate from processor
        self.target_sr = self.processor.feature_extractor.sampling_rate

        if prompt_config_path is not None:
            self.prompt_cfg = load_clap_prompt_config(prompt_config_path)
        else:
            self.prompt_cfg = None

        self.label_map = label_map

    def _build_text_prompts(self) -> list[list[str]]:
        """Build [num_templates, num_classes] prompt matrix."""
        templates = self.prompt_cfg["templates"]
        instrument_names = self.prompt_cfg["instrument_names"]
        classes = self.label_map["classes"]  # sorted by label_id

        prompts_by_template = []
        for tmpl in templates:
            prompts = []
            for family_label in classes:
                eng_name = instrument_names[family_label]
                prompts.append(tmpl.format(instrument=eng_name))
            prompts_by_template.append(prompts)

        return prompts_by_template

    @torch.no_grad()
    def predict_batch(
        self,
        waveforms: np.ndarray | torch.Tensor,
        source_sr: int = 24000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Zero-shot classification for a batch of audio.

        Args:
            waveforms: [batch_size, num_samples] float32.
            source_sr: Input sample rate.

        Returns:
            pred_label_ids: [batch_size], top1_scores: [batch_size].
        """
        if isinstance(waveforms, torch.Tensor):
            waveforms_tensor = waveforms
        else:
            waveforms_tensor = torch.from_numpy(waveforms)

        # Resample to target sample rate
        if source_sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(source_sr, self.target_sr)
            waveforms_tensor = resampler(waveforms_tensor)

        waveforms_np = waveforms_tensor.numpy()

        prompts_by_template = self._build_text_prompts()
        num_templates = len(prompts_by_template)
        batch_size = len(waveforms_np)

        # Accumulate logits across templates
        logits_accum = None

        for template_prompts in prompts_by_template:
            audio_list = [waveforms_np[i] for i in range(batch_size)]

            inputs = self.processor(
                text=template_prompts,
                audio=audio_list,
                return_tensors="pt",
                padding=True,
                sampling_rate=self.target_sr,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            # logits_per_audio: [batch_size, num_classes]
            logits = outputs.logits_per_audio

            if logits_accum is None:
                logits_accum = logits
            else:
                logits_accum = logits_accum + logits

        # Average across templates
        avg_logits = logits_accum / num_templates
        probs = avg_logits.softmax(dim=-1)

        pred_label_ids = avg_logits.argmax(dim=-1).cpu().numpy()
        top1_scores = probs.max(dim=-1).values.cpu().numpy()

        return pred_label_ids, top1_scores
