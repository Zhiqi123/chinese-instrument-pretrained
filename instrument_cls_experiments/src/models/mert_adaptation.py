"""MERT adaptation model (LoRA / full fine-tuning).

Architecture: MERT encoder → mean pooling → Linear(hidden_size, num_classes).
LoRA: freeze encoder, inject LoRA on q_proj/v_proj, train LoRA + classifier.
Full FT: train entire encoder + classifier.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModel


class MERTForAdaptation(nn.Module):
    """MERT model for LoRA and full fine-tuning.

    Args:
        num_classes: Number of classes.
        model_name: HuggingFace model name.
        cache_dir: Local model cache directory.
        adaptation_mode: "lora" or "full_ft".
        lora_config: LoRA config dict (rank, lora_alpha, lora_dropout, target_modules).
    """

    def __init__(
        self,
        num_classes: int = 6,
        model_name: str = "m-a-p/MERT-v1-95M",
        cache_dir: str | Path | None = None,
        adaptation_mode: str = "lora",
        lora_config: dict | None = None,
    ):
        super().__init__()
        self.adaptation_mode = adaptation_mode

        cache_dir_str = str(cache_dir) if cache_dir is not None else None

        # Load base encoder
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
        self.expected_sr = self.feature_extractor.sampling_rate

        # Classification head
        self.classifier = nn.Linear(hidden_size, num_classes)

        if adaptation_mode == "lora":
            self._setup_lora(lora_config)
        elif adaptation_mode == "full_ft":
            self._setup_full_ft()
        else:
            raise ValueError(f"Unknown adaptation_mode: {adaptation_mode}")

    def _setup_lora(self, lora_config: dict | None) -> None:
        """LoRA mode: freeze encoder → inject LoRA → verify target module count."""
        if lora_config is None:
            raise ValueError("lora_config is required for adaptation_mode='lora'")

        from peft import LoraConfig, get_peft_model

        # Freeze all encoder params
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Inject LoRA
        peft_config = LoraConfig(
            r=lora_config["rank"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            target_modules=lora_config["target_modules"],
            bias="none",
            task_type=None,
        )
        self.encoder = get_peft_model(self.encoder, peft_config)

        # Verify 24 LoRA modules (12 layers × q/v)
        lora_module_count = 0
        for name, module in self.encoder.named_modules():
            if hasattr(module, "lora_A"):
                lora_module_count += 1

        assert lora_module_count == 24, (
            f"LoRA module count {lora_module_count} != 24 (12 layers × 2)"
        )

        # Keep classifier trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

    def _setup_full_ft(self) -> None:
        """Full FT mode: all parameters trainable."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

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

        hidden_states = outputs.last_hidden_state

        # Mean pooling over time
        if attention_mask is not None:
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
        """Preprocess waveforms with feature_extractor."""
        waveform_list = [w.numpy() for w in waveforms]
        inputs = self.feature_extractor(
            waveform_list,
            sampling_rate=self.expected_sr,
            return_tensors="pt",
            padding=True,
        )
        return inputs

    def save_checkpoint(
        self,
        output_dir: Path,
        epoch: int,
        val_metric: float,
    ) -> None:
        """Save best checkpoint.

        LoRA: adapter weights + classifier_head.pt.
        Full FT: full state_dict.
        """
        artifacts_dir = output_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        if self.adaptation_mode == "lora":
            adapter_dir = artifacts_dir / "best_adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            self.encoder.save_pretrained(str(adapter_dir))

            torch.save(
                self.classifier.state_dict(),
                artifacts_dir / "classifier_head.pt",
            )

            meta = {
                "best_epoch": epoch,
                "best_val_macro_f1": val_metric,
                "adaptation_mode": self.adaptation_mode,
            }
            with open(artifacts_dir / "checkpoint_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

        elif self.adaptation_mode == "full_ft":
            # Save full state_dict
            checkpoint = {
                "model_state_dict": self.state_dict(),
                "best_epoch": epoch,
                "best_val_macro_f1": val_metric,
            }
            torch.save(checkpoint, output_dir / "best_checkpoint.pt")

    def load_best_checkpoint(self, output_dir: Path) -> None:
        """Load best checkpoint."""
        device = next(self.parameters()).device

        if self.adaptation_mode == "lora":
            artifacts_dir = output_dir / "artifacts"

            # Reload LoRA adapter
            from peft import PeftModel
            base_encoder = self.encoder.get_base_model()
            self.encoder = PeftModel.from_pretrained(
                base_encoder,
                str(artifacts_dir / "best_adapter"),
            )
            self.encoder = self.encoder.to(device)

            # Load classifier head
            cls_state = torch.load(
                artifacts_dir / "classifier_head.pt",
                map_location=device,
                weights_only=True,
            )
            self.classifier.load_state_dict(cls_state)

        elif self.adaptation_mode == "full_ft":
            ckpt = torch.load(
                output_dir / "best_checkpoint.pt",
                map_location=device,
                weights_only=False,
            )
            self.load_state_dict(ckpt["model_state_dict"])

    @classmethod
    def load_for_eval(
        cls,
        num_classes: int,
        model_name: str,
        cache_dir: str | Path,
        adaptation_mode: str,
        checkpoint_dir: Path,
        lora_config: dict | None = None,
    ) -> "MERTForAdaptation":
        """Load a complete model from saved checkpoint for evaluation."""
        if adaptation_mode == "lora":
            cache_dir_str = str(cache_dir) if cache_dir is not None else None

            from peft import PeftModel

            base_encoder = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_dir_str,
                local_files_only=True,
            )
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_dir_str,
                local_files_only=True,
            )

            adapter_dir = checkpoint_dir / "artifacts" / "best_adapter"
            encoder = PeftModel.from_pretrained(base_encoder, str(adapter_dir))

            hidden_size = base_encoder.config.hidden_size
            classifier = nn.Linear(hidden_size, num_classes)
            cls_state = torch.load(
                checkpoint_dir / "artifacts" / "classifier_head.pt",
                weights_only=True,
            )
            classifier.load_state_dict(cls_state)

            model = cls.__new__(cls)
            nn.Module.__init__(model)
            model.adaptation_mode = adaptation_mode
            model.encoder = encoder
            model.feature_extractor = feature_extractor
            model.classifier = classifier
            model.expected_sr = feature_extractor.sampling_rate
            return model

        elif adaptation_mode == "full_ft":
            model = cls(
                num_classes=num_classes,
                model_name=model_name,
                cache_dir=cache_dir,
                adaptation_mode="full_ft",
            )
            ckpt = torch.load(
                checkpoint_dir / "best_checkpoint.pt",
                weights_only=False,
            )
            model.load_state_dict(ckpt["model_state_dict"])
            return model

        else:
            raise ValueError(f"Unknown adaptation_mode: {adaptation_mode}")
