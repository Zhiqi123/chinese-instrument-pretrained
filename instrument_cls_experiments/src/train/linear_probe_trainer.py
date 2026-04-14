"""Linear probe trainer for frozen embeddings.

Shared by CLAP and MERT linear probes. Input: pre-extracted embeddings + labels.
Training: AdamW + CosineAnnealingLR, early stopping on val macro-F1.
Checkpoint: linear head weights only (no encoder).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def build_embedding_dataloader(
    embedding_dict: dict,
    batch_size: int,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
    worker_init_fn=None,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader from an embedding dict."""
    dataset = TensorDataset(
        embedding_dict["embeddings"],
        embedding_dict["label_ids"],
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        worker_init_fn=worker_init_fn,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
    )


class LinearProbeTrainer:
    """Train and evaluate a linear classification head.

    Args:
        embedding_dim: Input embedding dimension.
        num_classes: Number of classes.
        train_cfg: Training config (from default_supervised.yaml).
        device: Torch device.
        base_seed: Random seed.
        output_dir: Output directory.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        train_cfg: dict,
        device: torch.device,
        base_seed: int,
        output_dir: Path,
    ):
        self.device = device
        self.output_dir = output_dir
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Parse training config
        training = train_cfg["training"]
        self.max_epochs = training["max_epochs"]
        self.patience = training["early_stopping"]["patience"]

        lr = training["optimizer"]["lr"]
        weight_decay = training["optimizer"]["weight_decay"]
        T_max = training["scheduler"]["T_max"]

        # Seeded head initialization
        torch.manual_seed(base_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(base_seed)
        self.head = nn.Linear(embedding_dim, num_classes).to(device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.head.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=T_max,
        )

        # Early stopping state
        self.best_val_metric = -float("inf")
        self.best_epoch = -1
        self.early_stop_counter = 0
        self.train_log: list[dict] = []

        # Timing state
        self.epoch_times: list[float] = []
        self.step_times_ms: list[float] = []
        self.stopped_epoch: int = -1

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """Full training loop with early stopping. Loads best checkpoint on exit."""
        for epoch in range(self.max_epochs):
            t_epoch_start = time.perf_counter()

            # Train
            t_train_start = time.perf_counter()
            self.head.train()
            train_losses = []
            for emb, lid in train_loader:
                emb = emb.to(self.device)
                lid = lid.to(self.device)

                t_step = time.perf_counter()
                logits = self.head(emb)
                loss = self.criterion(logits, lid)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.step_times_ms.append((time.perf_counter() - t_step) * 1000)

                train_losses.append(loss.item())

            self.scheduler.step()
            epoch_train_sec = time.perf_counter() - t_train_start

            # Validate
            t_eval_start = time.perf_counter()
            val_metrics = self._evaluate_split(val_loader)
            epoch_eval_sec = time.perf_counter() - t_eval_start

            self.epoch_times.append(time.perf_counter() - t_epoch_start)

            # Logging
            current_lr = self.scheduler.get_last_lr()[0]
            val_metric = val_metrics["macro_f1"]

            is_best = False
            if val_metric > self.best_val_metric:
                self.best_val_metric = val_metric
                self.best_epoch = epoch
                self.early_stop_counter = 0
                is_best = True
                self._save_checkpoint(epoch)
            else:
                self.early_stop_counter += 1

            self.train_log.append({
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_metrics["val_loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "lr": float(current_lr),
                "epoch_train_sec": round(epoch_train_sec, 3),
                "epoch_eval_sec": round(epoch_eval_sec, 3),
                "is_best": is_best,
                "early_stop_counter": self.early_stop_counter,
            })

            if epoch % 5 == 0 or is_best or self.early_stop_counter >= self.patience:
                print(f"    epoch {epoch}: train_loss={float(np.mean(train_losses)):.4f}, "
                      f"val_f1={val_metric:.4f}, lr={current_lr:.6f}"
                      f"{' ★' if is_best else ''}")

            # Early stop
            if self.early_stop_counter >= self.patience:
                print(f"    Early stopping at epoch {epoch} "
                      f"(best epoch={self.best_epoch}, best val_f1={self.best_val_metric:.4f})")
                self.stopped_epoch = epoch
                break
        else:
            # Normal finish (no early stop)
            self.stopped_epoch = self.max_epochs - 1

        pd.DataFrame(self.train_log).to_csv(
            self.output_dir / "train_log.csv", index=False,
        )

        self._load_best_checkpoint()

    def predict(self, loader: DataLoader) -> tuple[list[int], list[float]]:
        """Predict with best checkpoint. Returns (pred_ids, top1_scores)."""
        self.head.eval()
        all_pred_ids: list[int] = []
        all_top1_scores: list[float] = []

        with torch.no_grad():
            for emb, lid in loader:
                emb = emb.to(self.device)
                logits = self.head(emb)
                probs = torch.softmax(logits, dim=-1)

                pred_ids = logits.argmax(dim=-1).cpu().tolist()
                top1_scores = probs.max(dim=-1).values.cpu().tolist()

                all_pred_ids.extend(pred_ids)
                all_top1_scores.extend(top1_scores)

        return all_pred_ids, all_top1_scores

    def _evaluate_split(self, loader: DataLoader) -> dict:
        """Evaluate on a split. Returns dict with accuracy, macro_f1, val_loss."""
        self.head.eval()
        all_preds: list[int] = []
        all_true: list[int] = []
        all_losses: list[float] = []

        with torch.no_grad():
            for emb, lid in loader:
                emb = emb.to(self.device)
                lid = lid.to(self.device)

                logits = self.head(emb)
                loss = self.criterion(logits, lid)
                all_losses.append(loss.item())

                preds = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_true.extend(lid.cpu().tolist())

        from src.eval.metrics import compute_metrics
        metrics = compute_metrics(all_true, all_preds, "val")
        metrics["val_loss"] = float(np.mean(all_losses))
        return metrics

    def _save_checkpoint(self, epoch: int) -> None:
        """Save best checkpoint (linear head only)."""
        checkpoint = {
            "head_state_dict": self.head.state_dict(),
            "embedding_dim": self.embedding_dim,
            "num_classes": self.num_classes,
            "best_epoch": epoch,
            "best_val_macro_f1": self.best_val_metric,
        }
        torch.save(checkpoint, self.output_dir / "best_checkpoint.pt")

    def _load_best_checkpoint(self) -> None:
        """Load best checkpoint."""
        ckpt_path = self.output_dir / "best_checkpoint.pt"
        if not ckpt_path.exists():
            print(f"    WARNING: best_checkpoint.pt not found, using last state")
            return
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.head.load_state_dict(checkpoint["head_state_dict"])
