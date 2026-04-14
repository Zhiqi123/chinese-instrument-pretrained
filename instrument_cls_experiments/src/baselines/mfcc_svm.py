"""MFCC feature extraction + SVM classifier.

Pipeline: waveform → MFCC(n_mfcc=40) → mean+std pooling (80-dim)
→ StandardScaler → SVC(rbf). Scores from decision_function + softmax.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def extract_mfcc_features(
    waveform: np.ndarray,
    sr: int = 24000,
    n_mfcc: int = 40,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    """Extract fixed-length MFCC features from a single waveform.

    Returns:
        features: [n_mfcc * 2] (mean + std concatenation).
    """
    mfcc = librosa.feature.mfcc(
        y=waveform, sr=sr,
        n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length,
    )  # [n_mfcc, time_frames]
    mean = mfcc.mean(axis=1)  # [n_mfcc]
    std = mfcc.std(axis=1)    # [n_mfcc]
    return np.concatenate([mean, std])  # [n_mfcc * 2]


def extract_features_from_dataloader(
    dataloader,
    sr: int = 24000,
    n_mfcc: int = 40,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Batch-extract MFCC features from a dataloader.

    Returns:
        (features [N, n_mfcc*2], label_ids [N], metadata_list).
    """
    all_features = []
    all_labels = []
    all_metadata = []

    for batch in dataloader:
        waveforms = batch["waveform"].numpy()  # [B, num_samples]
        labels = batch["label_id"].numpy()     # [B]
        metadata = batch["metadata"]           # list[dict]

        for i in range(len(waveforms)):
            feat = extract_mfcc_features(
                waveforms[i], sr=sr,
                n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length,
            )
            all_features.append(feat)
            all_labels.append(labels[i])
            all_metadata.append(metadata[i])

    return (
        np.array(all_features, dtype=np.float32),
        np.array(all_labels, dtype=np.int64),
        all_metadata,
    )


class MFCCSVMPipeline:
    """MFCC + SVM classification pipeline."""

    def __init__(
        self,
        n_mfcc: int = 40,
        n_fft: int = 1024,
        hop_length: int = 256,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str = "scale",
        base_seed: int = 3407,
    ):
        self.scaler = StandardScaler()
        self.svm = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            random_state=base_seed,
            decision_function_shape="ovr",
        )
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fit scaler and SVM on training data."""
        scaled = self.scaler.fit_transform(features)
        self.svm.fit(scaled, labels)

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict. Returns (pred_ids, top1_scores)."""
        scaled = self.scaler.transform(features)
        pred_ids = self.svm.predict(scaled)

        decision = self.svm.decision_function(scaled)  # [N, num_classes]
        probs = softmax(decision, axis=1)              # [N, num_classes]
        top1_scores = probs.max(axis=1)

        return pred_ids.astype(int), top1_scores.astype(float)

    def save(self, output_dir: str | Path) -> None:
        """Save scaler and SVM to artifacts/."""
        import joblib
        output_dir = Path(output_dir)
        artifacts_dir = output_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, artifacts_dir / "scaler.joblib")
        joblib.dump(self.svm, artifacts_dir / "model.joblib")
