"""Training pipeline for VOXGUARD Deepfake Voice Detection CNN.

Supports folder-based datasets (data/real, data/fake), file-level splitting
to prevent data leakage, audio augmentation, class weighting, and early stopping.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src.audio_utils import preprocess_audio
from src.features import apply_augmentation, extract_log_mel_spectrogram
from src.model import VoxGuardCNN


def seed_everything(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class VoxGuardAudioDataset(Dataset):
    """PyTorch Dataset that loads audio windows and converts to Mel-Spectrogram tensors."""

    def __init__(
        self,
        samples: List[Tuple[np.ndarray, int]],  # (window_array, label)
        augment: bool = False,
        sr: int = 16000,
    ):
        self.samples = samples
        self.augment = augment
        self.sr = sr

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        waveform, label = self.samples[idx]
        if self.augment:
            waveform = apply_augmentation(waveform, sr=self.sr)

        # Extract normalized Log-Mel Spectrogram -> (128, 64)
        spec = extract_log_mel_spectrogram(waveform, sr=self.sr)
        tensor = torch.from_numpy(spec).unsqueeze(0)  # Shape: (1, 128, 64)
        target = torch.tensor(label, dtype=torch.long)
        return tensor, target


def discover_and_split_files(
    data_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Discover real and fake audio files and partition by file to prevent window leakage."""
    real_paths = glob.glob(os.path.join(data_dir, "real", "*.wav")) + glob.glob(
        os.path.join(data_dir, "real", "*.mp3")
    )
    fake_paths = glob.glob(os.path.join(data_dir, "fake", "*.wav")) + glob.glob(
        os.path.join(data_dir, "fake", "*.mp3")
    )

    if not real_paths or not fake_paths:
        raise ValueError(
            f"Insufficient data in '{data_dir}'. Found {len(real_paths)} real and {len(fake_paths)} fake files. "
            "Run generate_demo_data.py or point to an external dataset directory."
        )

    rng = random.Random(seed)
    rng.shuffle(real_paths)
    rng.shuffle(fake_paths)

    def split_list(items: List[str], label: int) -> Tuple[List, List, List]:
        n = len(items)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        train = [(p, label) for p in items[:n_train]]
        val = [(p, label) for p in items[n_train : n_train + n_val]]
        test = [(p, label) for p in items[n_train + n_val :]]
        if not test:
            # If dataset is tiny, share val as test
            test = val.copy()
        return train, val, test

    real_train, real_val, real_test = split_list(real_paths, label=0)
    fake_train, fake_val, fake_test = split_list(fake_paths, label=1)

    train_files = real_train + fake_train
    val_files = real_val + fake_val
    test_files = real_test + fake_test

    rng.shuffle(train_files)
    rng.shuffle(val_files)
    rng.shuffle(test_files)

    return train_files, val_files, test_files


def extract_windows_from_file_list(
    file_list: List[Tuple[str, int]],
    sr: int = 16000,
) -> List[Tuple[np.ndarray, int]]:
    """Extract overlapping 2-second windows from list of audio files."""
    dataset_windows: List[Tuple[np.ndarray, int]] = []
    for file_path, label in file_list:
        try:
            _, windows, _ = preprocess_audio(file_path, sr=sr, window_sec=2.0, hop_sec=1.0)
            for win in windows:
                dataset_windows.append((win, label))
        except Exception as e:
            print(f"Warning: could not process {file_path}: {e}")
    return dataset_windows


def plot_training_curves(history: Dict[str, List[float]], save_path: str) -> None:
    """Plot and save training/validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#2563eb", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#dc2626", linewidth=2, linestyle="--")
    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Accuracy curve
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc", color="#16a34a", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Acc", color="#9333ea", linewidth=2, linestyle="--")
    ax2.set_title("Accuracy (%) vs. Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def train_model(
    data_dir: str = "data",
    models_dir: str = "models",
    reports_dir: str = "reports",
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 0.001,
    seed: int = 42,
) -> Tuple[VoxGuardCNN, Dict[str, Any]]:
    """Complete training and validation loop for VOXGUARD."""
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"VOXGUARD Training using compute device: {device}")

    train_files, val_files, test_files = discover_and_split_files(data_dir, seed=seed)
    print(
        f"Partitioned files: Train={len(train_files)} (Real={sum(1 for _, l in train_files if l==0)}, Fake={sum(1 for _, l in train_files if l==1)}) | "
        f"Val={len(val_files)} | Test={len(test_files)}"
    )

    train_windows = extract_windows_from_file_list(train_files)
    val_windows = extract_windows_from_file_list(val_files)
    test_windows = extract_windows_from_file_list(test_files)
    print(f"Extracted windows: Train={len(train_windows)}, Val={len(val_windows)}, Test={len(test_windows)}")

    # Compute class weights to counter potential class imbalance
    labels = [label for _, label in train_windows]
    count_0 = labels.count(0)
    count_1 = labels.count(1)
    total_samples = len(labels)
    weight_0 = total_samples / (2.0 * max(1, count_0))
    weight_1 = total_samples / (2.0 * max(1, count_1))
    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32).to(device)

    train_dataset = VoxGuardAudioDataset(train_windows, augment=True)
    val_dataset = VoxGuardAudioDataset(val_windows, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = VoxGuardCNN(in_channels=1, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    best_val_loss = float("inf")
    best_weights_path = os.path.join(models_dir, "voxguard_best.pt")
    os.makedirs(models_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(batch_y)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)

        scheduler.step()

        epoch_train_loss = running_loss / max(1, total)
        epoch_train_acc = correct / max(1, total)

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                val_running_loss += loss.item() * len(batch_y)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += len(batch_y)

                all_val_preds.extend(preds.cpu().numpy().tolist())
                all_val_targets.extend(batch_y.cpu().numpy().tolist())

        epoch_val_loss = val_running_loss / max(1, val_total)
        epoch_val_acc = val_correct / max(1, val_total)
        epoch_val_f1 = float(f1_score(all_val_targets, all_val_preds, zero_division=0))

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["val_f1"].append(epoch_val_f1)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.1f}% || "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.1f}% | Val F1: {epoch_val_f1:.3f}"
        )

        # Save checkpoint if best validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    "val_acc": epoch_val_acc,
                    "val_f1": epoch_val_f1,
                },
                best_weights_path,
            )

    print(f"Saved best model checkpoint to {best_weights_path} (Val Loss: {best_val_loss:.4f})")

    # Save training curves plot
    curves_path = os.path.join(reports_dir, "loss_acc_curves.png")
    plot_training_curves(history, curves_path)
    print(f"Saved training curves to {curves_path}")

    # Save test partition for evaluation module
    np.savez_compressed(
        os.path.join(models_dir, "test_split.npz"),
        windows=[w for w, _ in test_windows],
        labels=[l for _, l in test_windows],
    )

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VOXGUARD Deepfake Audio Detection Model")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data folder containing real/ and fake/")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
