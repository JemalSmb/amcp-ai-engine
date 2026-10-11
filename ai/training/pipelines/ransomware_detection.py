"""
Ransomware Behavior Detection Training Pipeline

BiLSTM-based sequence classifier that detects ransomware file access patterns.
Domain: Cybersecurity (Active)

Base model type: BEHAVIOR_ANALYZER (BiLSTM)
Input: Sequences of file system events (open, read, write, encrypt, delete)
Output: Binary classification (benign / ransomware)

Dataset options:
  - Synthetic file-system event sequences (development)
  - NeurIPS Ransomware Execution Traces
  - CIC-IDS-style file access logs
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import yaml
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split

# Shared utilities
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, generate_synthetic_sequences, get_device, export_pytorch_onnx, setup_mlflow, count_parameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Dataset


class SequenceDataset(Dataset):
    """Dataset for sequential event data."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]



# Model — Uses BehaviorAnalyzerBase architecture (BiLSTM)


class RansomwareDetector(nn.Module):
    """
    Lightweight BiLSTM for file access sequence classification.

    Architecture:
      - BiLSTM encoder (2 layers, hidden=128)
      - Attention pooling over timesteps
      - MLP head → 2 classes
    ~85K parameters.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        arch = config["architecture"]

        input_dim = arch["input_dim"]
        hidden_dim = arch["hidden_dim"]
        num_layers = arch["num_layers"]
        num_classes = arch["num_classes"]
        dropout = arch.get("dropout", 0.2)

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * 2  # bidirectional

        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

        # Attention weights over timesteps
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden*2)

        return self.classifier(context)



# Data Loading


def load_data(data_dir: str, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Load sequential file-system event data."""
    data_path = Path(data_dir)
    npy_files = list(data_path.glob("*.npy"))

    if npy_files:
        logger.info(f"Loading sequence data from {data_path}")
        sequences = np.load(data_path / "sequences.npy")
        labels = np.load(data_path / "labels.npy")
        logger.info(f"Loaded {len(sequences)} sequences, shape {sequences.shape}")
        return sequences.astype(np.float32), labels.astype(np.int32)

    # Synthetic fallback
    logger.warning("No data files found — generating synthetic sequences")
    arch = config["architecture"]
    return generate_synthetic_sequences(
        n_sequences=config["training"]["synthetic_samples"],
        seq_length=arch["seq_length"],
        n_features=arch["input_dim"],
        n_classes=arch["num_classes"],
        seed=42,
    )



# Training


def train_epoch(model, dataloader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0
    for sequences, labels in dataloader:
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device) -> Dict[str, float]:
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            logits = model(sequences)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"\n{classification_report(y_true, y_pred, target_names=['benign', 'ransomware'])}")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": float(auc),
    }



# Main


def main():
    parser = argparse.ArgumentParser(description="Train ransomware behaviour detector")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--mlflow-uri", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default="ransomware-detection")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_mlflow(args.mlflow_uri, args.experiment_name)
    device = get_device()
    logger.info(f"Device: {device}")

    # Data
    sequences, labels = load_data(args.data, config)
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_loader = DataLoader(
        SequenceDataset(X_train, y_train),
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        SequenceDataset(X_val, y_val),
        batch_size=config["training"]["batch_size"],
    )

    # Model
    model = RansomwareDetector(config).to(device)
    logger.info(f"Model parameters: {count_parameters(model):,}")

    # Training setup
    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor([1.0, 3.0]).to(device)  # Bias toward catching ransomware
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Training loop
    with mlflow.start_run():
        mlflow.log_params({
            "model_name": config["model"]["name"],
            "model_version": config["model"]["version"],
            "base_type": "BEHAVIOR_ANALYZER",
            "domain": "cybersecurity",
            "n_params": count_parameters(model),
            "batch_size": config["training"]["batch_size"],
            "learning_rate": config["training"]["learning_rate"],
        })

        best_f1 = 0
        patience_counter = 0

        for epoch in range(config["training"]["epochs"]):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            metrics = evaluate(model, val_loader, device)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_accuracy": metrics["accuracy"],
                "val_precision": metrics["precision"],
                "val_recall": metrics["recall"],
                "val_f1": metrics["f1"],
                "val_auc": metrics["auc_roc"],
            }, step=epoch)

            logger.info(
                f"Epoch {epoch+1}/{config['training']['epochs']} — "
                f"Loss: {train_loss:.4f}, F1: {metrics['f1']:.4f}"
            )

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                patience_counter = 0
                ckpt_path = Path(args.output) / "best_model.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)
                mlflow.log_artifact(str(ckpt_path))
            else:
                patience_counter += 1

            if patience_counter >= config["training"]["early_stopping_patience"]:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Export ONNX
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model.cpu().eval()
        arch = config["architecture"]
        dummy = torch.randn(1, arch["seq_length"], arch["input_dim"])
        onnx_path = Path(args.output) / f"{config['model']['name']}_v{config['model']['version']}.onnx"
        export_pytorch_onnx(model, dummy, str(onnx_path))
        mlflow.log_artifact(str(onnx_path))

        logger.info(f"Training complete. Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
