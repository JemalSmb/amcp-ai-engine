"""
Finance — Trading Behavior Analyzer Pipeline

BiLSTM for detecting abnormal trading/transaction sequences.
Domain: Finance (Active)

Base model type: BEHAVIOR_ANALYZER (BiLSTM)
Input: Sequences of trading actions over a time window
Output: Binary classification (normal / suspicious pattern)

Proves lego principle: same BiLSTM architecture as cybersecurity's ransomware
detector, but trained on financial trading sequences.

Dataset options:
  - LOBSTER Order Book Data
  - Synthetic trading sequences (development)
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, generate_synthetic_sequences, get_device, export_pytorch_onnx, setup_mlflow, count_parameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Dataset


class TradingSequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]



# Model — Lightweight BiLSTM (~65K params)


class TradingBehaviorAnalyzer(nn.Module):
    """
    BiLSTM for trading pattern classification.

    Architecture: BiLSTM (2 layers, hidden=64) → attention → MLP → 2 classes
    ~65K parameters. Detects wash trading, spoofing, layering patterns.
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
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out = hidden_dim * 2

        self.attention = nn.Sequential(
            nn.Linear(lstm_out, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn = torch.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * attn).sum(dim=1)
        return self.classifier(context)



# Data Loading


def load_data(data_dir: str, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    data_path = Path(data_dir)
    npy_files = list(data_path.glob("*.npy"))

    if npy_files:
        sequences = np.load(data_path / "sequences.npy")
        labels = np.load(data_path / "labels.npy")
        return sequences.astype(np.float32), labels.astype(np.int32)

    logger.warning("No data found — generating synthetic trading sequences")
    arch = config["architecture"]

    np.random.seed(42)
    n = config["training"]["synthetic_samples"]
    seq_len = arch["seq_length"]
    n_features = arch["input_dim"]

    # Normal trading: smooth, low volatility
    n_normal = n // 2
    normal_seqs = []
    for _ in range(n_normal):
        price = np.cumsum(np.random.randn(seq_len) * 0.01 + 0.0001)
        volume = np.abs(np.random.randn(seq_len) * 10 + 50)
        spread = np.abs(np.random.randn(seq_len) * 0.1 + 0.5)
        base = np.column_stack([price, volume, spread])
        if n_features > 3:
            extra = np.random.randn(seq_len, n_features - 3) * 0.2
            base = np.hstack([base, extra])
        normal_seqs.append(base)

    # Suspicious trading: sudden spikes, layering patterns
    n_suspicious = n - n_normal
    sus_seqs = []
    for _ in range(n_suspicious):
        price = np.cumsum(np.random.randn(seq_len) * 0.05)
        # Inject spike
        spike_idx = np.random.randint(seq_len // 3, 2 * seq_len // 3)
        price[spike_idx:spike_idx + 5] += np.random.uniform(2, 5)
        volume = np.abs(np.random.randn(seq_len) * 30 + 100)
        volume[spike_idx:spike_idx + 5] *= 5  # Volume spike
        spread = np.abs(np.random.randn(seq_len) * 0.5 + 1.5)
        base = np.column_stack([price, volume, spread])
        if n_features > 3:
            extra = np.random.randn(seq_len, n_features - 3) * 0.8
            base = np.hstack([base, extra])
        sus_seqs.append(base)

    sequences = np.array(normal_seqs + sus_seqs, dtype=np.float32)
    labels = np.array([0] * n_normal + [1] * n_suspicious, dtype=np.int32)

    idx = np.random.permutation(len(sequences))
    return sequences[idx], labels[idx]



# Training


def train_epoch(model, dataloader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0
    for seqs, labels in dataloader:
        seqs, labels = seqs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(seqs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device) -> Dict[str, float]:
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for seqs, labels in dataloader:
            seqs = seqs.to(device)
            logits = model(seqs)
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    y_true, y_pred, y_prob = np.array(all_labels), np.array(all_preds), np.array(all_probs)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0

    logger.info(f"\n{classification_report(y_true, y_pred, target_names=['normal', 'suspicious'])}")

    return {"accuracy": float(accuracy), "precision": float(precision), "recall": float(recall), "f1": float(f1), "auc_roc": float(auc)}



# Main


def main():
    parser = argparse.ArgumentParser(description="Train finance trading behavior analyzer")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--mlflow-uri", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default="finance-trading-behavior")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_mlflow(args.mlflow_uri, args.experiment_name)
    device = get_device()

    # Data
    sequences, labels = load_data(args.data, config)
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_loader = DataLoader(TradingSequenceDataset(X_train, y_train), batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(TradingSequenceDataset(X_val, y_val), batch_size=config["training"]["batch_size"])

    # Model
    model = TradingBehaviorAnalyzer(config).to(device)
    logger.info(f"Model parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 2.0]).to(device))
    optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])

    with mlflow.start_run():
        mlflow.log_params({
            "model_name": config["model"]["name"],
            "base_type": "BEHAVIOR_ANALYZER",
            "domain": "finance",
            "n_params": count_parameters(model),
        })

        best_f1 = 0
        patience_counter = 0

        for epoch in range(config["training"]["epochs"]):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            metrics = evaluate(model, val_loader, device)

            mlflow.log_metrics({"train_loss": train_loss, **{f"val_{k}": v for k, v in metrics.items()}}, step=epoch)
            logger.info(f"Epoch {epoch+1} — Loss: {train_loss:.4f}, F1: {metrics['f1']:.4f}")

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                patience_counter = 0
                ckpt = Path(args.output) / "best_model.pt"
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt)
            else:
                patience_counter += 1

            if patience_counter >= config["training"]["early_stopping_patience"]:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Export
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        model.cpu().eval()
        arch = config["architecture"]
        dummy = torch.randn(1, arch["seq_length"], arch["input_dim"])
        onnx_path = Path(args.output) / f"{config['model']['name']}_v{config['model']['version']}.onnx"
        export_pytorch_onnx(model, dummy, str(onnx_path))
        mlflow.log_artifact(str(onnx_path))

        logger.info(f"Done. Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
