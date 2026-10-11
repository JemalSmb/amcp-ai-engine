"""
Manufacturing — Process Anomaly Detection Pipeline

Autoencoder for detecting anomalies in manufacturing process parameters.
Domain: Manufacturing (Passive)

Base model type: ANOMALY_DETECTOR (Autoencoder)
Input: Machine sensor readings (vibration, temperature, current, pressure, etc.)
Output: Anomaly score (reconstruction error)

Dataset options:
  - NASA Turbofan Engine Degradation (C-MAPSS)
  - SECOM Semiconductor Manufacturing
  - Synthetic process data (development)
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, load_csv_data, get_device, export_pytorch_onnx, setup_mlflow, count_parameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessAnomalyDetector(nn.Module):
    """Autoencoder for process parameter anomaly detection. ~22K params."""
    def __init__(self, config):
        super().__init__()
        d = config["architecture"]["input_dim"]
        z = config["architecture"]["latent_dim"]
        self.encoder = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, z))
        self.decoder = nn.Sequential(nn.Linear(z, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, d))

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def anomaly_score(self, x):
        return torch.mean((x - self.forward(x)) ** 2, dim=1)


def load_data(data_dir, config):
    result = load_csv_data(data_dir, label_column="fault", alt_label_columns=["label", "anomaly", "failure"])
    if result is not None:
        return result

    logger.warning("No CSV found — generating synthetic process data")
    np.random.seed(42)
    nf = config["architecture"]["input_dim"]

    n_normal = 8000
    normal = np.column_stack([
        np.random.normal(0.5, 0.1, n_normal),   # vibration_rms
        np.random.normal(65, 5, n_normal),        # temperature_c
        np.random.normal(12, 1.5, n_normal),      # current_a
        np.random.normal(3.2, 0.3, n_normal),     # pressure_bar
        np.random.normal(1500, 50, n_normal),      # rpm
        np.random.normal(75, 3, n_normal),         # efficiency_pct
    ])
    if nf > 6: normal = np.hstack([normal, np.random.randn(n_normal, nf - 6) * 0.3])

    n_fault = 400
    fault = np.column_stack([
        np.random.normal(2.5, 0.8, n_fault),    # high vibration
        np.random.normal(95, 10, n_fault),        # overheating
        np.random.normal(20, 3, n_fault),          # high current
        np.random.normal(1.5, 0.5, n_fault),      # low pressure
        np.random.normal(800, 200, n_fault),       # rpm drop
        np.random.normal(45, 10, n_fault),         # low efficiency
    ])
    if nf > 6: fault = np.hstack([fault, np.random.randn(n_fault, nf - 6) * 1.5])

    features = np.vstack([normal, fault]).astype(np.float32)
    labels = np.array([0] * n_normal + [1] * n_fault, dtype=np.int32)
    idx = np.random.permutation(len(features))
    return features[idx], labels[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--mlflow-uri", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default="manufacturing-process-anomaly")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_mlflow(args.mlflow_uri, args.experiment_name)
    device = get_device()

    features, labels = load_data(args.data, config)
    X_train_full, X_test, y_train_full, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)
    X_train = X_train_full[y_train_full == 0]

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train)), batch_size=config["training"]["batch_size"], shuffle=True)
    model = ProcessAnomalyDetector(config).to(device)
    logger.info(f"Model parameters: {count_parameters(model):,}")
    optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"])

    with mlflow.start_run():
        mlflow.log_params({"model_name": config["model"]["name"], "base_type": "ANOMALY_DETECTOR", "domain": "manufacturing", "n_params": count_parameters(model)})

        best_f1 = 0
        for epoch in range(config["training"]["epochs"]):
            model.train()
            total_loss = 0
            for (batch,) in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = nn.MSELoss()(model(batch), batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            with torch.no_grad():
                scores = model.anomaly_score(torch.FloatTensor(X_test).to(device)).cpu().numpy()
            best_th_f1, best_th = 0, 0.5
            for th in np.linspace(np.percentile(scores, 80), np.percentile(scores, 99), 20):
                p = (scores >= th).astype(int)
                _, _, f1, _ = precision_recall_fscore_support(y_test, p, average="binary", zero_division=0)
                if f1 > best_th_f1: best_th_f1, best_th = f1, th

            mlflow.log_metrics({"train_loss": total_loss / len(train_loader), "val_f1": best_th_f1}, step=epoch)
            logger.info(f"Epoch {epoch+1} — Loss: {total_loss / len(train_loader):.4f}, F1: {best_th_f1:.4f}")

            if best_th_f1 > best_f1:
                best_f1 = best_th_f1
                ckpt = Path(args.output) / "best_model.pt"
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt)

        model.load_state_dict(torch.load(ckpt, weights_only=True))
        model.cpu().eval()
        dummy = torch.randn(1, config["architecture"]["input_dim"])
        onnx_path = Path(args.output) / f"{config['model']['name']}_v{config['model']['version']}.onnx"
        export_pytorch_onnx(model, dummy, str(onnx_path))
        mlflow.log_artifact(str(onnx_path))
        logger.info(f"Done. Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
