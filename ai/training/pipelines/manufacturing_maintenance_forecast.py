"""
Manufacturing — Predictive Maintenance Forecaster Pipeline

Transformer forecaster for remaining useful life (RUL) prediction.
Domain: Manufacturing (Active)

Base model type: FORECASTER (Transformer)
Input: Equipment sensor time-series (vibration, temp, current over time)
Output: Predicted future sensor readings (degradation curve)

Dataset options:
  - NASA C-MAPSS Turbofan Engine Degradation
  - PHM Data Challenge datasets
  - Synthetic degradation sequences (development)
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
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, generate_synthetic_timeseries, get_device, export_pytorch_onnx, setup_mlflow, count_parameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class MaintenanceForecaster(nn.Module):
    """Transformer for equipment degradation forecasting. ~40K params."""
    def __init__(self, config):
        super().__init__()
        arch = config["architecture"]
        d = arch["input_dim"]
        dm = arch["d_model"]
        fh = arch.get("forecast_horizon", 1)

        self.embed = nn.Linear(d, dm)
        self.pos = nn.Parameter(torch.randn(1, 512, dm) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model=dm, nhead=arch["nhead"], dim_feedforward=dm * 2, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, arch["num_layers"])
        self.fc = nn.Linear(dm, d * fh)
        self.fh = fh
        self.d = d

    def forward(self, x):
        sl = x.size(1)
        e = self.embed(x) + self.pos[:, :sl, :]
        t = self.transformer(e)
        return self.fc(t[:, -1, :]).view(-1, self.fh, self.d)


def load_data(data_dir, config):
    data_path = Path(data_dir)
    if list(data_path.glob("*.npy")):
        return np.load(data_path / "sequences.npy").astype(np.float32), np.load(data_path / "targets.npy").astype(np.float32)

    logger.warning("No data found — generating synthetic degradation data")
    arch = config["architecture"]
    np.random.seed(42)
    n = config["training"]["synthetic_samples"]
    sl = arch["seq_length"]
    nf = arch["input_dim"]
    fh = arch.get("forecast_horizon", 1)

    X_all, y_all = [], []
    for _ in range(n):
        # Simulate degradation: gradually increasing noise/drift
        t = np.linspace(0, 1, sl + fh)
        degradation = t ** 1.5 * np.random.uniform(0.5, 2.0)
        data = np.column_stack([
            np.random.randn(sl + fh) * (0.1 + degradation * 0.3) + degradation
            for _ in range(nf)
        ]).astype(np.float32)
        X_all.append(data[:sl])
        y_all.append(data[sl:sl + fh])

    return np.array(X_all, dtype=np.float32), np.array(y_all, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--mlflow-uri", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default="manufacturing-maintenance-forecast")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_mlflow(args.mlflow_uri, args.experiment_name)
    device = get_device()

    X, y = load_data(args.data, config)
    split = int(len(X) * 0.8)
    train_loader = DataLoader(TimeSeriesDataset(X[:split], y[:split]), batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X[split:], y[split:]), batch_size=config["training"]["batch_size"])

    model = MaintenanceForecaster(config).to(device)
    logger.info(f"Model parameters: {count_parameters(model):,}")
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    with mlflow.start_run():
        mlflow.log_params({"model_name": config["model"]["name"], "base_type": "FORECASTER", "domain": "manufacturing", "n_params": count_parameters(model)})

        best_mse = float("inf")
        patience = 0
        for epoch in range(config["training"]["epochs"]):
            model.train()
            tl = sum(criterion(model(xb.to(device)), yb.to(device)).backward() or criterion(model(xb.to(device)), yb.to(device)).item() for xb, yb in train_loader) if False else 0
            # Proper training loop
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            train_loss = total_loss / len(train_loader)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    val_loss += criterion(model(xb.to(device)), yb.to(device)).item()
            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            mlflow.log_metrics({"train_loss": train_loss, "val_mse": val_loss}, step=epoch)
            logger.info(f"Epoch {epoch+1} — Loss: {train_loss:.4f}, Val MSE: {val_loss:.4f}")

            if val_loss < best_mse:
                best_mse = val_loss
                patience = 0
                ckpt = Path(args.output) / "best_model.pt"
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt)
            else:
                patience += 1
            if patience >= config["training"]["early_stopping_patience"]: break

        model.load_state_dict(torch.load(ckpt, weights_only=True))
        model.cpu().eval()
        arch = config["architecture"]
        onnx_path = Path(args.output) / f"{config['model']['name']}_v{config['model']['version']}.onnx"
        export_pytorch_onnx(model, torch.randn(1, arch["seq_length"], arch["input_dim"]), str(onnx_path))
        mlflow.log_artifact(str(onnx_path))
        logger.info(f"Done. Best MSE: {best_mse:.4f}")


if __name__ == "__main__":
    main()
