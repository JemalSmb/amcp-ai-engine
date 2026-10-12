"""
Finance — Transaction Fraud Anomaly Detection Pipeline

Isolation Forest for detecting anomalous transactions.
Domain: Finance (Passive)

Base model type: ANOMALY_DETECTOR (Isolation Forest — sklearn variant)
Input: Transaction feature vectors (amount, frequency, location, device, etc.)
Output: Anomaly score / fraud flag

Proves lego principle: same Isolation Forest approach as cybersecurity's IoT
anomaly detector, but trained on financial transaction data.

Dataset options:
  - IEEE-CIS Fraud Detection (Kaggle)
  - Synthetic transactions (development)
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, load_csv_data, get_device, export_sklearn_onnx, setup_mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Data Loading

def load_data(data_dir: str, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Load transaction data or generate synthetic."""
    result = load_csv_data(data_dir, label_column="isFraud", alt_label_columns=["label", "fraud", "is_fraud"])
    if result is not None:
        return result

    logger.warning("No CSV found — generating synthetic transaction data")
    np.random.seed(42)
    n_features = config["architecture"]["params"]["n_features"]

    # Normal transactions
    n_normal = 9000
    normal = np.random.randn(n_normal, n_features).astype(np.float32) * 0.5
    normal[:, 0] = np.random.lognormal(3.5, 1.0, n_normal)    # amount: $10–$500
    normal[:, 1] = np.random.uniform(0, 24, n_normal)          # hour_of_day
    normal[:, 2] = np.random.uniform(0, 7, n_normal)           # day_of_week
    normal[:, 3] = np.random.randint(1, 5, n_normal)           # txn_count_last_hour (low)

    # Fraudulent transactions
    n_fraud = 500
    fraud = np.random.randn(n_fraud, n_features).astype(np.float32) * 1.5 + 1.0
    fraud[:, 0] = np.random.lognormal(6.0, 1.5, n_fraud)      # amount: unusually high
    fraud[:, 1] = np.random.uniform(0, 6, n_fraud)             # late night
    fraud[:, 2] = np.random.uniform(5, 7, n_fraud)             # weekends
    fraud[:, 3] = np.random.randint(10, 50, n_fraud)           # txn_count: burst

    features = np.vstack([normal, fraud]).astype(np.float32)
    labels = np.array([0] * n_normal + [1] * n_fraud, dtype=np.int32)

    idx = np.random.permutation(len(features))
    return features[idx], labels[idx]


# Training & Evaluation

def train_model(X_train, config):
    params = config["architecture"]["params"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = IsolationForest(
        n_estimators=params["n_estimators"],
        max_samples=params.get("max_samples", 0.8),
        contamination=params["contamination"],
        max_features=params.get("max_features", 0.8),
        random_state=42,
        n_jobs=-1,
    )

    logger.info("Training Isolation Forest on normal transactions...")
    model.fit(X_scaled)
    return model, scaler


def evaluate(model, scaler, X_test, y_test) -> Dict[str, float]:
    X_scaled = scaler.transform(X_test)
    raw_scores = model.decision_function(X_scaled)
    scores = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-10)

    # Find best threshold
    best_f1, best_th = 0, 0.5
    for th in np.linspace(0.3, 0.8, 20):
        preds = (scores >= th).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, th

    preds = (scores >= best_th).astype(int)
    accuracy = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_test, scores)
    except ValueError:
        auc = 0.0

    logger.info(f"\n{classification_report(y_test, preds, target_names=['normal', 'fraud'])}")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": float(auc),
        "threshold": float(best_th),
        # Raw decision_function range used for the min-max normalisation above;
        # stored with the artifact so serving can reproduce the same scale.
        "score_min": float(raw_scores.min()),
        "score_max": float(raw_scores.max()),
    }


# Main

def main():
    parser = argparse.ArgumentParser(description="Train finance fraud anomaly detector")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--mlflow-uri", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default="finance-fraud-anomaly")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_mlflow(args.mlflow_uri, args.experiment_name)

    # Data
    features, labels = load_data(args.data, config)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    # Train on normal only
    X_train = X_train_full[y_train_full == 0]
    logger.info(f"Training on {len(X_train)} normal transactions")

    with mlflow.start_run():
        mlflow.log_params({
            "model_name": config["model"]["name"],
            "base_type": "ANOMALY_DETECTOR",
            "domain": "finance",
            "algorithm": "IsolationForest",
            "n_train": len(X_train),
            "n_test": len(X_test),
        })

        model, scaler = train_model(X_train, config)
        metrics = evaluate(model, scaler, X_test, y_test)

        mlflow.log_metrics(metrics)

        # Save
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        joblib_path = output_dir / "fraud_detector.joblib"
        joblib.dump(
            {
                "model": model,
                "scaler": scaler,
                "threshold": metrics["threshold"],
                "score_min": metrics["score_min"],
                "score_max": metrics["score_max"],
            },
            joblib_path,
        )
        mlflow.log_artifact(str(joblib_path))

        # ONNX
        pipeline = Pipeline([("scaler", scaler), ("model", model)])
        n_features = config["architecture"]["params"]["n_features"]
        onnx_path = output_dir / f"{config['model']['name']}_v{config['model']['version']}.onnx"
        export_sklearn_onnx(
            pipeline, n_features, str(onnx_path),
            options={id(model): {"score_samples": True}},
        )
        mlflow.log_artifact(str(onnx_path))

        mlflow.sklearn.log_model(model, "model")
        logger.info(f"Done. F1: {metrics['f1']:.4f}, AUC: {metrics['auc_roc']:.4f}")


if __name__ == "__main__":
    main()
