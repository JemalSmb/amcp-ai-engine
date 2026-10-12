"""
Anomaly Detection Model Training Pipeline

Unsupervised anomaly detection using Isolation Forest.
Detects anomalous network/IoT traffic without labeled attack data.

Base model type: ANOMALY_DETECTOR
See: ai/models/bases/__init__.py for cross-domain base architectures.
See: domains/cybersecurity/models/service_models.yaml for service-model mapping.

Note: This pipeline uses scikit-learn IsolationForest for the trained model.
      The PyTorch AnomalyDetectorBase (Autoencoder) in bases/ serves as the
      deep-learning alternative for services requiring gradient-based detection.
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import yaml
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(data_dir: str, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data from directory.

    Expected: CSV with numeric features + a 'label' column.
    label=0 -> benign, label=1 -> attack/anomaly.

    Falls back to synthetic data for development.
    """
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))

    if csv_files:
        import pandas as pd

        logger.info(f"Loading real data from {csv_files[0]}")
        df = pd.read_csv(csv_files[0])

        # Separate features and labels
        if "label" in df.columns:
            labels = df["label"].values
            features = df.drop(columns=["label"]).select_dtypes(include=[np.number]).values
        else:
            # Unsupervised: no labels available
            features = df.select_dtypes(include=[np.number]).values
            labels = np.zeros(len(features))  # All assumed benign

        # Replace inf/nan
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)

        logger.info(f"Loaded {len(features)} samples, {features.shape[1]} features")
        return features.astype(np.float32), labels.astype(np.int32)

    # Synthetic fallback for development
    logger.warning("No CSV found — generating synthetic data for development")
    np.random.seed(42)

    n_features = config["input"]["shape"][1]

    # Benign traffic: tight cluster
    n_benign = 9000
    benign = np.random.randn(n_benign, n_features).astype(np.float32) * 0.5

    # Anomalous traffic: scattered outliers
    n_anomaly = 500
    anomaly = np.random.randn(n_anomaly, n_features).astype(np.float32) * 3.0 + 2.0

    features = np.vstack([benign, anomaly])
    labels = np.array([0] * n_benign + [1] * n_anomaly, dtype=np.int32)

    # Shuffle
    idx = np.random.permutation(len(features))
    return features[idx], labels[idx]


def train_model(
    X_train: np.ndarray, config: Dict[str, Any]
) -> Tuple[IsolationForest, StandardScaler]:
    """Train Isolation Forest on benign-only data."""

    params = config["architecture"]["params"]

    # Fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Train Isolation Forest
    model = IsolationForest(
        n_estimators=params["n_estimators"],
        max_samples=params["max_samples"],
        contamination=params["contamination"],
        max_features=params["max_features"],
        random_state=params["random_state"],
        n_jobs=-1,
    )

    logger.info("Training Isolation Forest...")
    model.fit(X_scaled)
    logger.info("Training complete")

    return model, scaler


def evaluate(
    model: IsolationForest,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate anomaly detector on labeled test data."""

    X_scaled = scaler.transform(X_test)

    # Isolation Forest: decision_function returns anomaly scores
    # More negative = more anomalous
    raw_scores = model.decision_function(X_scaled)

    # Normalize scores to [0, 1] where 1 = most anomalous
    scores = 1 - (raw_scores - raw_scores.min()) / (
        raw_scores.max() - raw_scores.min() + 1e-10
    )

    # Apply threshold: score >= threshold -> anomaly (1)
    predictions = (scores >= threshold).astype(int)

    # Metrics
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", zero_division=0
    )

    try:
        auc = roc_auc_score(y_test, scores)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(y_test, predictions)

    logger.info(f"\n{classification_report(y_test, predictions, target_names=['benign', 'anomaly'])}")
    logger.info(f"Confusion Matrix:\n{cm}")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": float(auc),
        "true_negatives": int(cm[0][0]),
        "false_positives": int(cm[0][1]),
        "false_negatives": int(cm[1][0]),
        "true_positives": int(cm[1][1]),
        # Raw decision_function range used for the min-max normalisation above;
        # stored with the artifact so serving can reproduce the same scale.
        "score_min": float(raw_scores.min()),
        "score_max": float(raw_scores.max()),
    }


def export_onnx(
    model: IsolationForest,
    scaler: StandardScaler,
    config: Dict[str, Any],
    output_path: str,
):
    """Export Isolation Forest to ONNX format."""

    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([("scaler", scaler), ("model", model)])

    n_features = config["input"]["shape"][1]
    initial_type = [("input", FloatTensorType([None, n_features]))]

    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_type,
        target_opset=config["export"]["opset_version"],
        options={id(model): {"score_samples": True}},
    )

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    logger.info(f"Model exported to ONNX: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train anomaly detection model")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--data", type=str, required=True, help="Data directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--mlflow-uri", type=str, default=None, help="MLflow tracking URI")
    parser.add_argument("--experiment-name", type=str, default="anomaly-detection")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup MLflow
    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)

    # Load data
    features, labels = load_data(args.data, config)

    # Split: train on benign only, test on everything
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        features,
        labels,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=labels,
    )

    # Filter training set to benign-only (unsupervised: model learns "normal")
    if config["training"].get("use_only_benign", True):
        benign_mask = y_train_full == 0
        X_train = X_train_full[benign_mask]
        logger.info(
            f"Training on {len(X_train)} benign samples "
            f"(filtered from {len(X_train_full)} total)"
        )
    else:
        X_train = X_train_full

    # Train
    with mlflow.start_run():
        params = config["architecture"]["params"]
        mlflow.log_params(
            {
                "model_name": config["model"]["name"],
                "model_version": config["model"]["version"],
                "algorithm": "IsolationForest",
                "n_estimators": params["n_estimators"],
                "max_samples": params["max_samples"],
                "contamination": params["contamination"],
                "n_features": features.shape[1],
                "n_train_samples": len(X_train),
                "n_test_samples": len(X_test),
            }
        )

        model, scaler = train_model(X_train, config)

        # Evaluate at multiple thresholds
        best_f1 = 0
        best_threshold = 0.5

        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            metrics = evaluate(model, scaler, X_test, y_test, threshold=threshold)
            logger.info(f"Threshold {threshold}: F1={metrics['f1']:.4f}, AUC={metrics['auc_roc']:.4f}")

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_threshold = threshold

        # Log best results
        final_metrics = evaluate(model, scaler, X_test, y_test, threshold=best_threshold)
        mlflow.log_metrics(
            {
                "best_threshold": best_threshold,
                "accuracy": final_metrics["accuracy"],
                "precision": final_metrics["precision"],
                "recall": final_metrics["recall"],
                "f1": final_metrics["f1"],
                "auc_roc": final_metrics["auc_roc"],
            }
        )

        # Save artifacts
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save sklearn model
        joblib_path = output_dir / "anomaly_detector.joblib"
        joblib.dump(
            {
                "model": model,
                "scaler": scaler,
                "threshold": best_threshold,
                "score_min": final_metrics["score_min"],
                "score_max": final_metrics["score_max"],
            },
            joblib_path,
        )
        mlflow.log_artifact(joblib_path)

        # Export to ONNX
        onnx_path = output_dir / f"{config['model']['name']}_v{config['model']['version']}.onnx"
        export_onnx(model, scaler, config, str(onnx_path))
        mlflow.log_artifact(onnx_path)

        # Log sklearn model to MLflow
        mlflow.sklearn.log_model(model, "model")

        logger.info(
            f"Training complete. Best F1: {best_f1:.4f} at threshold {best_threshold}"
        )


if __name__ == "__main__":
    main()
