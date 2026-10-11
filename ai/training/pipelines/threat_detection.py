"""
Threat Detection Model Training Pipeline

Multi-class Random Forest classifier for network threat classification.
Domain: Cybersecurity (Passive)

Base model type: THREAT_CLASSIFIER
Input: Network flow features (CTU-13 binetflow derived)
Output: Multi-class classification (background / normal / botnet)

Dataset:
  - CTU-13 (preprocessed binetflow → ctu13_flows.csv)
  - Synthetic data fallback for development
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
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
    Load threat detection data.

    Expected: CSV with numeric features + 'label' column.
    label: 0=background, 1=normal, 2=botnet

    Falls back to synthetic data for development.
    """
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))

    if csv_files:
        import pandas as pd

        logger.info(f"Loading real data from {csv_files[0]}")
        df = pd.read_csv(csv_files[0])

        if "label" in df.columns:
            labels = df["label"].values
            features = df.drop(columns=["label"]).select_dtypes(include=[np.number]).values
        else:
            raise ValueError("Dataset must have a 'label' column")

        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)

        logger.info(
            f"Loaded {len(features)} samples, {features.shape[1]} features, "
            f"{len(np.unique(labels))} classes"
        )
        return features.astype(np.float32), labels.astype(np.int32)

    # Synthetic fallback
    logger.warning("No CSV found — generating synthetic threat data")
    np.random.seed(42)

    n_features = config["input"]["shape"][1]
    n_classes = config["architecture"]["output"]["num_classes"]

    # Background: low-activity flows
    n_bg = 5000
    bg = np.random.randn(n_bg, n_features).astype(np.float32) * 0.3
    bg[:, 8] = np.random.randint(1, 5, n_bg)    # low packet counts
    bg[:, 9] = np.random.uniform(40, 200, n_bg)  # low bytes

    # Normal: typical web/service flows
    n_norm = 3000
    normal = np.random.randn(n_norm, n_features).astype(np.float32) * 0.5 + 0.5
    normal[:, 8] = np.random.randint(5, 50, n_norm)
    normal[:, 9] = np.random.uniform(200, 5000, n_norm)

    # Botnet: distinct patterns
    n_bot = 1000
    botnet = np.random.randn(n_bot, n_features).astype(np.float32) * 1.5 + 2.0
    botnet[:, 8] = np.random.randint(50, 500, n_bot)
    botnet[:, 9] = np.random.uniform(1000, 50000, n_bot)

    features = np.vstack([bg, normal, botnet])
    labels = np.array([0] * n_bg + [1] * n_norm + [2] * n_bot, dtype=np.int32)

    idx = np.random.permutation(len(features))
    return features[idx], labels[idx]


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict[str, Any],
) -> Tuple[RandomForestClassifier, StandardScaler]:
    """Train Random Forest multi-class classifier."""
    params = config["architecture"]["params"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        class_weight=params["class_weight"],
        random_state=params["random_state"],
        n_jobs=params["n_jobs"],
    )

    logger.info("Training Random Forest threat classifier...")

    cv_scores = cross_val_score(model, X_scaled, y_train, cv=5, scoring="f1_macro")
    logger.info(f"Cross-validation F1 (macro): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    model.fit(X_scaled, y_train)
    logger.info("Training complete")

    return model, scaler


def evaluate(
    model: RandomForestClassifier,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate threat classifier on test data."""
    X_scaled = scaler.transform(X_test)
    predictions = model.predict(X_scaled)

    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro", zero_division=0
    )

    cm = confusion_matrix(y_test, predictions)

    target_names = ["background", "normal", "botnet"]
    logger.info(f"\n{classification_report(y_test, predictions, target_names=target_names)}")
    logger.info(f"Confusion Matrix:\n{cm}")

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def export_onnx(
    model: RandomForestClassifier,
    scaler: StandardScaler,
    config: Dict[str, Any],
    output_path: str,
):
    """Export classifier to ONNX format."""
    pipeline = Pipeline([("scaler", scaler), ("model", model)])

    n_features = config["input"]["shape"][1]
    initial_type = [("input", FloatTensorType([None, n_features]))]

    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_type,
        target_opset=config["export"]["opset_version"],
    )

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    logger.info(f"Model exported to ONNX: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train threat detection model")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--data", type=str, required=True, help="Data directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--mlflow-uri", type=str, default=None, help="MLflow tracking URI")
    parser.add_argument("--experiment-name", type=str, default="threat-detection")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)

    features, labels = load_data(args.data, config)

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=labels,
    )

    with mlflow.start_run():
        params = config["architecture"]["params"]
        mlflow.log_params({
            "model_name": config["model"]["name"],
            "model_version": config["model"]["version"],
            "algorithm": "RandomForest",
            "n_estimators": params["n_estimators"],
            "n_features": features.shape[1],
            "n_classes": len(np.unique(labels)),
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
        })

        model, scaler = train_model(X_train, y_train, config)
        metrics = evaluate(model, scaler, X_test, y_test)

        mlflow.log_metrics(metrics)

        # Save artifacts
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        joblib_path = output_dir / "threat_classifier.joblib"
        joblib.dump({"model": model, "scaler": scaler}, joblib_path)
        mlflow.log_artifact(joblib_path)

        onnx_path = output_dir / f"{config['model']['name']}_v{config['model']['version']}.onnx"
        export_onnx(model, scaler, config, str(onnx_path))
        mlflow.log_artifact(onnx_path)

        mlflow.sklearn.log_model(model, "model")

        logger.info(f"Training complete. F1 (macro): {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
