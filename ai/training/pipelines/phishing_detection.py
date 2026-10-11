"""
Phishing Classification Model Training Pipeline

Binary classification using Random Forest to detect phishing URLs/emails.

Base model types: CONTENT_ANALYZER, THREAT_CLASSIFIER, ANOMALY_DETECTOR
See: ai/models/bases/__init__.py for cross-domain base architectures.
See: domains/cybersecurity/models/service_models.yaml for service-model mapping.

Note: This pipeline uses scikit-learn RandomForest for URL feature classification.
      The PyTorch ContentAnalyzerBase in bases/ serves as the deep-learning
      alternative for raw text/token-level analysis.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import yaml
import mlflow
import mlflow.sklearn
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import log_and_register_model, upload_artifact_to_supabase
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split, cross_val_score
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
    Load phishing dataset from directory.

    Expected: CSV with numeric features + a 'label' column.
    label=0 -> legitimate, label=1 -> phishing.

    Falls back to synthetic data for development.
    """
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))

    if csv_files:
        import pandas as pd

        logger.info(f"Loading real data from {csv_files[0]}")
        df = pd.read_csv(csv_files[0])

        label_candidates = [
            "label",
            "result",
            "CLASS_LABEL",
            "class_label",
            "target",
            "Target",
        ]
        label_column = next((column for column in label_candidates if column in df.columns), None)

        if label_column is not None:
            labels = df[label_column].values
            labels = np.where(labels == -1, 0, labels)

            feature_frame = df.drop(columns=[label_column])
            for identifier_column in ["id", "Id", "ID", "url", "URL"]:
                if identifier_column in feature_frame.columns:
                    feature_frame = feature_frame.drop(columns=[identifier_column])

            features = feature_frame.select_dtypes(include=[np.number]).values
        else:
            raise ValueError("Dataset must have a recognized label column")

        # Replace inf/nan
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)

        logger.info(
            f"Loaded {len(features)} samples, {features.shape[1]} features, "
            f"phishing ratio: {labels.mean():.2%}"
        )
        return features.astype(np.float32), labels.astype(np.int32)

    # Synthetic fallback for development
    logger.warning("No CSV found — generating synthetic data for development")
    np.random.seed(42)

    n_features = config["input"]["shape"][1]
    n_samples = 8000

    # Legitimate URLs: short, simple, low entropy
    n_legit = 5000
    legit = np.random.randn(n_legit, n_features).astype(np.float32) * 0.5
    legit[:, 0] = np.random.uniform(10, 50, n_legit)  # url_length: short
    legit[:, 3] = np.random.randint(1, 3, n_legit)  # num_dots: few
    legit[:, 13] = 1  # has_https: yes

    # Phishing URLs: long, complex, high entropy
    n_phish = 3000
    phish = np.random.randn(n_phish, n_features).astype(np.float32) * 1.5 + 1.0
    phish[:, 0] = np.random.uniform(50, 200, n_phish)  # url_length: long
    phish[:, 3] = np.random.randint(3, 8, n_phish)  # num_dots: many
    phish[:, 10] = np.random.choice([0, 1], n_phish, p=[0.3, 0.7])  # has_ip: often
    phish[:, 13] = np.random.choice([0, 1], n_phish, p=[0.6, 0.4])  # has_https: rarely

    features = np.vstack([legit, phish])
    labels = np.array([0] * n_legit + [1] * n_phish, dtype=np.int32)

    idx = np.random.permutation(len(features))
    return features[idx], labels[idx]


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict[str, Any],
) -> Tuple[RandomForestClassifier, StandardScaler]:
    """Train Random Forest classifier."""

    params = config["architecture"]["params"]

    # Fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Train Random Forest
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

    logger.info("Training Random Forest classifier...")

    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_scaled, y_train, cv=5, scoring="f1")
    logger.info(f"Cross-validation F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Full training
    model.fit(X_scaled, y_train)
    logger.info("Training complete")

    return model, scaler


def evaluate(
    model: RandomForestClassifier,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate phishing classifier on test data."""

    X_scaled = scaler.transform(X_test)

    # Predictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]  # P(phishing)

    # Metrics
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", zero_division=0
    )

    try:
        auc = roc_auc_score(y_test, probabilities)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(y_test, predictions)

    # Feature importances
    importances = model.feature_importances_

    logger.info(
        f"\n{classification_report(y_test, predictions, target_names=['legitimate', 'phishing'])}"
    )
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
        "feature_importances": importances.tolist(),
        "cv_f1_mean": 0.0,  # Filled in main()
    }


def export_onnx(
    model: RandomForestClassifier,
    scaler: StandardScaler,
    n_features: int,
    opset_version: int,
    output_path: str,
):
    """Export Random Forest to ONNX format."""

    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([("scaler", scaler), ("model", model)])

    initial_type = [("input", FloatTensorType([None, n_features]))]

    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_type,
        target_opset=opset_version,
        options={id(model): {"zipmap": False}},  # Return probabilities as array
    )

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    logger.info(f"Model exported to ONNX: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train phishing classification model")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--data", type=str, required=True, help="Data directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--mlflow-uri", type=str, default=None, help="MLflow tracking URI")
    parser.add_argument("--experiment-name", type=str, default="phishing-detection")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup MLflow
    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)

    # Load data
    features, labels = load_data(args.data, config)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=labels,
    )

    logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    # Train
    with mlflow.start_run() as run:
        params = config["architecture"]["params"]
        mlflow.log_params(
            {
                "model_name": config["model"]["name"],
                "model_version": config["model"]["version"],
                "algorithm": "RandomForest",
                "n_estimators": params["n_estimators"],
                "max_depth": params["max_depth"],
                "n_features": features.shape[1],
                "n_train_samples": len(X_train),
                "n_test_samples": len(X_test),
                "phishing_ratio_train": float(y_train.mean()),
                "phishing_ratio_test": float(y_test.mean()),
            }
        )

        model, scaler = train_model(X_train, y_train, config)

        # Evaluate
        metrics = evaluate(model, scaler, X_test, y_test)

        # Log metrics
        mlflow.log_metrics(
            {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "auc_roc": metrics["auc_roc"],
            }
        )

        # Log top feature importances
        importances = np.array(metrics["feature_importances"])
        top_indices = np.argsort(importances)[::-1][:10]
        feature_names = config["input"].get("features", [f"f{i}" for i in range(len(importances))])

        logger.info("Top 10 features:")
        for rank, idx in enumerate(top_indices):
            name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            logger.info(f"  {rank + 1}. {name}: {importances[idx]:.4f}")
            mlflow.log_metric(f"importance_{name}", importances[idx])

        # Save artifacts
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save sklearn model
        joblib_path = output_dir / "phishing_classifier.joblib"
        joblib.dump({"model": model, "scaler": scaler}, joblib_path)
        mlflow.log_artifact(joblib_path)

        # Export to ONNX
        onnx_path = output_dir / f"{config['model']['name']}_v{config['model']['version']}.onnx"
        export_onnx(
            model,
            scaler,
            features.shape[1],
            config["export"]["opset_version"],
            str(onnx_path),
        )
        mlflow.log_artifact(onnx_path)

        # Push both artifacts to Supabase Storage (no-ops with a warning if
        # SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY aren't set), so the deployed
        # inference service can load them without a redeploy or rebuilt image.
        upload_artifact_to_supabase(str(joblib_path), "phishing_detection")
        upload_artifact_to_supabase(str(onnx_path), "phishing_detection")

        # Log the {scaler, model} pair as one pyfunc model and register it,
        # so serving can poll the registry for new versions instead of
        # requiring a redeploy per model update.
        registered_name = config["model"]["name"]
        mv = log_and_register_model(model, scaler, run.info.run_id, registered_name)
        logger.info(f"Registered '{registered_name}' as version {mv.version}")

        logger.info(
            f"Training complete. F1: {metrics['f1']:.4f}, AUC: {metrics['auc_roc']:.4f}"
        )


if __name__ == "__main__":
    main()
