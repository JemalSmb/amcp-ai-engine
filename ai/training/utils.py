"""
Shared Training Utilities

Common functions reused across all domain training pipelines.
Keeps individual pipeline scripts lightweight and DRY.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import yaml
import mlflow
from dotenv import load_dotenv

# Reuses inference/.env so training and serving share one set of Supabase
# credentials instead of keeping two copies in sync.
load_dotenv(Path(__file__).resolve().parent.parent / "inference" / ".env")

logger = logging.getLogger(__name__)



# Configuration


def load_config(config_path: str) -> Dict[str, Any]:
    """Load model/training configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)



# Data Loading Helpers


def load_csv_data(
    data_dir: str,
    label_column: str = "label",
    alt_label_columns: List[str] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load first CSV found in data_dir.

    Returns (features, labels) or None if no CSV found.
    Handles common label column names and encodings.
    """
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))

    if not csv_files:
        return None

    import pandas as pd

    logger.info(f"Loading data from {csv_files[0]}")
    df = pd.read_csv(csv_files[0])

    # Find label column
    label_col = None
    all_candidates = [label_column] + (alt_label_columns or [])
    for col in all_candidates:
        if col in df.columns:
            label_col = col
            break

    if label_col:
        labels = df[label_col].values
        features = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).values
        # Map -1 → 0 if present
        if labels.min() == -1:
            labels = np.where(labels == -1, 0, labels)
    else:
        features = df.select_dtypes(include=[np.number]).values
        labels = np.zeros(len(features))

    features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)

    logger.info(f"Loaded {len(features)} samples, {features.shape[1]} features")
    return features.astype(np.float32), labels.astype(np.int32)



# Synthetic Data Generators (per base model type)


def generate_synthetic_classification(
    n_samples: int,
    n_features: int,
    n_classes: int,
    class_ratios: List[float] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic multi-class classification data."""
    np.random.seed(seed)

    if class_ratios is None:
        class_ratios = [1.0 / n_classes] * n_classes

    # Normalise
    class_ratios = np.array(class_ratios)
    class_ratios /= class_ratios.sum()

    features_all, labels_all = [], []
    for cls_idx, ratio in enumerate(class_ratios):
        n = int(n_samples * ratio)
        # Each class gets a different centroid
        centroid = np.random.randn(n_features) * 0.5
        data = np.random.randn(n, n_features).astype(np.float32) * 0.3 + centroid
        features_all.append(data)
        labels_all.append(np.full(n, cls_idx, dtype=np.int32))

    features = np.vstack(features_all)
    labels = np.concatenate(labels_all)

    idx = np.random.permutation(len(features))
    return features[idx], labels[idx]


def generate_synthetic_anomaly(
    n_benign: int,
    n_anomaly: int,
    n_features: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic anomaly detection data."""
    np.random.seed(seed)

    benign = np.random.randn(n_benign, n_features).astype(np.float32) * 0.5
    anomaly = np.random.randn(n_anomaly, n_features).astype(np.float32) * 3.0 + 2.0

    features = np.vstack([benign, anomaly])
    labels = np.array([0] * n_benign + [1] * n_anomaly, dtype=np.int32)

    idx = np.random.permutation(len(features))
    return features[idx], labels[idx]


def generate_synthetic_sequences(
    n_sequences: int,
    seq_length: int,
    n_features: int,
    n_classes: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic sequential data for BiLSTM / Transformer models."""
    np.random.seed(seed)

    sequences = []
    labels = []

    per_class = n_sequences // n_classes

    for cls in range(n_classes):
        freq = 0.1 * (cls + 1)
        noise_level = 0.2 + 0.1 * cls

        for _ in range(per_class):
            t = np.linspace(0, 2 * np.pi, seq_length)
            base = np.sin(t * freq).reshape(-1, 1) * np.ones((1, n_features))
            noise = np.random.randn(seq_length, n_features) * noise_level
            sequences.append((base + noise).astype(np.float32))
            labels.append(cls)

    sequences = np.array(sequences)
    labels = np.array(labels, dtype=np.int32)

    idx = np.random.permutation(len(sequences))
    return sequences[idx], labels[idx]


def generate_synthetic_timeseries(
    n_samples: int,
    seq_length: int,
    n_features: int,
    forecast_horizon: int = 1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic time-series for forecasting models."""
    np.random.seed(seed)

    data = np.zeros((n_samples, seq_length + forecast_horizon, n_features), dtype=np.float32)

    for i in range(n_samples):
        freq = np.random.uniform(0.5, 3.0)
        phase = np.random.uniform(0, 2 * np.pi)
        t = np.linspace(0, 4 * np.pi, seq_length + forecast_horizon)
        for f in range(n_features):
            data[i, :, f] = np.sin(t * freq + phase + f * 0.3) + np.random.randn(len(t)) * 0.1

    X = data[:, :seq_length, :]
    y = data[:, seq_length:seq_length + forecast_horizon, :]

    return X, y



# MLflow Helpers


def setup_mlflow(tracking_uri: Optional[str], experiment_name: str):
    """Configure MLflow tracking."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


class ScaledClassifier(mlflow.pyfunc.PythonModel):
    """
    Bundles a fitted scaler + classifier as one MLflow model so serving
    only has to call .predict() and never has to know about a separate
    scaling step. Registering this (instead of the bare estimator) is
    what lets the inference service pull a real, versioned model from
    the registry instead of a static local file.
    """

    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, context, model_input, params=None):
        scaled = self.scaler.transform(model_input)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(scaled)[:, -1]
        return self.model.predict(scaled)


def log_and_register_model(model, scaler, run_id: str, registered_name: str):
    """
    Log a {scaler, model} pair as a single pyfunc model and register it
    in the MLflow Model Registry under `registered_name`. Returns the
    resulting ModelVersion.
    """
    import mlflow.pyfunc

    wrapper = ScaledClassifier(model, scaler)
    mlflow.pyfunc.log_model(name="model", python_model=wrapper)
    return mlflow.register_model(f"runs:/{run_id}/model", registered_name)



# Cloud Artifact Upload (Supabase Storage)


def upload_artifact_to_supabase(local_path: str, model_id: str, artifact_name: Optional[str] = None) -> bool:
    """
    Upload a trained artifact (.onnx / .joblib) to Supabase Storage so
    inference (registry.py's `_try_load_from_supabase`) can pull it without
    the file living in the repo.

    Call this at the end of a training pipeline, right after saving the
    local artifact, e.g.:

        upload_artifact_to_supabase(str(joblib_path), "phishing_detection")
        upload_artifact_to_supabase(str(onnx_path), "phishing_detection")

    Uploads to <bucket>/<model_id>/artifacts/<filename>, matching the path
    layout registry.py expects. Requires SUPABASE_URL and
    SUPABASE_SERVICE_ROLE_KEY in the environment; no-ops (logs a warning)
    if they're not set, so training still works without cloud storage.
    """
    supabase_url = os.environ.get("SUPABASE_URL", "")
    service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    bucket = os.environ.get("SUPABASE_MODELS_BUCKET", "models")

    if not supabase_url or not service_key:
        logger.warning("SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY not set -- skipping cloud upload for %s", local_path)
        return False

    import requests

    local_path = Path(local_path)
    name = artifact_name or local_path.name
    object_path = f"{model_id}/artifacts/{name}"
    url = f"{supabase_url}/storage/v1/object/{bucket}/{object_path}"

    with open(local_path, "rb") as f:
        resp = requests.put(
            url,
            headers={
                "Authorization": f"Bearer {service_key}",
                "x-upsert": "true",  # overwrite if this version was already uploaded
            },
            data=f.read(),
        )

    if resp.status_code >= 300:
        logger.error("Supabase upload failed for %s: %s %s", object_path, resp.status_code, resp.text)
        return False

    logger.info("Uploaded %s -> supabase://%s/%s", local_path, bucket, object_path)
    return True



# PyTorch Helpers


def get_device():
    """Get best available torch device."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model) -> int:
    """Count trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# ONNX Export


def export_pytorch_onnx(model, dummy_input, output_path: str, opset: int = 17):
    """Export a PyTorch model to ONNX."""
    import torch

    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    logger.info(f"ONNX model exported → {output_path}")


def export_sklearn_onnx(pipeline, n_features: int, output_path: str, opset: int = 17, options: dict = None):
    """Export a scikit-learn pipeline to ONNX."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_type,
        target_opset=opset,
        options=options,
    )

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    logger.info(f"ONNX model exported → {output_path}")
