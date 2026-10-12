"""
Model registry for the inference service.

Prefers pulling models from the MLflow Model Registry (real, versioned
artifacts a training run registered) and falls back to a static local
file, and finally to deterministic mock scoring: the same input always
produces the same output, so the endpoint contract is real and testable,
but the score has no predictive value until the model is actually trained
and registered.

`check_for_updates()` re-polls the registry for a newer version of each
registered model and hot-swaps it in place, so a new training run doesn't
require redeploying this service.
"""

import hashlib
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()  # picks up inference/.env when running outside Docker

logger = logging.getLogger(__name__)

AI_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = AI_ROOT / "models"
TRAINING_ROOT = AI_ROOT / "training"

# Registered models' pyfunc wrappers were pickled as `utils.ScaledClassifier`
# by the training pipelines; this makes that module importable here too.
sys.path.insert(0, str(TRAINING_ROOT))

DEFAULT_MLFLOW_URI = f"sqlite:///{TRAINING_ROOT / 'pipelines' / 'mlflow.db'}"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_URI)

# Supabase Storage: cloud-hosted fallback for trained artifacts (.onnx /
# .joblib) so they don't have to live in the git repo. Bucket layout is
# expected to mirror the local one: <bucket>/<model_id>/<artifact_rel>.
# https://supabase.com/dashboard -> Project Settings -> API for the URL/key,
# Storage -> create a bucket (e.g. "models") for SUPABASE_MODELS_BUCKET.
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_MODELS_BUCKET = os.environ.get("SUPABASE_MODELS_BUCKET", "models")
MODEL_CACHE_DIR = Path(os.environ.get("MODEL_CACHE_DIR", "/tmp/amcp-model-cache"))

# Minimum model set served, per the MVP plan. Each maps to its domain, an
# optional MLflow registered-model name (preferred source), and/or a
# static local artifact file (fallback).
SUPPORTED_MODELS: Dict[str, Dict[str, str]] = {
    "anomaly_detection": {"domain": "cybersecurity"},
    "phishing_detection": {
        "domain": "cybersecurity",
        "registered_name": "phishing_classifier",
        "artifact": "artifacts/phishing_classifier.joblib",
    },
    "finance_fraud_anomaly": {"domain": "finance"},
    "manufacturing_process_anomaly": {"domain": "manufacturing"},
}

MOCK_LABELS: Dict[str, tuple] = {
    "anomaly_detection": ("normal", "anomalous"),
    "phishing_detection": ("legitimate", "phishing"),
    "finance_fraud_anomaly": ("normal", "fraud_suspected"),
    "manufacturing_process_anomaly": ("normal", "anomalous"),
}


@dataclass
class LoadedModel:
    id: str
    domain: str
    name: str
    version: str
    architecture: str
    features: List[str]
    trained: bool
    source: str = "mock"  # "registry" | "local_file" | "mock"
    estimator: Optional[Any] = field(default=None, repr=False)
    scaler: Optional[Any] = field(default=None, repr=False)
    pyfunc_model: Optional[Any] = field(default=None, repr=False)
    load_error: Optional[str] = None


def _deterministic_score(model_id: str, features: Dict[str, float]) -> float:
    """Hash the input into a stable [0, 1] score. Same input -> same output."""
    payload = f"{model_id}:" + ",".join(f"{k}={v:.6f}" for k, v in sorted(features.items()))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _mlflow_client():
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.tracking.MlflowClient()


def _latest_registered_version(registered_name: str) -> Optional[str]:
    """Return the highest version number registered under this name, or None."""
    try:
        client = _mlflow_client()
        versions = client.search_model_versions(f"name='{registered_name}'")
    except Exception as exc:  # noqa: BLE001 - registry may be unreachable
        logger.warning("could not query MLflow registry for '%s': %s", registered_name, exc)
        return None
    if not versions:
        return None
    return str(max(int(v.version) for v in versions))


class ModelRegistry:
    def __init__(self) -> None:
        self._models: Dict[str, LoadedModel] = {}

    def load_all(self) -> None:
        for model_id, meta in SUPPORTED_MODELS.items():
            self._models[model_id] = self._load_one(model_id, meta)
        logger.info("Loaded %d model(s): %s", len(self._models), list(self._models))

    def _base_config(self, model_id: str) -> Dict[str, Any]:
        config_path = MODELS_ROOT / model_id / "config.yaml"
        if not config_path.exists():
            logger.warning("No config.yaml for model '%s' at %s", model_id, config_path)
            return {}
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}

    def _load_one(self, model_id: str, meta: Dict[str, str]) -> LoadedModel:
        config = self._base_config(model_id)
        model_meta = config.get("model", {})
        arch_meta = config.get("architecture", {})
        input_meta = config.get("input", {})
        declared_features = list(input_meta.get("features", []))

        base = LoadedModel(
            id=model_id,
            domain=meta["domain"],
            name=model_meta.get("name", model_id),
            version=str(model_meta.get("version", "0.0.0")),
            architecture=arch_meta.get("type", "unknown"),
            features=declared_features,
            trained=False,
        )

        registered_name = meta.get("registered_name")
        if registered_name:
            loaded = self._try_load_from_registry(model_id, registered_name, base)
            if loaded is not None:
                return loaded

        artifact_rel = meta.get("artifact")
        if artifact_rel:
            loaded = self._try_load_from_supabase(model_id, artifact_rel, base)
            if loaded is not None:
                return loaded

            loaded = self._try_load_from_file(model_id, artifact_rel, base)
            if loaded is not None:
                return loaded

        base.load_error = base.load_error or "no trained artifact available"
        return base

    def _try_load_from_registry(
        self, model_id: str, registered_name: str, base: LoadedModel
    ) -> Optional[LoadedModel]:
        version = _latest_registered_version(registered_name)
        if version is None:
            return None
        try:
            import mlflow

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            pyfunc_model = mlflow.pyfunc.load_model(f"models:/{registered_name}/{version}")
        except Exception as exc:  # noqa: BLE001 - fall back to file/mock
            logger.exception("failed to load '%s' v%s from MLflow", registered_name, version)
            base.load_error = f"registry load failed: {exc}"
            return None

        base.trained = True
        base.source = "registry"
        base.pyfunc_model = pyfunc_model
        base.version = version
        base.load_error = None
        logger.info("Loaded '%s' from MLflow registry, version %s", model_id, version)
        return base

    def _try_load_from_supabase(self, model_id: str, artifact_rel: str, base: LoadedModel) -> Optional[LoadedModel]:
        """Download the artifact from Supabase Storage into a local cache, then load it."""
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            return None  # not configured -- caller falls back to local file / mock

        cache_path = MODEL_CACHE_DIR / model_id / artifact_rel
        if not cache_path.exists():
            try:
                import requests

                object_path = f"{model_id}/{artifact_rel}"
                url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_MODELS_BUCKET}/{object_path}"
                resp = requests.get(
                    url,
                    headers={"Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"},
                    timeout=15,
                )
                resp.raise_for_status()
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(resp.content)
                logger.info("Downloaded '%s' artifact from Supabase Storage -> %s", model_id, cache_path)
            except Exception as exc:  # noqa: BLE001 - fall back to local file / mock
                logger.warning("could not fetch '%s' from Supabase Storage: %s", model_id, exc)
                return None

        return self._try_load_from_file(model_id, artifact_rel, base, base_dir=MODEL_CACHE_DIR / model_id)

    def _try_load_from_file(
        self, model_id: str, artifact_rel: str, base: LoadedModel, base_dir: Optional[Path] = None
    ) -> Optional[LoadedModel]:
        artifact_path = (base_dir or (MODELS_ROOT / model_id)) / artifact_rel
        if not artifact_path.exists():
            base.load_error = f"artifact not found: {artifact_path}"
            logger.warning(base.load_error)
            return None

        try:
            loaded = joblib.load(artifact_path)
        except Exception as exc:  # noqa: BLE001 - surface as degraded, not a crash
            base.load_error = str(exc)
            logger.exception("Failed to load artifact for '%s'", model_id)
            return None

        estimator = loaded.get("model") if isinstance(loaded, dict) else loaded
        scaler = loaded.get("scaler") if isinstance(loaded, dict) else None

        expected = getattr(estimator, "n_features_in_", None)
        if expected is not None and base.features and expected != len(base.features):
            base.load_error = (
                f"artifact/config mismatch: trained on {expected} features, "
                f"config.yaml declares {len(base.features)}"
            )
            logger.warning("'%s': %s", model_id, base.load_error)
            return None

        base.trained = True
        base.source = "local_file"
        base.estimator = estimator
        base.scaler = scaler
        base.load_error = None
        return base

    def check_for_updates(self) -> List[Dict[str, str]]:
        """Re-poll the registry for each model; hot-swap and report changes."""
        changes: List[Dict[str, str]] = []
        for model_id, meta in SUPPORTED_MODELS.items():
            registered_name = meta.get("registered_name")
            if not registered_name:
                continue

            current = self._models.get(model_id)
            latest_version = _latest_registered_version(registered_name)
            if latest_version is None:
                continue
            if current is not None and current.source == "registry" and current.version == latest_version:
                continue

            reloaded = self._load_one(model_id, meta)
            self._models[model_id] = reloaded
            if reloaded.source == "registry":
                changes.append({"model_id": model_id, "version": reloaded.version})
                logger.info("Hot-swapped '%s' to version %s", model_id, reloaded.version)
        return changes

    def list_models(self) -> List[LoadedModel]:
        return list(self._models.values())

    def get(self, model_id: str) -> Optional[LoadedModel]:
        return self._models.get(model_id)

    def predict(self, model_id: str, features: Dict[str, float]) -> Dict[str, Any]:
        model = self.get(model_id)
        if model is None:
            raise KeyError(model_id)

        warnings: List[str] = []
        normal_label, positive_label = MOCK_LABELS.get(model_id, ("normal", "flagged"))

        if model.trained and model.pyfunc_model is not None:
            score = self._score_with_pyfunc(model, features, warnings)
        elif model.trained and model.estimator is not None:
            vector = self._vectorize(model, features, warnings)
            if model.scaler is not None:
                vector = model.scaler.transform(vector)
            score = self._score_with_estimator(model.estimator, vector)
        else:
            if model.load_error:
                warnings.append(f"using mock scoring: {model.load_error}")
            else:
                warnings.append("using mock scoring: model not yet trained")
            score = _deterministic_score(model_id, features)

        label = positive_label if score >= 0.5 else normal_label
        return {"score": score, "label": label, "warnings": warnings}

    @staticmethod
    def _vectorize(model: LoadedModel, features: Dict[str, float], warnings: List[str]) -> np.ndarray:
        missing = [f for f in model.features if f not in features]
        if missing:
            warnings.append(f"missing features filled with 0: {missing}")
        row = [features.get(name, 0.0) for name in model.features]
        return np.array([row], dtype=np.float64)

    @staticmethod
    def _score_with_pyfunc(model: LoadedModel, features: Dict[str, float], warnings: List[str]) -> float:
        import pandas as pd

        vector = ModelRegistry._vectorize(model, features, warnings)
        frame = pd.DataFrame(vector, columns=model.features)
        result = model.pyfunc_model.predict(frame)
        return float(np.asarray(result).reshape(-1)[0])

    @staticmethod
    def _score_with_estimator(estimator: Any, vector: np.ndarray) -> float:
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(vector)[0]
            # Assume binary classifier with the positive class last.
            return float(proba[-1])
        if hasattr(estimator, "decision_function"):
            raw = float(estimator.decision_function(vector)[0])
            return float(1 / (1 + np.exp(-raw)))
        prediction = estimator.predict(vector)[0]
        return float(prediction)


registry = ModelRegistry()
