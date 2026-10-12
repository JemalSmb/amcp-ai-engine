"""Unit tests for the model registry: scoring, artifact resolution order, fallback."""

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import registry as reg
from registry import LoadedModel, ModelRegistry, SUPPORTED_MODELS, _deterministic_score


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """Registry with MLflow, Supabase and local artifacts all unavailable."""
    monkeypatch.setattr(reg, "_latest_registered_version", lambda name: None)
    monkeypatch.setattr(reg, "SUPABASE_URL", "")
    monkeypatch.setattr(reg, "SUPABASE_SERVICE_KEY", "")
    monkeypatch.setattr(reg, "MODEL_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(reg, "MODELS_ROOT", tmp_path / "models")
    return ModelRegistry()


def _base(features=("a", "b")):
    return LoadedModel(id="phishing_detection", domain="cybersecurity", name="n", version="0.0.0",
                       architecture="x", features=list(features), trained=False)


def _artifact(path, n_features=2, wrap=True):
    X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]] * 5, dtype=float)[:, :n_features]
    y = np.array([0, 1, 1, 0] * 5)
    clf = LogisticRegression().fit(X, y)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "scaler": None} if wrap else clf, path)


# --- deterministic mock scoring -------------------------------------------
def test_deterministic_score_is_stable_and_bounded():
    f = {"a": 1.0, "b": 2.5}
    assert _deterministic_score("m", f) == _deterministic_score("m", dict(reversed(list(f.items()))))
    assert 0.0 <= _deterministic_score("m", f) <= 1.0


def test_deterministic_score_depends_on_model_and_input():
    assert _deterministic_score("m1", {"a": 1}) != _deterministic_score("m2", {"a": 1})
    assert _deterministic_score("m", {"a": 1}) != _deterministic_score("m", {"a": 2})


# --- load_all / fallback ----------------------------------------------------
def test_load_all_registers_every_supported_model(isolated):
    isolated.load_all()
    assert {m.id for m in isolated.list_models()} == set(SUPPORTED_MODELS)


def test_unavailable_artifact_falls_back_to_mock(isolated):
    isolated.load_all()
    m = isolated.get("phishing_detection")
    assert not m.trained and m.source == "mock" and m.load_error


def test_predict_mock_returns_label_score_warning(isolated):
    isolated.load_all()
    out = isolated.predict("phishing_detection", {"x": 1.0})
    assert set(out) == {"score", "label", "warnings"}
    assert out["label"] in ("legitimate", "phishing")
    assert any("mock scoring" in w for w in out["warnings"])


def test_predict_label_follows_half_threshold(isolated, monkeypatch):
    isolated.load_all()
    monkeypatch.setattr(reg, "_deterministic_score", lambda *_: 0.5)
    assert isolated.predict("finance_fraud_anomaly", {})["label"] == "fraud_suspected"
    monkeypatch.setattr(reg, "_deterministic_score", lambda *_: 0.49)
    assert isolated.predict("finance_fraud_anomaly", {})["label"] == "normal"


def test_predict_unknown_model_raises_keyerror(isolated):
    with pytest.raises(KeyError):
        isolated.predict("nope", {})


def test_repeated_predictions_are_identical(isolated):
    isolated.load_all()
    results = [isolated.predict("anomaly_detection", {"a": 3.0}) for _ in range(50)]
    assert all(r == results[0] for r in results)


# --- local file ---------------------------------------------------------------
def test_local_file_loaded_and_used(isolated):
    _artifact(reg.MODELS_ROOT / "phishing_detection" / "artifacts/p.joblib")
    out = isolated._try_load_from_file("phishing_detection", "artifacts/p.joblib", _base())
    assert out.trained and out.source == "local_file"
    isolated._models["phishing_detection"] = out
    res = isolated.predict("phishing_detection", {"a": 1.0, "b": 1.0})
    assert 0.0 <= res["score"] <= 1.0 and res["warnings"] == []


def test_missing_features_filled_with_zero_and_warned(isolated):
    _artifact(reg.MODELS_ROOT / "phishing_detection" / "artifacts/p.joblib")
    m = isolated._try_load_from_file("phishing_detection", "artifacts/p.joblib", _base())
    isolated._models["phishing_detection"] = m
    res = isolated.predict("phishing_detection", {"a": 1.0})
    assert any("missing features" in w for w in res["warnings"])


def test_feature_count_mismatch_rejected(isolated):
    _artifact(reg.MODELS_ROOT / "phishing_detection" / "artifacts/p.joblib", n_features=2)
    base = _base(features=("a", "b", "c"))
    assert isolated._try_load_from_file("phishing_detection", "artifacts/p.joblib", base) is None
    assert "mismatch" in base.load_error


def test_corrupt_artifact_degrades_not_crashes(isolated):
    p = reg.MODELS_ROOT / "phishing_detection" / "artifacts/p.joblib"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"not a pickle")
    base = _base()
    assert isolated._try_load_from_file("phishing_detection", "artifacts/p.joblib", base) is None
    assert base.load_error


# --- Supabase Storage ---------------------------------------------------------
def test_supabase_skipped_when_unconfigured(isolated):
    assert isolated._try_load_from_supabase("phishing_detection", "artifacts/p.joblib", _base()) is None


def test_supabase_download_cached_and_served(isolated, monkeypatch, tmp_path):
    monkeypatch.setattr(reg, "SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setattr(reg, "SUPABASE_SERVICE_KEY", "key")
    src = tmp_path / "src.joblib"
    _artifact(src)
    calls = []

    class Resp:
        content = src.read_bytes()
        def raise_for_status(self): pass

    def fake_get(url, headers, timeout):
        calls.append((url, headers))
        return Resp()

    import requests
    monkeypatch.setattr(requests, "get", fake_get)
    out = isolated._try_load_from_supabase("phishing_detection", "artifacts/p.joblib", _base())
    assert out.trained
    assert calls[0][0] == "https://x.supabase.co/storage/v1/object/models/phishing_detection/artifacts/p.joblib"
    assert calls[0][1] == {"Authorization": "Bearer key"}
    # second load uses the cache: no new download
    isolated._try_load_from_supabase("phishing_detection", "artifacts/p.joblib", _base())
    assert len(calls) == 1


def test_supabase_failure_falls_through(isolated, monkeypatch):
    monkeypatch.setattr(reg, "SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setattr(reg, "SUPABASE_SERVICE_KEY", "key")
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "get", boom)
    assert isolated._try_load_from_supabase("phishing_detection", "artifacts/p.joblib", _base()) is None


# --- resolution order & hot reload ------------------------------------------
def test_resolution_order_registry_first(isolated, monkeypatch):
    order = []
    monkeypatch.setattr(isolated, "_try_load_from_registry", lambda *a: order.append("mlflow") or None)
    monkeypatch.setattr(isolated, "_try_load_from_supabase", lambda *a: order.append("supabase") or None)
    monkeypatch.setattr(isolated, "_try_load_from_file", lambda *a, **k: order.append("file") or None)
    m = isolated._load_one("phishing_detection", SUPPORTED_MODELS["phishing_detection"])
    assert order == ["mlflow", "supabase", "file"]
    assert m.source == "mock"


def test_registry_hit_short_circuits(isolated, monkeypatch):
    def hit(model_id, name, base):
        base.trained, base.source, base.version = True, "registry", "3"
        return base

    monkeypatch.setattr(isolated, "_try_load_from_registry", hit)
    monkeypatch.setattr(isolated, "_try_load_from_file", lambda *a, **k: pytest.fail("should not be reached"))
    assert isolated._load_one("phishing_detection", SUPPORTED_MODELS["phishing_detection"]).version == "3"


def test_check_for_updates_noop_when_registry_empty(isolated):
    isolated.load_all()
    assert isolated.check_for_updates() == []


def test_check_for_updates_hot_swaps_new_version(isolated, monkeypatch):
    isolated.load_all()
    monkeypatch.setattr(reg, "_latest_registered_version", lambda n: "2")

    def hit(model_id, name, base):
        base.trained, base.source, base.version = True, "registry", "2"
        return base

    monkeypatch.setattr(isolated, "_try_load_from_registry", hit)
    assert isolated.check_for_updates() == [{"model_id": "phishing_detection", "version": "2"}]
    assert isolated.get("phishing_detection").version == "2"
    assert isolated.check_for_updates() == []  # already current


# --- estimator scoring branches ----------------------------------------------
class _DF:
    def decision_function(self, v): return np.array([0.0])

class _P:
    def predict(self, v): return np.array([1])

def test_score_with_decision_function_is_sigmoid():
    assert ModelRegistry._score_with_estimator(_DF(), np.zeros((1, 1))) == pytest.approx(0.5)

def test_score_with_plain_predict():
    assert ModelRegistry._score_with_estimator(_P(), np.zeros((1, 1))) == 1.0


# --- Isolation Forest scoring ------------------------------------------------
def _iforest_artifact(path, calibrated=True):
    from sklearn.ensemble import IsolationForest

    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(500, 2))
    forest = IsolationForest(n_estimators=50, random_state=0).fit(X)
    raw = forest.decision_function(X)
    payload = {"model": forest, "scaler": None, "threshold": 0.6}
    if calibrated:
        payload.update(score_min=float(raw.min()), score_max=float(raw.max()))
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def test_isolation_forest_scores_outliers_higher_than_inliers(isolated, tmp_path):
    _iforest_artifact(tmp_path / "models" / "anomaly_detection" / "artifacts" / "a.joblib")
    base = _base()
    m = isolated._try_load_from_file("anomaly_detection", "artifacts/a.joblib", base)
    assert m.trained and m.threshold == 0.6 and m.score_range is not None
    isolated._models["anomaly_detection"] = m
    inlier = isolated.predict("anomaly_detection", {"a": 0.0, "b": 0.0})
    outlier = isolated.predict("anomaly_detection", {"a": 8.0, "b": 8.0})
    assert 0.0 <= inlier["score"] <= 1.0 and 0.0 <= outlier["score"] <= 1.0
    assert outlier["score"] > inlier["score"]
    assert outlier["label"] == "anomalous" and inlier["label"] == "normal"


def test_isolation_forest_without_calibration_is_not_inverted(isolated, tmp_path):
    _iforest_artifact(tmp_path / "models" / "anomaly_detection" / "artifacts" / "a.joblib", calibrated=False)
    m = isolated._try_load_from_file("anomaly_detection", "artifacts/a.joblib", _base())
    assert m.trained and m.score_range is None
    isolated._models["anomaly_detection"] = m
    inlier = isolated.predict("anomaly_detection", {"a": 0.0, "b": 0.0})["score"]
    outlier = isolated.predict("anomaly_detection", {"a": 8.0, "b": 8.0})["score"]
    assert outlier > inlier


def test_artifact_threshold_decides_the_label(isolated, tmp_path):
    _iforest_artifact(tmp_path / "models" / "anomaly_detection" / "artifacts" / "a.joblib")
    m = isolated._try_load_from_file("anomaly_detection", "artifacts/a.joblib", _base())
    m.threshold = 0.0  # everything at or above 0 is flagged
    isolated._models["anomaly_detection"] = m
    assert isolated.predict("anomaly_detection", {"a": 0.0, "b": 0.0})["label"] == "anomalous"
