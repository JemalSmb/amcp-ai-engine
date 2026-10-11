# AI Inference Service

Serves the platform's trained models behind a standard HTTP contract.

## Endpoints

- `GET /health` — service + per-model status
- `GET /models` — registered models, their domain, version, source (`registry` | `local_file` | `mock`), and whether they're trained
- `POST /models/{model_id}/predict` — score a `{"features": {"name": value, ...}}` payload
- `POST /models/{model_id}/reload` — force an immediate re-check against the MLflow registry instead of waiting for the next poll

## Model sources, in priority order

1. **MLflow Model Registry** (`registered_name` in `registry.py`'s `SUPPORTED_MODELS`) — the preferred source. A background task polls every `MODEL_POLL_INTERVAL_SECONDS` (default 20s) for a newer version and hot-swaps it in without restarting the process. On a swap, a `platform.models.updated` event is published on NATS (if reachable; otherwise just logged) so dashboards/services can react.
2. **Static local file** (`artifact` in `SUPPORTED_MODELS`) — used if there's no registered model, or the registry is unreachable.
3. **Deterministic mock scoring** — used if neither of the above produced a usable model. The same input always produces the same output, so the endpoint contract is real and testable, but the score has no predictive value. `PredictResponse.trained`/`.warnings` and `ModelInfo.source` tell you which case you're in.

## Supported models (MVP set)

| model_id                      | domain         | source                                      |
|--------------------------------|----------------|-----------------------------------------------|
| `anomaly_detection`            | cybersecurity  | mock                                          |
| `phishing_detection`           | cybersecurity  | MLflow registry (`phishing_classifier`), local `.joblib` fallback |
| `finance_fraud_anomaly`        | finance        | mock                                          |
| `manufacturing_process_anomaly`| manufacturing  | mock                                          |

## Registering a new model version

Training pipelines register models via `log_and_register_model()` in
`ai/training/utils.py` (see `ai/training/pipelines/phishing_detection.py`
for the pattern) — it bundles a fitted scaler + estimator as one pyfunc
model and registers it under a name this service polls for. Run the
training pipeline again with `--mlflow-uri` pointed at the same tracking
store this service uses (`MLFLOW_TRACKING_URI` env var) and the running
service will pick up the new version within one poll interval, no
redeploy needed.

### Environment variables

- `MLFLOW_TRACKING_URI` — defaults to a local sqlite file under `../training/pipelines/mlflow.db`; point this at the docker-compose `mlflow` service in a real deployment.
- `MODEL_POLL_INTERVAL_SECONDS` — default `20`.
- `NATS_URL` — default `nats://localhost:4222`; the service degrades to log-only if unreachable.

## Run locally

```bash
cd amcp-ai-engine/ai/inference
pip install -r requirements.txt
uvicorn serve:app --host 0.0.0.0 --port 8000 --reload
```

## Run in Docker

Build context is `amcp-ai-engine/ai` (so both `inference/` and `models/` are available):

```bash
cd amcp-ai-engine/ai
docker build -f inference/Dockerfile -t amcp-inference .
docker run -p 8000:8000 amcp-inference
```
