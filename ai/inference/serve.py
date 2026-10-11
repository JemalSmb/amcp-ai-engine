"""
AI Inference Service


Lightweight FastAPI entrypoint that serves the platform's trained models
behind a standard request/response contract.

Run locally:
    uvicorn serve:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health
    GET  /models
    POST /models/{model_id}/predict
    POST /models/{model_id}/reload   -- force an immediate registry re-check

Models registered in MLflow are polled periodically; a new version is
hot-swapped in without restarting the process, and a
`platform.models.updated` event is published on NATS (if reachable) so
other services/dashboards can react.
"""

import asyncio
import json
import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from registry import registry
from schemas import HealthResponse, ModelInfo, PredictRequest, PredictResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_POLL_INTERVAL_SECONDS = float(os.environ.get("MODEL_POLL_INTERVAL_SECONDS", "20"))
NATS_URL = os.environ.get("NATS_URL", "nats://localhost:4222")
MODEL_UPDATE_SUBJECT = "platform.models.updated"

app = FastAPI(title="AMCP Inference Service", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_nats_client = None
_poll_task: asyncio.Task | None = None


async def _connect_nats() -> None:
    global _nats_client
    try:
        import nats
    except ImportError:
        logger.warning("nats-py not installed; model-update events will only be logged")
        return
    try:
        _nats_client = await nats.connect(NATS_URL, connect_timeout=2, max_reconnect_attempts=1)
        logger.info("connected to NATS at %s", NATS_URL)
    except Exception as exc:  # noqa: BLE001 - degrade to log-only mode
        logger.warning("could not connect to NATS at %s: %s", NATS_URL, exc)
        _nats_client = None


async def _publish_model_updates(changes: list[dict]) -> None:
    if not changes:
        return
    payload = json.dumps({"updates": changes}).encode()
    if _nats_client is not None:
        await _nats_client.publish(MODEL_UPDATE_SUBJECT, payload)
        logger.info("published model updates to %s: %s", MODEL_UPDATE_SUBJECT, changes)
    else:
        logger.info("(no NATS connection) model updates: %s", changes)


async def _poll_loop() -> None:
    while True:
        await asyncio.sleep(MODEL_POLL_INTERVAL_SECONDS)
        try:
            changes = registry.check_for_updates()
            await _publish_model_updates(changes)
        except Exception:  # noqa: BLE001 - polling must never crash the service
            logger.exception("model update poll failed")


@app.on_event("startup")
async def startup() -> None:
    global _poll_task
    registry.load_all()
    await _connect_nats()
    _poll_task = asyncio.create_task(_poll_loop())


@app.on_event("shutdown")
async def shutdown() -> None:
    if _poll_task is not None:
        _poll_task.cancel()
    if _nats_client is not None:
        await _nats_client.close()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    models = registry.list_models()
    loaded = sum(1 for m in models if m.trained)
    status = "ok" if models else "degraded"
    details = {m.id: m.source for m in models}
    details["_image_version"] = os.environ.get("MODEL_VERSION", "dev")
    return HealthResponse(
        status=status,
        models_loaded=loaded,
        models_total=len(models),
        details=details,
    )


@app.get("/models", response_model=list[ModelInfo])
def list_models() -> list[ModelInfo]:
    result = []
    for m in registry.list_models():
        result.append(
            ModelInfo(
                id=m.id,
                name=m.name,
                domain=m.domain,
                version=m.version,
                status="healthy" if m.trained else "degraded",
                trained=m.trained,
                architecture=m.architecture,
                source=m.source,
            )
        )
    return result


@app.post("/models/{model_id}/predict", response_model=PredictResponse)
def predict(model_id: str, request: PredictRequest) -> PredictResponse:
    model = registry.get(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"unknown model '{model_id}'")

    if not isinstance(request.features, dict):
        raise HTTPException(status_code=422, detail="'features' must be an object of name -> number")

    result = registry.predict(model_id, request.features)

    return PredictResponse(
        model_id=model.id,
        domain=model.domain,
        version=model.version,
        label=result["label"],
        score=result["score"],
        trained=model.trained,
        warnings=result["warnings"],
    )


@app.post("/models/{model_id}/reload")
async def reload_model(model_id: str) -> dict:
    if registry.get(model_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown model '{model_id}'")
    changes = registry.check_for_updates()
    await _publish_model_updates(changes)
    model = registry.get(model_id)
    return {
        "model_id": model_id,
        "version": model.version,
        "source": model.source,
        "changed": any(c["model_id"] == model_id for c in changes),
    }
