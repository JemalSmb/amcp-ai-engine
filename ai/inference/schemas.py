"""Pydantic request/response schemas for the inference API."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

ModelStatus = Literal["healthy", "degraded", "offline"]


class ModelInfo(BaseModel):
    id: str
    name: str
    domain: str
    version: str
    status: ModelStatus
    trained: bool
    architecture: str
    source: str


class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(default_factory=dict)


class PredictResponse(BaseModel):
    model_id: str
    domain: str
    version: str
    label: str
    score: float
    trained: bool
    warnings: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    models_loaded: int
    models_total: int
    details: Dict[str, Any] = Field(default_factory=dict)
