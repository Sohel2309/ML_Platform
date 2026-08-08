"""
main.py
-------
FastAPI gateway: entry point for the whole ML platform.

Endpoints
---------
GET  /health              liveness/readiness probe
GET  /metrics             Prometheus exposition format
GET  /models               list registered models + metrics
POST /predict              route request to serving model, run shadow in bg
GET  /compare               A/B statistical comparison summary
GET  /drift                 feature drift report (KS test vs baseline)
POST /admin/canary          set canary traffic percentage (0-100)
GET  /admin/rollback-check  force a rollback health check right now
"""
import os
import time
import uuid
import logging

from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from serving.model_registry import get_registry
from monitoring.drift_detector import DriftDetector
from monitoring.prometheus_metrics import (
    REQUEST_COUNT, ROUTED_MODEL_COUNT, REQUEST_LATENCY,
    MODEL_INFERENCE_LATENCY, MODEL_ERROR_COUNT,
)
from comparison.ab_engine import ABComparisonEngine
from gateway.router import TrafficRouter, RolloutConfig
from gateway.shadow_runner import run_shadow_prediction
from gateway import db, cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gateway")

ASYNC_LOGGING = os.getenv("ASYNC_LOGGING", "false").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    logger.info("ML Platform gateway started. Models: %s", list(registry.list_models().keys()))
    yield


app = FastAPI(
    title="ML Platform - Multi-Model Serving with Shadow Deployment",
    version="1.0.0",
    lifespan=lifespan,
)

registry = get_registry()
router = TrafficRouter(RolloutConfig())
ab_engine = ABComparisonEngine()
drift_detector = DriftDetector(registry.baseline_distribution, registry.feature_columns[:5])


class PredictRequest(BaseModel):
    age: int = Field(..., ge=18, le=100)
    education_num: int = Field(..., ge=1, le=20)
    hours_per_week: int = Field(..., ge=1, le=100)
    capital_gain: float = Field(0, ge=0)
    capital_loss: float = Field(0, ge=0)
    workclass: str
    marital_status: str
    occupation: str
    request_id: str | None = None

    @field_validator("workclass", "marital_status", "occupation")
    @classmethod
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("must not be empty")
        return v


def _log(route_type, model_name, payload, result, latency_ms, request_id):
    if ASYNC_LOGGING:
        try:
            from gateway.tasks import log_prediction_task
            log_prediction_task.delay(request_id, route_type, model_name, payload, result, latency_ms)
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("Celery unavailable, falling back to sync logging: %s", exc)
    db.log_prediction(request_id, route_type, model_name, payload, result, latency_ms)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(registry.list_models().keys())}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/models")
def list_models():
    return registry.list_models()


@app.post("/predict")
def predict(req: PredictRequest, background_tasks: BackgroundTasks):
    request_id = req.request_id or str(uuid.uuid4())
    payload = req.model_dump(exclude={"request_id"})

    serving_model_name = router.route(request_id)
    REQUEST_COUNT.labels(route="predict", model=serving_model_name).inc()
    ROUTED_MODEL_COUNT.labels(model=serving_model_name).inc()

    cached = cache.get_cached(serving_model_name, payload)
    start = time.perf_counter()
    if cached is not None:
        result = cached
        cache_hit = True
    else:
        model_start = time.perf_counter()
        try:
            result = registry.predict(serving_model_name, payload)
        except Exception as exc:  # noqa: BLE001
            MODEL_ERROR_COUNT.labels(model=serving_model_name).inc()
            router.record_canary_result(success=False, latency_seconds=time.perf_counter() - model_start)
            logger.exception("Prediction failed: %s", exc)
            raise HTTPException(status_code=500, detail="Model inference failed") from exc
        MODEL_INFERENCE_LATENCY.labels(model=serving_model_name).observe(time.perf_counter() - model_start)
        cache.set_cached(serving_model_name, payload, result)
        cache_hit = False

    latency_seconds = time.perf_counter() - start
    REQUEST_LATENCY.observe(latency_seconds)
    router.record_canary_result(success=True, latency_seconds=latency_seconds)

    background_tasks.add_task(
        _log, "serving", serving_model_name, payload, result, latency_seconds * 1000, request_id
    )

    drift_detector.record(payload)

    if router.config.shadow_enabled:
        background_tasks.add_task(
            run_shadow_prediction,
            registry, ab_engine, router.config.shadow_model, payload, result,
            lambda model_name, p, r: _log("shadow", model_name, p, r, 0.0, request_id),
        )

    rollback_status = router.maybe_rollback()

    return {
        "request_id": request_id,
        "model_used": serving_model_name,
        "prediction": result["prediction"],
        "probability": round(result["probability"], 4),
        "cache_hit": cache_hit,
        "latency_ms": round(latency_seconds * 1000, 2),
        "canary_traffic_percent": router.config.canary_traffic_percent,
        "rollback_check": rollback_status,
    }


@app.get("/compare")
def compare():
    return ab_engine.summary()


@app.get("/drift")
def drift():
    return drift_detector.check_drift()


class CanaryUpdate(BaseModel):
    percent: float = Field(..., ge=0, le=100)


@app.post("/admin/canary")
def set_canary(update: CanaryUpdate):
    router.set_canary_traffic(update.percent)
    return {"canary_traffic_percent": router.config.canary_traffic_percent}


@app.get("/admin/rollback-check")
def rollback_check():
    return router.maybe_rollback()
