"""
shadow_runner.py
-----------------
Runs the shadow model on the SAME request payload as the serving model,
but its output is never returned to the client. It only feeds the
comparison engine and Prometheus metrics.

Designed to run as a FastAPI BackgroundTask so it never adds latency to
the user-facing response.
"""
import time
import logging

from serving.model_registry import ModelRegistry
from monitoring.prometheus_metrics import (
    SHADOW_INVOCATIONS, MODEL_INFERENCE_LATENCY, MODEL_ERROR_COUNT,
    PREDICTION_DISAGREEMENT,
)

logger = logging.getLogger("shadow_runner")


def run_shadow_prediction(
    registry: ModelRegistry,
    ab_engine,
    shadow_model_name: str,
    payload: dict,
    prod_result: dict,
    log_fn=None,
):
    """
    Executed in the background AFTER the response has already been sent
    to the user. Any exception here must never crash the app.
    """
    SHADOW_INVOCATIONS.labels(model=shadow_model_name).inc()
    start = time.perf_counter()
    try:
        shadow_result = registry.predict(shadow_model_name, payload)
    except Exception as exc:  # noqa: BLE001
        MODEL_ERROR_COUNT.labels(model=shadow_model_name).inc()
        logger.exception("Shadow model '%s' failed: %s", shadow_model_name, exc)
        return
    finally:
        MODEL_INFERENCE_LATENCY.labels(model=shadow_model_name).observe(time.perf_counter() - start)

    if prod_result["prediction"] != shadow_result["prediction"]:
        PREDICTION_DISAGREEMENT.inc()

    ab_engine.record(
        prod_proba=prod_result["probability"],
        shadow_proba=shadow_result["probability"],
        prod_pred=prod_result["prediction"],
        shadow_pred=shadow_result["prediction"],
    )

    if log_fn is not None:
        log_fn(shadow_model_name, payload, shadow_result)
