"""
tasks.py
--------
Celery tasks. Currently just one: persist a prediction log entry to the
database asynchronously so the API response is never blocked on a DB write.
"""
from gateway.celery_app import celery_app
from gateway.db import log_prediction as _log_prediction_sync


@celery_app.task(name="tasks.log_prediction_task")
def log_prediction_task(request_id, route_type, model_name, payload, result, latency_ms):
    _log_prediction_sync(request_id, route_type, model_name, payload, result, latency_ms)
    return {"status": "logged", "request_id": request_id}
