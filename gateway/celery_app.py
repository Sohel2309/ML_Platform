"""
celery_app.py
-------------
Celery application used to offload prediction-logging (DB writes) to a
background worker so the /predict endpoint never waits on I/O.

Start a worker with:
    celery -A gateway.celery_app worker --loglevel=info

If no worker is running, main.py falls back to writing logs synchronously
(see ASYNC_LOGGING env var), so the API still works for beginners who
haven't started Celery yet.
"""
import os
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "ml_platform",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["gateway.tasks"],
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    broker_connection_retry_on_startup=True,
)
