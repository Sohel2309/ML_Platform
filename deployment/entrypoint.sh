#!/bin/bash
set -e

if [ ! -f "/app/artifacts/registry.json" ]; then
    echo ">>> No trained models found. Training models now (first run only) ..."
    python serving/train_models.py
else
    echo ">>> Trained models found in /app/artifacts. Skipping training."
fi

if [ "$1" = "worker" ]; then
    echo ">>> Starting Celery worker ..."
    exec celery -A gateway.celery_app worker --loglevel=info --concurrency=2
else
    echo ">>> Starting API server ..."
    exec uvicorn gateway.main:app --host 0.0.0.0 --port 8000
fi
