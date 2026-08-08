"""
cache.py
--------
Simple Redis-backed cache for /predict responses, keyed by a hash of the
request payload + which model served it. If Redis is not reachable (e.g.
you haven't started it yet), the app degrades gracefully to "no cache"
instead of crashing -- this is important for beginners running the project
for the first time without every service up.
"""
import os
import json
import hashlib
import logging

import redis

logger = logging.getLogger("cache")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "60"))

_client = None
_available = None  # tri-state: None=unknown, True/False cached after first check


def _get_client():
    global _client, _available
    if _client is None:
        _client = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=0.5, socket_timeout=0.5)
    return _client


def is_available() -> bool:
    global _available
    if _available is not None:
        return _available
    try:
        _get_client().ping()
        _available = True
    except Exception:  # noqa: BLE001
        logger.warning("Redis not reachable at %s -- caching disabled", REDIS_URL)
        _available = False
    return _available


def make_key(model_name: str, payload: dict) -> str:
    raw = json.dumps({"model": model_name, "payload": payload}, sort_keys=True)
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return f"predict:{model_name}:{digest}"


def get_cached(model_name: str, payload: dict):
    if not is_available():
        return None
    try:
        raw = _get_client().get(make_key(model_name, payload))
        return json.loads(raw) if raw else None
    except Exception:  # noqa: BLE001
        return None


def set_cached(model_name: str, payload: dict, result: dict):
    if not is_available():
        return
    try:
        _get_client().setex(make_key(model_name, payload), CACHE_TTL_SECONDS, json.dumps(result))
    except Exception:  # noqa: BLE001
        pass
