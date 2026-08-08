"""
router.py
---------
Deterministic, weighted traffic router.

- SERVING_MODEL is the model that actually answers user requests today.
- CANARY_MODEL (optional) is a new model getting a small % of real traffic.
- The SHADOW_MODEL always runs in parallel on 100% of traffic, but its
  result is never returned to the user (see shadow_runner.py).

Routing is deterministic per-request (hash of a request id) so repeated
retries of the "same" request land on the same bucket -- this avoids flaky
A/B assignment.

Auto-rollback: if the canary's rolling error rate or latency crosses a
threshold, `maybe_rollback()` resets canary traffic to 0% and the caller
is notified via the returned dict.
"""
import hashlib
import os
from dataclasses import dataclass, field
from collections import deque

from monitoring.prometheus_metrics import ROLLBACK_EVENTS, CANARY_TRAFFIC_PERCENT


@dataclass
class RolloutConfig:
    serving_model: str = "model_a"
    canary_model: str = "model_b"
    shadow_model: str = "model_b"
    canary_traffic_percent: float = float(os.getenv("CANARY_TRAFFIC_PERCENT", "0"))
    shadow_enabled: bool = os.getenv("SHADOW_ENABLED", "true").lower() == "true"
    max_error_rate: float = 0.05      # rollback if canary error rate > 5%
    max_latency_seconds: float = 0.5  # rollback if canary p99 latency > 500ms


class TrafficRouter:
    def __init__(self, config: RolloutConfig = None):
        self.config = config or RolloutConfig()
        self._canary_errors = deque(maxlen=200)
        self._canary_latencies = deque(maxlen=200)
        CANARY_TRAFFIC_PERCENT.set(self.config.canary_traffic_percent)

    def route(self, request_id: str) -> str:
        """Return which model name should SERVE this request (prod or canary)."""
        if self.config.canary_traffic_percent <= 0:
            return self.config.serving_model

        bucket = int(hashlib.sha256(request_id.encode()).hexdigest(), 16) % 100
        if bucket < self.config.canary_traffic_percent:
            return self.config.canary_model
        return self.config.serving_model

    def record_canary_result(self, success: bool, latency_seconds: float):
        self._canary_errors.append(0 if success else 1)
        self._canary_latencies.append(latency_seconds)

    def maybe_rollback(self) -> dict:
        """Check rolling canary health; roll back to 0% canary traffic if unhealthy."""
        if self.config.canary_traffic_percent <= 0 or len(self._canary_errors) < 20:
            return {"rolled_back": False}

        error_rate = sum(self._canary_errors) / len(self._canary_errors)
        sorted_lat = sorted(self._canary_latencies)
        p99_index = max(0, int(len(sorted_lat) * 0.99) - 1)
        p99_latency = sorted_lat[p99_index] if sorted_lat else 0

        if error_rate > self.config.max_error_rate or p99_latency > self.config.max_latency_seconds:
            previous = self.config.canary_traffic_percent
            self.config.canary_traffic_percent = 0
            CANARY_TRAFFIC_PERCENT.set(0)
            ROLLBACK_EVENTS.inc()
            return {
                "rolled_back": True,
                "reason": "error_rate" if error_rate > self.config.max_error_rate else "latency",
                "error_rate": round(error_rate, 4),
                "p99_latency_seconds": round(p99_latency, 4),
                "previous_canary_percent": previous,
            }
        return {"rolled_back": False, "error_rate": round(error_rate, 4), "p99_latency_seconds": round(p99_latency, 4)}

    def set_canary_traffic(self, percent: float):
        percent = max(0.0, min(100.0, percent))
        self.config.canary_traffic_percent = percent
        CANARY_TRAFFIC_PERCENT.set(percent)
        self._canary_errors.clear()
        self._canary_latencies.clear()
