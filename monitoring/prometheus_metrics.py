"""
prometheus_metrics.py
----------------------
Central place where every Prometheus metric used by the platform is defined.
Import `REGISTRY_METRICS` object attributes wherever you need to increment /
observe a metric. Exposed at GET /metrics via prometheus_client's
generate_latest().
"""
from prometheus_client import Counter, Histogram, Gauge

# --- Traffic / routing ---
REQUEST_COUNT = Counter(
    "ml_platform_requests_total",
    "Total number of prediction requests received",
    ["route", "model"],
)

ROUTED_MODEL_COUNT = Counter(
    "ml_platform_routed_model_total",
    "Number of requests routed to each model as the SERVING (production-facing) model",
    ["model"],
)

SHADOW_INVOCATIONS = Counter(
    "ml_platform_shadow_invocations_total",
    "Number of times the shadow model was invoked",
    ["model"],
)

# --- Latency ---
REQUEST_LATENCY = Histogram(
    "ml_platform_request_latency_seconds",
    "End-to-end latency of the /predict endpoint (serving path only)",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5),
)

MODEL_INFERENCE_LATENCY = Histogram(
    "ml_platform_model_inference_latency_seconds",
    "Latency of a single model's predict() call",
    ["model"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

# --- Model quality / comparison ---
PREDICTION_DISAGREEMENT = Counter(
    "ml_platform_prediction_disagreement_total",
    "Number of times production and shadow model disagreed on the predicted class",
)

MODEL_ERROR_COUNT = Counter(
    "ml_platform_model_errors_total",
    "Number of exceptions raised while running a model",
    ["model"],
)

# --- Drift ---
FEATURE_DRIFT_SCORE = Gauge(
    "ml_platform_feature_drift_ks_statistic",
    "KS-test statistic of a feature's live distribution vs training baseline",
    ["feature"],
)

DRIFT_ALERT = Gauge(
    "ml_platform_drift_alert",
    "1 if a feature has drifted beyond threshold, else 0",
    ["feature"],
)

# --- Rollout / canary state ---
CANARY_TRAFFIC_PERCENT = Gauge(
    "ml_platform_canary_traffic_percent",
    "Current percentage of traffic routed to the canary/shadow-as-serving model",
)

ROLLBACK_EVENTS = Counter(
    "ml_platform_rollback_events_total",
    "Number of times an automatic rollback was triggered",
)
