# ML Platform — Multi-Model Serving with Shadow Deployment

A production-style machine learning serving platform implementing **shadow deployment, canary releases with automatic rollback, statistical A/B model comparison, and Kolmogorov–Smirnov feature-drift detection** — containerized with Docker, deployed to Kubernetes, monitored with Prometheus/Grafana, and validated with Continuous Integration.

Built to demonstrate the operational side of ML engineering: how a new model candidate is safely evaluated against production traffic before it's ever trusted to serve a single real user.

[![CI](https://github.com/Sohel2309/ml-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/ml-platform/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Tests](https://img.shields.io/badge/tests-11%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)


---

## Why this project exists

Companies can't safely ship a new model straight to production traffic. **Shadow deployment** solves this: a candidate model runs silently on every real request, alongside the production model, without ever affecting what the user sees. Its predictions are logged and statistically compared against the production model's — so by the time it's promoted, its behavior is already understood, not guessed at.

This project implements that pattern end-to-end, plus the surrounding infrastructure a real ML platform needs: weighted canary rollout, automatic rollback on degraded health, drift detection, and full observability.

---

## Architecture

```
                              Client Request
                                    │
                                    ▼
                       ┌────────────────────────┐
                       │     FastAPI Gateway     │
                       │  routing · caching ·    │
                       │  canary · rollback       │
                       └───────────┬─────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                              │
              Serving traffic                Shadow traffic
             (prod or canary)                 (always 100%)
                    │                              │
                    ▼                              ▼
            ┌───────────────┐              ┌───────────────┐
            │    Model A     │              │    Model B     │
            │ Random Forest  │              │ Gradient Boost │
            │  (production)  │              │   (shadow)     │
            └───────┬────────┘              └───────┬────────┘
                    │                                │
                    │        Response to user        │
                    │◄───────────────────────────────┘
                    │        (only Model A's result)
                    ▼
            ┌────────────────────────────────────┐
            │        Comparison Engine            │
            │  disagreement rate · paired t-test  │
            │  Wilson confidence intervals        │
            └────────────────────────────────────┘

     ┌───────────┐    ┌───────────┐    ┌────────────┐    ┌───────────┐
     │  Celery   │◄──►│   Redis   │    │ Prometheus │───►│  Grafana  │
     │  worker   │    │  cache /  │    │  metrics    │    │ dashboard │
     │(async log)│    │  broker   │    │             │    │            │
     └───────────┘    └───────────┘    └────────────┘    └───────────┘

                       Kubernetes (Minikube)
              4× Gateway replicas · Redis · Celery worker
```

---

## Key features

### Multi-model serving & shadow deployment
- **Model A** (RandomForest) — production, serves the actual response
- **Model B** (GradientBoosting) — shadow, runs on every request in the background via `BackgroundTasks`, never adds latency to the user-facing response
- Model registry with versioned artifacts and metadata (`serving/model_registry.py`)

### Canary releases with automatic rollback
- Deterministic, hash-based weighted traffic split — `POST /admin/canary` sets what % of *real* traffic goes to the canary model
- A rolling window tracks the canary's error rate and P99 latency on every request
- If error rate exceeds 5% or P99 exceeds 500ms, traffic is automatically reset to 0% — no human intervention required (`gateway/router.py`)

### Statistical model comparison
- `GET /compare` — prediction disagreement rate with a **Wilson score confidence interval**, plus a **paired t-test** on predicted probabilities to check whether production and shadow models differ significantly
- All statistics are sanitized for non-finite values (`NaN`/`±Inf`) before JSON serialization — a real bug found and fixed during development when near-identical prediction arrays produced a `-inf` t-statistic

### Feature drift detection
- `GET /drift` — two-sample **Kolmogorov–Smirnov test** comparing live incoming feature distributions against the training-time baseline, per feature, with configurable significance thresholds

### Observability
- 10+ custom Prometheus metrics (`monitoring/prometheus_metrics.py`): request counts, latency histograms, disagreement counts, drift scores, rollback events, canary traffic gauge
- Pre-provisioned Grafana dashboard — no manual setup, panels for request rate, P99 latency, disagreement rate, drift scores, canary %, and rollback events

### Async processing & caching
- Redis-backed response caching, keyed on model + payload hash
- Celery worker offloads prediction-logging DB writes off the request path
- Both degrade gracefully to synchronous fallbacks if Redis/Celery aren't running — the API never hard-depends on them

---

## Deployment

### Docker Compose
```
┌─────────────────────────────┐
│       Docker Compose        │
├─────────────────────────────┤
│ FastAPI Gateway              │
│ Celery Worker                │
│ Redis                        │
│ Prometheus                   │
│ Grafana                      │
└─────────────────────────────┘
```
One command (`docker compose up --build`) brings up all five services. The gateway container trains its own models on first boot if none exist — no manual setup step required.

### Kubernetes (Minikube)
- Namespace-isolated deployment: 4 gateway replicas, Redis, Celery worker, ConfigMap, Service
- Verified: `4/4` gateway pods `Running`, `0` restarts, across gateway, Celery, and Redis
- Horizontal scaling demonstrated via `kubectl scale deployment ml-platform-gateway --replicas=4`

---

## Performance benchmark

Load tested with [`hey`](https://github.com/rakyll/hey) — 30 seconds, 10 concurrent clients, POST `/predict`, against the single-instance Docker Compose deployment:

| Metric | Result |
|---|---|
| Throughput | **101.27 RPS** |
| P50 latency | 82.5 ms |
| P95 latency | 159.0 ms |
| P99 latency | 382.6 ms |
| HTTP errors | 0 |
| Successful requests | 3,044 |

The Kubernetes deployment was independently load tested as well, both through the Minikube service tunnel and in-cluster (pod-to-pod, bypassing the tunnel). Both measured lower throughput than the Compose baseline — expected on a single-node Minikube cluster, where the app pods share CPU with the entire Kubernetes control plane (etcd, API server, kube-proxy, CoreDNS) rather than owning the machine outright. That result is reported here for transparency rather than used as a headline number, since a single-node local cluster isn't a fair throughput comparison against a dedicated container.

---

## Model performance

Both models are trained on a synthetic, Adult-Income-style dataset generated locally (`serving/train_models.py`) — no external data dependency, fully reproducible via a fixed random seed.

**Model A — Random Forest (production)**

| Metric | Score |
|---|---|
| Accuracy | 0.8580 |
| F1 | 0.6682 |
| ROC-AUC | 0.9127 |

**Model B — Gradient Boosting (shadow)**

| Metric | Score |
|---|---|
| Accuracy | 0.8663 |
| F1 | 0.7043 |
| ROC-AUC | 0.9182 |

Model B currently outperforms Model A on all three metrics — exactly the scenario shadow deployment exists for: comparing a stronger candidate against production on real traffic before promoting it.

---

## API reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness/readiness check |
| `/models` | GET | Registered model metadata |
| `/predict` | POST | Route request to serving model, run shadow in background |
| `/compare` | GET | A/B statistical comparison (prod vs. shadow) |
| `/drift` | GET | Per-feature KS-test drift report |
| `/metrics` | GET | Prometheus exposition format |
| `/admin/canary` | POST | Set canary traffic percentage (0–100) |
| `/admin/rollback-check` | GET | Force a rollback health check |

<details>
<summary><strong>Example request/response</strong></summary>

```json
POST /predict
{
  "age": 40,
  "education_num": 14,
  "hours_per_week": 50,
  "capital_gain": 0,
  "capital_loss": 0,
  "workclass": "Government",
  "marital_status": "Single",
  "occupation": "Prof-specialty"
}
```

```json
{
  "request_id": "b95ff814-73b6-4873-9c55-d952be5f37d7",
  "model_used": "model_a",
  "prediction": 0,
  "probability": 0.3933,
  "cache_hit": false,
  "latency_ms": 30.14,
  "canary_traffic_percent": 0.0,
  "rollback_check": { "rolled_back": false }
}
```
</details>

---

## Testing & Continuous Integration (CI)

```
11 passed in 5.22s
```

Automated tests cover health/model endpoints, valid and invalid predictions, unknown-category graceful fallback, the `/metrics` Prometheus format, `/compare` and `/drift` response shapes, and canary admin validation.

**GitHub Actions** (`.github/workflows/ci.yml`) runs on every push and pull request to `main`:

```
Push/PR → Install deps → Train models → pytest (quality gate) → Docker build
```

The build job only runs if all tests pass — a broken model or API change can't ship an image.

---

## Tech stack

| Layer | Technologies |
|---|---|
| **API** | Python, FastAPI, Uvicorn, Pydantic |
| **ML** | scikit-learn (RandomForest, GradientBoosting), MLflow, joblib |
| **Async / caching** | Redis, Celery |
| **Monitoring** | Prometheus, Grafana |
| **Deployment** | Docker, Docker Compose, Kubernetes (Minikube) |
| **Testing / CI** | Pytest, GitHub Actions, Docker Buildx, `hey` |

---

## Project structure

```
ml-platform/
├── gateway/                  FastAPI app, routing, caching, DB logging, Celery
│   ├── main.py
│   ├── router.py
│   ├── cache.py
│   ├── db.py
│   ├── celery_app.py
│   ├── tasks.py
│   └── shadow_runner.py
├── serving/                  Model training + registry
│   ├── train_models.py
│   └── model_registry.py
├── comparison/
│   └── ab_engine.py           A/B statistical comparison
├── monitoring/
│   ├── drift_detector.py       KS-test feature drift
│   └── prometheus_metrics.py
├── tests/
│   └── test_gateway.py         11 automated tests
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── entrypoint.sh
│   ├── prometheus.yml
│   ├── grafana/provisioning/
│   └── k8s/                    namespace, configmap, deployment, service,
│                                redis, celery-worker manifests
├── .github/workflows/ci.yml
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Running locally

### 1. Environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Train the models
```bash
python serving/train_models.py
```

### 3. Run tests
```bash
pytest tests/ -v
```

### 4. Start the API
```bash
uvicorn gateway.main:app --host 0.0.0.0 --port 8000
```
Interactive docs at **http://localhost:8000/docs**.

### Docker Compose (full stack)
```bash
cd deployment
docker compose up --build
```
| Service | URL |
|---|---|
| API | http://localhost:8000 |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |

### Kubernetes (Minikube)
```bash
minikube start
eval $(minikube docker-env)
docker build -f deployment/Dockerfile -t ml-platform-gateway:local .

kubectl apply -f deployment/k8s/namespace.yaml
kubectl apply -f deployment/k8s/configmap.yaml
kubectl apply -f deployment/k8s/redis.yaml
kubectl apply -f deployment/k8s/deployment.yaml
kubectl apply -f deployment/k8s/service.yaml
kubectl apply -f deployment/k8s/celery-worker.yaml

kubectl get pods -n ml-platform
kubectl scale deployment ml-platform-gateway -n ml-platform --replicas=4
```

---

## Engineering highlights

- Designed a multi-model serving architecture with production and shadow models evaluated on live traffic
- Implemented deterministic weighted canary routing with automatic rollback triggered by rolling error-rate and P99-latency thresholds
- Built a statistical A/B comparison engine (Wilson confidence intervals, paired t-tests) and a KS-test feature-drift detector
- Containerized the full stack (API, worker, cache, monitoring) with Docker Compose; deployed the platform to Kubernetes with a horizontally scaled API gateway (4 replicas), Redis, and a Celery worker
- Instrumented 10+ custom Prometheus metrics with a pre-provisioned Grafana dashboard
- Built a GitHub Actions CI pipeline with a pytest quality gate blocking image builds on test failure
- Load tested the platform with `hey` across three deployment configurations (Docker Compose, K8s via tunnel, K8s in-cluster), reporting measured results honestly rather than a cherry-picked figure
- 11/11 automated tests passing

---

## License

MIT — see [LICENSE](LICENSE).
