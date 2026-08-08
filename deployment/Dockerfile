FROM python:3.12-slim

WORKDIR /app

# System deps needed by scipy/scikit-learn wheels build tools (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gateway/ ./gateway/
COPY serving/ ./serving/
COPY monitoring/ ./monitoring/
COPY comparison/ ./comparison/
COPY deployment/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    DATABASE_URL=sqlite:////app/logs/requests.db \
    REDIS_URL=redis://redis:6379/0

RUN mkdir -p /app/logs /app/artifacts /app/data /app/mlruns

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
