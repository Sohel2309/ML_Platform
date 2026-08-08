"""
db.py
-----
Request/response logging.

Defaults to a local SQLite file (zero setup, works out of the box on any
laptop). Set DATABASE_URL to a Postgres connection string to use Postgres
instead, e.g.:

    export DATABASE_URL="postgresql+psycopg2://mluser:mlpass@localhost:5432/ml_platform"

No code changes needed -- SQLAlchemy handles both.
"""
import os
import json
import datetime as dt
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SQLITE_PATH = os.path.join(ROOT, "logs", "requests.db")
# .as_posix() forces forward slashes even on Windows, which keeps the
# sqlite:/// URL unambiguous (backslashes are technically tolerated by
# SQLAlchemy's sqlite dialect, but not worth risking on every OS/version).
_default_sqlite_url = "sqlite:///" + Path(DEFAULT_SQLITE_PATH).as_posix()
DATABASE_URL = os.getenv("DATABASE_URL", _default_sqlite_url)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
if DATABASE_URL.startswith("sqlite"):
    # Ensure the directory for the sqlite file exists BEFORE the engine is
    # created, regardless of whether init_db()/lifespan has run yet (e.g.
    # under test clients that don't trigger startup events).
    os.makedirs(os.path.dirname(DEFAULT_SQLITE_PATH) or ".", exist_ok=True)
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(64), index=True)
    route_type = Column(String(16))       # "serving" or "shadow"
    model_name = Column(String(32))
    payload_json = Column(Text)
    prediction = Column(Integer)
    probability = Column(Float)
    latency_ms = Column(Float)
    created_at = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc))


def init_db():
    os.makedirs(os.path.dirname(DEFAULT_SQLITE_PATH), exist_ok=True)
    Base.metadata.create_all(bind=engine)


def log_prediction(request_id, route_type, model_name, payload, result, latency_ms):
    session = SessionLocal()
    try:
        entry = PredictionLog(
            request_id=request_id,
            route_type=route_type,
            model_name=model_name,
            payload_json=json.dumps(payload),
            prediction=result["prediction"],
            probability=result["probability"],
            latency_ms=latency_ms,
        )
        session.add(entry)
        session.commit()
    finally:
        session.close()
