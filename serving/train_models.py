"""
train_models.py
----------------
Generates a synthetic "Adult Income"-style tabular dataset (same shape/columns
as the classic UCI Adult dataset, but generated locally so the project has
zero external data dependencies) and trains two classifiers:

  - Model A (production) : RandomForestClassifier
  - Model B (shadow)     : GradientBoostingClassifier

Both are logged to MLflow (local file-store, no server required) and saved
to disk under artifacts/ so the FastAPI gateway can load them directly.

Run:
    python serving/train_models.py
"""
import os
import json
import joblib
import numpy as np
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    "age", "education_num", "hours_per_week", "capital_gain",
    "capital_loss", "workclass", "marital_status", "occupation",
]
CATEGORICAL_COLUMNS = ["workclass", "marital_status", "occupation"]
NUMERIC_COLUMNS = ["age", "education_num", "hours_per_week", "capital_gain", "capital_loss"]

WORKCLASS = ["Private", "Self-emp", "Government", "Without-pay"]
MARITAL = ["Married", "Single", "Divorced", "Widowed"]
OCCUPATION = ["Tech", "Sales", "Exec-managerial", "Craft-repair", "Service", "Prof-specialty"]


def generate_dataset(n_samples: int = 20000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 75, size=n_samples)
    education_num = rng.integers(1, 16, size=n_samples)
    hours_per_week = rng.integers(5, 80, size=n_samples)
    capital_gain = rng.choice([0, 0, 0, 0, rng.integers(1000, 20000)], size=n_samples).astype(float)
    capital_gain = np.where(rng.random(n_samples) < 0.1, rng.integers(1000, 20000, size=n_samples), 0)
    capital_loss = np.where(rng.random(n_samples) < 0.05, rng.integers(500, 4000, size=n_samples), 0)
    workclass = rng.choice(WORKCLASS, size=n_samples, p=[0.6, 0.15, 0.2, 0.05])
    marital_status = rng.choice(MARITAL, size=n_samples, p=[0.45, 0.35, 0.15, 0.05])
    occupation = rng.choice(OCCUPATION, size=n_samples)

    # Construct a realistic-ish latent score that determines the label,
    # so the models actually have signal to learn (not pure noise).
    score = (
        0.04 * age
        + 0.35 * education_num
        + 0.02 * hours_per_week
        + 0.0004 * capital_gain
        - 0.0006 * capital_loss
        + np.where(marital_status == "Married", 1.2, 0.0)
        + np.where(np.isin(occupation, ["Tech", "Exec-managerial", "Prof-specialty"]), 1.0, 0.0)
        + rng.normal(0, 1.5, size=n_samples)  # noise
    )
    threshold = np.quantile(score, 0.75)  # ~25% positive class, like real Adult dataset
    income_above_50k = (score > threshold).astype(int)

    df = pd.DataFrame({
        "age": age,
        "education_num": education_num,
        "hours_per_week": hours_per_week,
        "capital_gain": capital_gain,
        "capital_loss": capital_loss,
        "workclass": workclass,
        "marital_status": marital_status,
        "occupation": occupation,
        "income_above_50k": income_above_50k,
    })
    return df


def encode_features(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    df = df.copy()
    if encoders is None:
        encoders = {}
    for col in CATEGORICAL_COLUMNS:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = df[col].map(lambda v: v if v in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
    return df, encoders


def main():
    print("[1/6] Generating synthetic Adult-Income-style dataset ...")
    df = generate_dataset()
    df.to_csv(os.path.join(DATA_DIR, "adult_income_synthetic.csv"), index=False)
    print(f"      -> {len(df)} rows saved to data/adult_income_synthetic.csv")
    print(f"      -> Positive class rate: {df['income_above_50k'].mean():.3f}")

    print("[2/6] Encoding categorical features ...")
    df_encoded, encoders = encode_features(df, fit=True)
    X = df_encoded[FEATURE_COLUMNS]
    y = df_encoded["income_above_50k"]

    # Save baseline feature distribution (used later by the drift detector)
    baseline_stats = {
        col: {"mean": float(X[col].mean()), "std": float(X[col].std())}
        for col in FEATURE_COLUMNS
    }
    baseline_sample = X.sample(n=min(2000, len(X)), random_state=42).to_dict(orient="list")
    with open(os.path.join(ARTIFACTS_DIR, "baseline_distribution.json"), "w") as f:
        json.dump({"stats": baseline_stats, "sample": baseline_sample}, f)
    print("      -> Baseline distribution saved for drift detection")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Path(...).as_uri() correctly builds a file:// URI on both Windows
    # (file:///C:/Users/...) and Mac/Linux (file:///home/...) -- a plain
    # os.path.join + f-string breaks on Windows because it leaves backslashes
    # in the URI, which MLflow rejects.
    mlruns_dir = Path(ROOT, "mlruns")
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(mlruns_dir.as_uri())
    mlflow.set_experiment("ml-platform-shadow-deployment")

    print("[3/6] Training Model A (RandomForest) -- PRODUCTION model ...")
    with mlflow.start_run(run_name="model_a_random_forest"):
        model_a = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
        model_a.fit(X_train, y_train)
        preds_a = model_a.predict(X_test)
        proba_a = model_a.predict_proba(X_test)[:, 1]
        metrics_a = {
            "accuracy": accuracy_score(y_test, preds_a),
            "f1": f1_score(y_test, preds_a),
            "roc_auc": roc_auc_score(y_test, proba_a),
        }
        mlflow.log_params({"n_estimators": 150, "max_depth": 10})
        mlflow.log_metrics(metrics_a)
        mlflow.sklearn.log_model(model_a, "model")
        print(f"      -> Model A metrics: {metrics_a}")

    print("[4/6] Training Model B (GradientBoosting) -- SHADOW model ...")
    with mlflow.start_run(run_name="model_b_gradient_boosting"):
        model_b = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42)
        model_b.fit(X_train, y_train)
        preds_b = model_b.predict(X_test)
        proba_b = model_b.predict_proba(X_test)[:, 1]
        metrics_b = {
            "accuracy": accuracy_score(y_test, preds_b),
            "f1": f1_score(y_test, preds_b),
            "roc_auc": roc_auc_score(y_test, proba_b),
        }
        mlflow.log_params({"n_estimators": 150, "max_depth": 3})
        mlflow.log_metrics(metrics_b)
        mlflow.sklearn.log_model(model_b, "model")
        print(f"      -> Model B metrics: {metrics_b}")

    print("[5/6] Saving models + encoders to artifacts/ ...")
    joblib.dump(model_a, os.path.join(ARTIFACTS_DIR, "model_a.joblib"))
    joblib.dump(model_b, os.path.join(ARTIFACTS_DIR, "model_b.joblib"))
    joblib.dump(encoders, os.path.join(ARTIFACTS_DIR, "encoders.joblib"))

    print("[6/6] Writing model registry metadata ...")
    registry = {
        "models": {
            "model_a": {
                "version": "1.0.0",
                "algorithm": "RandomForestClassifier",
                "role": "production",
                "path": "artifacts/model_a.joblib",
                "metrics": metrics_a,
            },
            "model_b": {
                "version": "1.0.0",
                "algorithm": "GradientBoostingClassifier",
                "role": "shadow",
                "path": "artifacts/model_b.joblib",
                "metrics": metrics_b,
            },
        },
        "feature_columns": FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
    }
    with open(os.path.join(ARTIFACTS_DIR, "registry.json"), "w") as f:
        json.dump(registry, f, indent=2)

    print("\nDONE. Artifacts written to:", ARTIFACTS_DIR)
    print(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
