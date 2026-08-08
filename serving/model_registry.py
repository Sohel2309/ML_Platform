"""
model_registry.py
------------------
Loads trained models + encoders from artifacts/ and exposes a simple
in-memory registry that the gateway can query by name ("model_a" / "model_b").
"""
import os
import json
import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")


class ModelRegistry:
    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        registry_path = os.path.join(artifacts_dir, "registry.json")
        if not os.path.exists(registry_path):
            raise FileNotFoundError(
                f"registry.json not found at {registry_path}. "
                "Run `python serving/train_models.py` first."
            )
        with open(registry_path) as f:
            self.metadata = json.load(f)

        self.feature_columns = self.metadata["feature_columns"]
        self.categorical_columns = self.metadata["categorical_columns"]
        self.encoders = joblib.load(os.path.join(artifacts_dir, "encoders.joblib"))

        self.models = {}
        for name, info in self.metadata["models"].items():
            model_path = os.path.join(ROOT, info["path"])
            self.models[name] = joblib.load(model_path)

        baseline_path = os.path.join(artifacts_dir, "baseline_distribution.json")
        with open(baseline_path) as f:
            self.baseline_distribution = json.load(f)

    def get_model(self, name: str):
        if name not in self.models:
            raise KeyError(f"Model '{name}' not found. Available: {list(self.models.keys())}")
        return self.models[name]

    def get_model_info(self, name: str) -> dict:
        return self.metadata["models"][name]

    def list_models(self) -> dict:
        return self.metadata["models"]

    def prepare_features(self, payload: dict) -> pd.DataFrame:
        """Encode a raw request payload into the exact feature frame the models expect."""
        row = {col: payload[col] for col in self.feature_columns}
        df = pd.DataFrame([row])
        for col in self.categorical_columns:
            le = self.encoders[col]
            val = df.at[0, col]
            if val not in le.classes_:
                val = le.classes_[0]  # fallback for unseen category
                df.at[0, col] = val
            df[col] = le.transform(df[col])
        return df[self.feature_columns]

    def predict(self, name: str, payload: dict) -> dict:
        model = self.get_model(name)
        X = self.prepare_features(payload)
        pred = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][1])
        return {"prediction": pred, "probability": proba}


# Singleton instance used across the app
_registry_instance = None


def get_registry() -> ModelRegistry:
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance
