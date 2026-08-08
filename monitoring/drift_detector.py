"""
drift_detector.py
------------------
Compares live incoming numeric feature values against the training-time
baseline distribution using the two-sample Kolmogorov-Smirnov (KS) test.

A KS statistic close to 0 means the two samples come from the same
distribution (no drift). We flag drift when the statistic exceeds
DRIFT_THRESHOLD AND the p-value is below 0.05 (statistically significant).
"""
from collections import deque
from typing import Dict
from scipy.stats import ks_2samp

from monitoring.prometheus_metrics import FEATURE_DRIFT_SCORE, DRIFT_ALERT

DRIFT_THRESHOLD = 0.15   # KS statistic above this = meaningful drift
MIN_SAMPLES = 30         # need at least this many live samples before testing
WINDOW_SIZE = 500        # rolling window of recent requests kept in memory


class DriftDetector:
    def __init__(self, baseline_distribution: dict, numeric_columns):
        self.baseline_sample = baseline_distribution["sample"]
        self.numeric_columns = numeric_columns
        self.live_windows: Dict[str, deque] = {
            col: deque(maxlen=WINDOW_SIZE) for col in numeric_columns
        }

    def record(self, payload: dict):
        for col in self.numeric_columns:
            if col in payload:
                self.live_windows[col].append(float(payload[col]))

    def check_drift(self) -> dict:
        results = {}
        for col in self.numeric_columns:
            live_values = list(self.live_windows[col])
            baseline_values = self.baseline_sample.get(col, [])
            if len(live_values) < MIN_SAMPLES or len(baseline_values) < MIN_SAMPLES:
                results[col] = {"ks_statistic": None, "p_value": None, "drifted": False,
                                 "reason": "insufficient_samples"}
                continue
            stat, p_value = ks_2samp(baseline_values, live_values)
            drifted = bool(stat > DRIFT_THRESHOLD and p_value < 0.05)
            results[col] = {
                "ks_statistic": round(float(stat), 4),
                "p_value": round(float(p_value), 6),
                "drifted": drifted,
            }
            FEATURE_DRIFT_SCORE.labels(feature=col).set(stat)
            DRIFT_ALERT.labels(feature=col).set(1 if drifted else 0)
        return results
