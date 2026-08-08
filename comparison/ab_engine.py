"""
ab_engine.py
------------
Statistical comparison between the production (serving) model and the
shadow model, based on the predictions collected so far.

Since ground-truth labels aren't available in real time in production,
we compare the two models on:
  1. Disagreement rate (% of requests where predicted class differs),
     with a Wilson score confidence interval.
  2. Mean absolute difference in predicted probability, with a paired
     t-test to check whether the difference is statistically significant.

This gives an early-warning signal: if the shadow model's behaviour is
statistically indistinguishable from the production model, it's a safer
candidate for a canary rollout.
"""
import math
from collections import deque
from typing import Deque, Tuple
from scipy import stats

WINDOW_SIZE = 1000


class ABComparisonEngine:
    def __init__(self, window_size: int = WINDOW_SIZE):
        self.prod_probs: Deque[float] = deque(maxlen=window_size)
        self.shadow_probs: Deque[float] = deque(maxlen=window_size)
        self.disagreements: Deque[int] = deque(maxlen=window_size)

    def record(self, prod_proba: float, shadow_proba: float, prod_pred: int, shadow_pred: int):
        self.prod_probs.append(prod_proba)
        self.shadow_probs.append(shadow_proba)
        self.disagreements.append(int(prod_pred != shadow_pred))

    @staticmethod
    def _wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
        if n == 0:
            return (0.0, 0.0)
        phat = successes / n
        denom = 1 + z**2 / n
        centre = phat + z**2 / (2 * n)
        adj = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)
        lower = (centre - adj) / denom
        upper = (centre + adj) / denom
        return (max(0.0, lower), min(1.0, upper))

    def summary(self) -> dict:
        n = len(self.disagreements)
        if n == 0:
            return {"sample_size": 0, "message": "No comparison data yet"}

        disagreement_count = sum(self.disagreements)
        disagreement_rate = disagreement_count / n
        ci_low, ci_high = self._wilson_ci(disagreement_count, n)

        
        result = {
            "sample_size": n,
            "disagreement_rate": round(disagreement_rate, 4),
            "disagreement_ci_95": [round(ci_low, 4), round(ci_high, 4)],
        }

        if n >= 2:
            prod = list(self.prod_probs)
            shadow = list(self.shadow_probs)
            t_stat, p_value = stats.ttest_rel(prod, shadow)
            mean_abs_diff = sum(abs(p - s) for p, s in zip(prod, shadow)) / n
            t_stat = float(t_stat)
            p_value = float(p_value)

            result.update({
                "mean_abs_probability_diff": round(mean_abs_diff, 4),
                "paired_ttest_statistic": round(t_stat, 4) if math.isfinite(t_stat) else None,
                "paired_ttest_p_value": round(p_value, 6) if math.isfinite(p_value) else None,
                "statistically_significant_difference": (
                    p_value < 0.05 if math.isfinite(p_value) else False
                ),
            })
        return result
