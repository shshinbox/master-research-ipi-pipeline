import numpy as np
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
)
from typing import Tuple


class ThresholdOptimizer:

    @staticmethod
    def find_optimal_threshold(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        metric: str = "f1",
        threshold_range: Tuple[float, float] = (0.1, 0.9),
        num_thresholds: int = 81,
    ) -> Tuple[float, float]:

        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
        best_threshold = 0.5
        best_score = 0.0

        for threshold in thresholds:
            y_pred = (y_probs >= threshold).astype(int)

            if metric == "f1":
                score = f1_score(y_true, y_pred)
            elif metric == "youden":
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = sensitivity + specificity - 1
            elif metric == "cost_sensitive":
                # FP 비용 1, FN 비용 2로 가정
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                score = -(fp * 1 + fn * 2)  # 음수로 최소화를 최대화 문제로 변환
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return float(best_threshold), float(best_score)
