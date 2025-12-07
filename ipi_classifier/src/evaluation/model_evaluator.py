import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
)
from .evaluation_metrics import EvaluationMetrics


class ModelEvaluator:

    @staticmethod
    def evaluate(
        y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray
    ) -> EvaluationMetrics:
        return EvaluationMetrics(
            accuracy=accuracy_score(y_true, y_pred),  # type: ignore
            f1=f1_score(y_true, y_pred),  # type: ignore
            precision=precision_score(y_true, y_pred),  # type: ignore
            recall=recall_score(y_true, y_pred),  # type: ignore
            roc_auc=roc_auc_score(y_true, y_probs),  # type: ignore
            confusion_matrix=confusion_matrix(y_true, y_pred),
        )

    @staticmethod
    def evaluate_multiple_thresholds(
        y_true: np.ndarray, y_probs: np.ndarray, thresholds: list
    ):
        for threshold in thresholds:
            y_pred = (y_probs >= threshold).astype(int)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)

            print(
                f"  임계값 {threshold:.1f}: "
                f"Acc={acc:.3f}, F1={f1:.3f}, "
                f"Precision={precision:.3f}, Recall={recall:.3f}"
            )
