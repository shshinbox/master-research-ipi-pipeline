import numpy as np
from typing import Dict
from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    accuracy: float
    f1: float
    precision: float
    recall: float
    roc_auc: float
    confusion_matrix: np.ndarray

    def to_dict(self) -> Dict:
        return {
            "accuracy": self.accuracy,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "roc_auc": self.roc_auc,
        }

    def print_metrics(self, title: str = "성능 평가"):
        print(f"{title}:")
        print(f"  정확도: {self.accuracy:.4f}")
        print(f"  F1 점수: {self.f1:.4f}")
        print(f"  정밀도: {self.precision:.4f}")
        print(f"  재현율: {self.recall:.4f}")
        print(f"  ROC AUC: {self.roc_auc:.4f}")
