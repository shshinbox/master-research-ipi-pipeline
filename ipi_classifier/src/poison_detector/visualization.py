import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


class Visualizer:

    @staticmethod
    def save_confusion_matrix(cm: np.ndarray, labels: list, title: str, save_path: str):
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.title(title)
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"혼동행렬 저장: {save_path}")
