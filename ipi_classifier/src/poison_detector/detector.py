import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
import time


class PoisonDetector:

    def __init__(
        self,
        C: float,
        max_iter: int,
        random_state: int,
        n_jobs: int,
    ):

        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.threshold = 0.5
        self.random_state = random_state

    def train(
        self, X_train: np.ndarray, y_train: np.ndarray, shuffle_data: bool = True
    ) -> float:

        if shuffle_data:
            X_train, y_train = shuffle(X_train, y_train, random_state=self.random_state)  # type: ignore

        print("\n모델 학습 시작")
        start_time = time.time()

        self.model.fit(X_train, y_train)

        train_time = time.time() - start_time
        print(f"모델 학습 완료 (소요 시간: {train_time:.2f}초)")

        return train_time

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.model.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def set_threshold(self, threshold: float):
        self.threshold = threshold
        print(f"임계값 설정: {threshold:.3f}")
