import json
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import sys


class DeltaVectorDataLoader:
    def __init__(
        self,
        start_idx: int,
        end_idx: int,
        vec_len: int,
    ):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.vec_len = vec_len
        self.feature_dim = end_idx - start_idx

    def prepare_balanced_data(
        self, X_clean: np.ndarray, X_poison: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(X_clean) == 0 or len(X_poison) == 0:
            raise ValueError("Clean 또는 Poison 데이터가 비어있습니다.")

        min_samples = min(len(X_clean), len(X_poison))

        X = np.vstack([X_clean[:min_samples], X_poison[:min_samples]])
        y = np.concatenate([np.zeros(min_samples), np.ones(min_samples)])

        print(f"각 클래스 {min_samples}개로 균형 맞춤")

        return X, y

    def prepare_data(
        self,
        file_path_list: list[str],
        max_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        X_clean, X_poison = [], []

        if any("combined" in file for file in file_path_list):
            combined_train_file = [
                file for file in file_path_list if "combined" in file
            ]

            for file in combined_train_file:
                _X_c, _X_p = self._load_combined_file(file, max_samples)
                X_clean.append(_X_c)
                X_poison.append(_X_p)

        if any("clean" in file for file in file_path_list):
            clean_train_file = [file for file in file_path_list if "clean" in file]

            for file in clean_train_file:
                _X_c = self._load_single_class_file(file, max_samples)
                X_clean.append(_X_c)

        if any("poison" in file for file in file_path_list):
            poison_train_file = [file for file in file_path_list if "poison" in file]

            for file in poison_train_file:
                _X_p = self._load_single_class_file(file, max_samples)
                X_poison.append(_X_p)

        X_clean_final = np.concatenate(X_clean, axis=0)
        X_poison_final = np.concatenate(X_poison, axis=0)

        return X_clean_final, X_poison_final

    def _load_combined_file(
        self, file_path: str, max_samples_per_class: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        path = Path(file_path)
        if not path.exists():
            print(f"파일 없음: {path}")
            sys.exit(1)

        print(f"파일 로딩: {path.name}")

        X_clean, X_poison = [], []
        clean_count, poison_count = 0, 0

        with open(path, "rb") as f:
            for line in f:
                if max_samples_per_class:
                    if (
                        clean_count >= max_samples_per_class
                        and poison_count >= max_samples_per_class
                    ):
                        break

                try:
                    obj = json.loads(line)
                    label = obj.get("label")
                    delta_vec = obj.get("delta_vec")

                    if (
                        delta_vec
                        and len(delta_vec) >= self.vec_len
                        and label is not None
                    ):
                        features = np.array(
                            delta_vec[self.start_idx : self.end_idx], dtype=np.float32
                        )

                        if np.isfinite(features).all():
                            if label == 0:  # Clean
                                if (
                                    not max_samples_per_class
                                    or clean_count < max_samples_per_class
                                ):
                                    X_clean.append(features)
                                    clean_count += 1
                            elif label == 1:  # Poison
                                if (
                                    not max_samples_per_class
                                    or poison_count < max_samples_per_class
                                ):
                                    X_poison.append(features)
                                    poison_count += 1
                except Exception as e:
                    continue

        X_clean = (
            np.array(X_clean) if X_clean else np.array([]).reshape(0, self.feature_dim)
        )
        X_poison = (
            np.array(X_poison)
            if X_poison
            else np.array([]).reshape(0, self.feature_dim)
        )

        print(f"Clean: {len(X_clean)}개, Poison: {len(X_poison)}개 로드됨")

        return X_clean, X_poison

    def _load_single_class_file(
        self, file_path: str, max_samples_per_class: Optional[int] = None
    ) -> np.ndarray:
        path = Path(file_path)
        if not path.exists():
            print(f"파일 없음: {path}")
            sys.exit(1)

        print(f"파일 로딩: {path.name}")

        X_sample = []
        count = 0

        with open(path, "rb") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    delta_vec = obj.get("delta_vec")
                    if delta_vec and len(delta_vec) >= self.vec_len:
                        features = np.array(
                            delta_vec[self.start_idx : self.end_idx], dtype=np.float32
                        )

                        if np.isfinite(features).all():
                            if (
                                not max_samples_per_class
                                or count < max_samples_per_class
                            ):
                                X_sample.append(features)
                                count += 1

                except Exception as e:
                    print(f"경고: JSON 파싱 실패 - {e}")
                    continue

        X_sample = (
            np.array(X_sample)
            if X_sample
            else np.array([]).reshape(0, self.feature_dim)
        )

        print(f"Samples: {len(X_sample)}개 로드됨")

        return X_sample
