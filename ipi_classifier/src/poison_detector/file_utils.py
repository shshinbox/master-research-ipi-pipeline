import json
import joblib
from pathlib import Path
from typing import Dict, Any


class FileUtils:

    @staticmethod
    def save_model(model: Any, save_path: str):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
        print(f"모델 저장: {save_path}")

    @staticmethod
    def load_model(load_path: str) -> Any:
        return joblib.load(load_path)

    @staticmethod
    def save_json(data: Dict, save_path: str, indent: int = 2):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(data, f, indent=indent)

        print(f"JSON 저장: {save_path}")

    @staticmethod
    def load_json(load_path: str) -> Dict:
        with open(load_path, "r") as f:
            return json.load(f)
