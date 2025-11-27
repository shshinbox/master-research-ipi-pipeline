from typing import List, Dict
import json


def save_samples(samples: List[Dict], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"저장 완료: {output_path} ({len(samples)}개 샘플)")


def load_jsonl_data(jsonl_path: str) -> List[Dict]:
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    print(f"{jsonl_path} JSONL 데이터 로드: {len(data)}개")
    return data
