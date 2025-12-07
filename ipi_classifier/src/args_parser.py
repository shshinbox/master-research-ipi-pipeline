import argparse


def create_parser():
    parser = argparse.ArgumentParser(description="Linear Probe")

    parser.add_argument(
        "--train_files",
        nargs="*",
        default=[
            # "clean_samples.jsonl",
            # "poisoned_samples.jsonl",
            "squadv1_train_deltas_clean_samples.jsonl",
            "squadv1_train_deltas_poisoned_samples.jsonl",
        ],
        help="훈련 데이터 파일",
    )
    parser.add_argument(
        "--val_files",
        nargs="*",
        default=[
            # "clean_samples.jsonl",
            # "poisoned_samples.jsonl",
            "squadv1val_deltas_clean_samples.jsonl",
            "squadv1val_deltas_syntest_poisoned_samples.jsonl",
        ],
        help="검증 데이터 파일",
    )
    parser.add_argument(
        "--test_files",
        nargs="*",
        default=[
            # "clean_samples.jsonl",
            # "poisoned_samples.jsonl",
            "hotpotqa_val_deltas_clean_samples.jsonl",
            "hotpotqa_val_test_syn_deltas_poisoned_samples.jsonl",
            "squadv1val_deltas_clean_samples.jsonl",
            "squadv1val_deltas_syntest_poisoned_samples.jsonl",
        ],
        help="테스트 데이터 파일",
    )

    parser.add_argument(
        "--idx_type", type=str, default="llama3_layer15", help="델타 벡터 인덱스 타입"
    )
    parser.add_argument(
        "--max_samples", type=int, default=90000, help="클래스당 최대 샘플 수"
    )

    # 모델 하이퍼파라미터
    parser.add_argument("--C", type=float, default=1.0, help="정규화 강도")
    parser.add_argument("--max_iter", type=int, default=2000, help="최대 반복 횟수")
    parser.add_argument("--random_state", type=int, default=42, help="랜덤 시드")

    # 출력 설정
    parser.add_argument("--output_dir", default="./outputs", help="모델 저장 경로")
    parser.add_argument("--model_name", default="mitral31_clf", help="모델 이름")

    return parser
