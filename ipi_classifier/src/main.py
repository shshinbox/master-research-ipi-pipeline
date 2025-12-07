import time
from pathlib import Path

from poison_detector import DeltaVectorDataLoader, PoisonDetector, Visualizer, FileUtils
from evaluation import ModelEvaluator
from args_parser import create_parser


index_constants = [
    {"key": "llama3_layer15", "start_idx": 0, "end_idx": 4096, "vec_len": 4096},
    {"key": "llama3_layer23", "start_idx": 4096, "end_idx": 8192, "vec_len": 4096},
    {"key": "llama3_layer31", "start_idx": 8192, "end_idx": 12288, "vec_len": 4096},
    {"key": "mitral_layer15", "start_idx": 0, "end_idx": 4096, "vec_len": 4096},
    {"key": "mitral_layer23", "start_idx": 4096, "end_idx": 8192, "vec_len": 4096},
    {"key": "mitral_layer31", "start_idx": 8192, "end_idx": 12288, "vec_len": 4096},
    {"key": "phi3_layer15", "start_idx": 0, "end_idx": 3072, "vec_len": 3072},
    {"key": "phi3_layer23", "start_idx": 3072, "end_idx": 6144, "vec_len": 3072},
    {"key": "phi3_layer31", "start_idx": 6144, "end_idx": 9216, "vec_len": 3072},
]


def main():
    parser = create_parser()
    args = parser.parse_args()
    start_time = time.time()

    start_idx = 0
    end_idx = 0
    vec_len = 0

    for const in index_constants:
        if const["key"] == args.idx_type:
            start_idx = const["start_idx"]
            end_idx = const["end_idx"]
            vec_len = const["vec_len"]
            break

    print("\n === LogisticRegression ===")

    data_loader = DeltaVectorDataLoader(
        start_idx=start_idx, end_idx=end_idx, vec_len=vec_len
    )

    print("\n training 데이터 로드")
    X_clean, X_poison = data_loader.prepare_data(
        file_path_list=args.train_files, max_samples=args.max_samples
    )
    X_train, y_train = data_loader.prepare_balanced_data(X_clean, X_poison)

    print(
        f"train 데이터: {len(y_train)}개 "
        f"(Clean: {sum(y_train == 0)}, Poison: {sum(y_train == 1)}, "
        f"차원: {X_train.shape[1]})"
    )

    print("\n validation 데이터 로드")
    X_val_clean, X_val_poison = data_loader.prepare_data(
        file_path_list=args.val_files,
    )
    X_val, y_val = data_loader.prepare_balanced_data(X_val_clean, X_val_poison)
    print(
        f"validation 데이터: {len(y_val)}개 "
        f"(Clean: {sum(y_val == 0)}, Poison: {sum(y_val == 1)})"
    )

    print(f"데이터 준비 완료 (소요 시간: {time.time() - start_time:.2f}초)")

    print("\n 모델 초기화 및 훈련 시작")
    detector = PoisonDetector(
        C=args.C, max_iter=args.max_iter, random_state=args.random_state, n_jobs=-1
    )
    train_time = detector.train(X_train, y_train)

    val_probs = detector.predict_proba(X_val)
    val_pred = detector.predict(X_val)
    val_metrics = ModelEvaluator.evaluate(y_val, val_pred, val_probs)
    val_metrics.print_metrics("검증 데이터 성능")

    X_test_clean, X_test_poison = data_loader.prepare_data(
        file_path_list=args.test_files,
    )
    X_test, y_test = data_loader.prepare_balanced_data(X_test_clean, X_test_poison)

    test_probs = detector.predict_proba(X_test)
    test_pred = detector.predict(X_test)

    test_metrics = ModelEvaluator.evaluate(y_test, test_pred, test_probs)
    test_metrics.print_metrics(f"테스트 성능 (샘플: {len(y_test)}개)")
    test_metrics.print_metrics("테스트 데이터 성능")

    print(f"\n 결과 저장: {args.output_dir}")
    timestamp = int(start_time)

    model_path = Path(args.output_dir) / f"{args.model_name}_{timestamp}.joblib"
    FileUtils.save_model(detector.model, str(model_path))

    val_cm_path = Path(args.output_dir) / f"{args.model_name}_val_cm_{timestamp}.png"
    Visualizer.save_confusion_matrix(
        val_metrics.confusion_matrix,
        ["Clean", "Poison"],
        "Validation Confusion Matrix",
        str(val_cm_path),
    )

    test_cm_path = Path(args.output_dir) / f"{args.model_name}_test_cm_{timestamp}.png"
    Visualizer.save_confusion_matrix(
        test_metrics.confusion_matrix,
        ["Clean", "Poison"],
        "Test Confusion Matrix",
        str(test_cm_path),
    )

    total_time = time.time() - start_time
    print(f"\n 완료! 총 소요 시간: {total_time:.2f}초")
    print(f"\n 저장된 파일:")
    print(f"  - 모델: {model_path}")
    print(f"  - 검증 혼동행렬: {val_cm_path}")
    print(f"  - 테스트 혼동행렬: {test_cm_path}")


if __name__ == "__main__":
    main()
