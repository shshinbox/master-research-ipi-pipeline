from argparse import ArgumentParser, RawDescriptionHelpFormatter


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Activations Extractor",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="",
    )

    parser.add_argument(
        "--in",
        dest="in_jsonl",
        type=str,
        default="squadv2val_clean_samples.jsonl",
        help="입력 JSONL 파일 경로",
    )
    parser.add_argument(
        "--out",
        dest="out_jsonl",
        type=str,
        default="squadv2val_val_deltas_clean_samples.jsonl",
        help="출력 JSONL 파일 경로",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="mistralai/Mistral-7B-v0.3",
        help="사용할 모델 ID",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        choices=["auto", "cuda:0", "cuda:1"],
        help="사용할 디바이스",
    )
    parser.add_argument("--max_len", type=int, default=8192, help="최대 시퀀스 길이")
    parser.add_argument(
        "--eos_token", type=str, default="<EOS>", help="EOS 토큰 문자열"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="15,23,31",
        help='처리할 레이어 인덱스 (콤마 구분). 예: "-4,-3,-2,-1" 또는 "15,23,31"',
    )
    parser.add_argument("--batch_size", type=int, default=1, help="배치 크기")
    parser.add_argument(
        "--limit", type=int, default=0, help="처리할 최대 샘플 수 (0이면 전체)"
    )
    parser.add_argument(
        "--flush_every", type=int, default=100, help="주기적 플러시 간격"
    )
    parser.add_argument("--no_8bit", action="store_true", help="8bit 양자화 비활성화")
    parser.add_argument("--seed", type=int, default=42, help="시드")

    return parser
