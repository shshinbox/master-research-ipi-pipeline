from argparse import ArgumentParser


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Data Synthesizer")

    parser.add_argument(
        "--samples_file",
        default="samples_file.jsonl",
        help="samples JSONL 파일 경로",
    )
    parser.add_argument(
        "--injections_file",
        default="injections_file.json",
        help="Injected Instructions JSONL 파일 경로",
    )
    parser.add_argument(
        "--output_dir",
        default="data/synszr/output",
        help="출력 디렉토리",
    )
    parser.add_argument(
        "--max_samples", type=int, default=0, help="클래스당 최대 샘플 수"
    )
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")

    return parser
