import os
import sys
import torch


def validate_args(args) -> None:
    if not os.path.exists(args.in_jsonl):
        print(f"입력 파일을 찾을 수 없습니다: {args.in_jsonl}")
        sys.exit(1)

    out_dir = os.path.dirname(args.out_jsonl)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    try:
        layers = parse_layers_arg(args.layers)
        if not layers:
            print("유효한 레이어 인덱스가 없습니다.")
            sys.exit(1)
    except ValueError as e:
        print(f"레이어 인자 파싱 오류: {e}")
        sys.exit(1)


def set_repro(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def parse_layers_arg(layers_str: str) -> list[int]:
    parts = [p.strip() for p in layers_str.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def require_cuda() -> str:
    if not torch.cuda.is_available():
        print("CUDA GPU가 필요합니다.")
        sys.exit(1)
    return "cuda:0"
