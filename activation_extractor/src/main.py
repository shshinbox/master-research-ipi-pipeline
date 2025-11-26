import sys

from actdt import ActivationsPipeline

from exec import parse_layers_arg, validate_args, set_repro, require_cuda

from exec import mistral7b_create_parser as create_parser

# from exec import llama3_create_parser as create_parser

# from exec import phi3_create_parser as create_parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    validate_args(args)

    try:
        set_repro(args.seed)
        device = require_cuda()

        pipeline = ActivationsPipeline(
            model_id=args.model_id,
            device=device,
            load_in_8bit=not args.no_8bit,
            eos_token=args.eos_token,
        )

        selected_layers = parse_layers_arg(args.layers)

        written = pipeline.process_dataset(
            in_jsonl=args.in_jsonl,
            out_jsonl=args.out_jsonl,
            selected_layers=selected_layers,
            max_len=args.max_len,
            batch_size=args.batch_size,
            limit=args.limit,
            flush_every=args.flush_every,
        )

        print(f"처리 완료! 총 {written}개 샘플 처리되었습니다.")

    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"오류 발생: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
