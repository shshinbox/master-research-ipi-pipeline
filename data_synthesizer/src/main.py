import random
from pathlib import Path

from synszr import DataSynthesizer
from synszr import utils


def main():
    from args_parser import create_parser

    parser = create_parser()
    args = parser.parse_args()

    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Data Synthesizer ===")

    samples = utils.load_jsonl_data(args.samples_file)
    injections = utils.load_jsonl_data(args.injections_file)

    synthesizer = DataSynthesizer(samples, injections, args.max_samples)
    clean_samples, poisoned_samples = synthesizer.synthesize_dataset()

    clean_file = output_dir / Path("clean_samples.jsonl")
    utils.save_samples(clean_samples, str(clean_file))

    poisoned_file = output_dir / Path("poisoned_samples.jsonl")
    utils.save_samples(poisoned_samples, str(poisoned_file))

    combined_dataset = synthesizer.create_combined_dataset(
        clean_samples, poisoned_samples
    )
    combined_file = output_dir / Path("combined_dataset.jsonl")
    utils.save_samples(combined_dataset, str(combined_file))

    print(f"\n완료. 파일들이 {output_dir}에 저장되었습니다.")


if __name__ == "__main__":
    main()
