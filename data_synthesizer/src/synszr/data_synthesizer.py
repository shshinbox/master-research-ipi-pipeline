import random
from typing import List, Dict, Tuple
from synszr.triggers_constants import TRIGGERS


class DataSynthesizer:
    def __init__(
        self,
        samples: List[Dict],
        injections: List[Dict],
        max_samples_per_class: int,
    ):
        self.samples = samples
        self.injections = injections
        self.max_samples_per_class = max_samples_per_class
        self.triggers = TRIGGERS

    def synthesize_dataset(self) -> Tuple[List[Dict], List[Dict]]:
        clean_samples = []
        poisoned_samples = []

        try:
            for s in (
                self.samples[: self.max_samples_per_class]
                if self.max_samples_per_class != 0
                else self.samples
            ):
                clean_samples.append(self.create_clean_sample(s))

            positions = ["beginning", "middle", "end"]

            total_samples_to_use = (
                self.max_samples_per_class
                if self.max_samples_per_class != 0
                else len(self.samples)
            )
            samples_per_position = total_samples_to_use // 3
            remainder = total_samples_to_use % 3

            for pos_idx, position in enumerate(positions):
                sample_subset = self.samples[
                    pos_idx
                    * samples_per_position : (pos_idx + 1)
                    * samples_per_position
                ]

                for sample in sample_subset:
                    injection = random.choice(self.injections)
                    poisoned_sample = self.create_poisoned_sample(
                        sample, injection, position
                    )
                    poisoned_samples.append(poisoned_sample)

            # 나머지 샘플 처리
            for i in range(remainder):
                sample = self.samples[3 * samples_per_position + i]
                injection = random.choice(self.injections)
                poisoned_sample = self.create_poisoned_sample(
                    sample, injection, positions[2]  # 'end' 위치에 삽입
                )
                poisoned_samples.append(poisoned_sample)

            print(
                f"합성 완료 - Clean: {len(clean_samples)}, Poisoned: {len(poisoned_samples)}"
            )
        except Exception as e:
            print(f"데이터 합성 중 오류 발생: {e}")
            raise e

        return clean_samples, poisoned_samples

    def create_clean_sample(self, sample: Dict) -> Dict:
        return {
            "id": f"clean_{sample.get('id')}",
            "question": sample.get("question"),
            "context": sample.get("context"),
            "label": 0,
            "injected_instruction": None,
            "injection_position": None,
            "trigger": None,
        }

    def create_poisoned_sample(
        self, sample: Dict, injection: Dict, position: str
    ) -> Dict:
        trigger = random.choice(self.triggers)
        injection_prompt = injection.get("instruction")
        assert injection_prompt is not None, "instruction이 없습니다."

        full_instruction = f"{trigger} {injection_prompt}"
        original_context = sample.get("context")
        assert original_context is not None, "context가 없습니다."

        poisoned_context = self.insert_at_position(
            original_context, full_instruction, position
        )
        return {
            "id": f"poisoned_{sample.get('id')}_{position}",
            "question": sample.get("question", ""),
            "context": poisoned_context,
            "label": 1,
            "injected_instruction": full_instruction,
            "injection_position": position,
            "trigger": trigger,
        }

    def insert_at_position(self, text: str, insert_text: str, position: str) -> str:
        sentences = text.split(". ")
        if position == "beginning":
            return f"{sentences[0]}. {insert_text} {'. '.join(sentences[1:])}"
        elif position == "middle":
            mid_idx = len(sentences) // 2
            return (
                ". ".join(sentences[:mid_idx])
                + f". {insert_text}. "
                + ". ".join(sentences[mid_idx:])
            )
        elif position == "end":
            return ". ".join(sentences[:-1]) + f". {insert_text} {sentences[-1]}"
        raise ValueError(f"알 수 없는 위치: {position}")

    def create_combined_dataset(
        self, clean_samples: List[Dict], poisoned_samples: List[Dict]
    ) -> List[Dict]:
        min_size = min(len(clean_samples), len(poisoned_samples))

        balanced_clean = clean_samples[:min_size]
        balanced_poisoned = poisoned_samples[:min_size]

        combined = balanced_clean + balanced_poisoned
        random.shuffle(combined)

        return combined
