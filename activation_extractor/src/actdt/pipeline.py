import json
import os
import torch
from tqdm import tqdm
from typing import List, Any, Iterator, Dict, Set

from actdt.model_loader import load_model_tokenizer
from actdt.layer_processor import LayerProcessor
from actdt.hidden_collector import HiddenStateCollector
from actdt.template_builder import PromptTemplateBuilder


class ActivationsPipeline:
    def __init__(
        self,
        model_id: str,
        device: str,
        load_in_8bit: bool,
        eos_token: str,
    ):

        print("파이프라인 초기화 시작")

        self.model_id = model_id
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.eos_token = eos_token
        self.device = device

        print("모델&토크나이저 로딩")

        self.tokenizer, self.model, self.eos_id = load_model_tokenizer(
            model_id=self.model_id,
            device=self.device,
            load_in_8bit=self.load_in_8bit,
            eos_token=self.eos_token,
            compute_dtpye=torch.float16,
        )

        self.layer_processor = LayerProcessor(self.model, self.tokenizer)
        self.template_builder = PromptTemplateBuilder()
        self.hidden_collector = HiddenStateCollector(
            self.model, self.tokenizer, self.layer_processor
        )

        print("파이프라인 초기화 완료")
        print(f"모델: {self.model_id}")
        print(f"디바이스: {self.device}")
        print(f"총 레이어 수:{self.layer_processor.get_layer_count()}")

    def process_dataset(
        self,
        in_jsonl: str,
        out_jsonl: str,
        selected_layers: List[int],
        max_len: int,
        batch_size: int,
        limit: int,
        flush_every: int,
        resume: bool = True,
    ) -> int:
        valid_layers = self.layer_processor.validate_selected_layers(
            selected_layers=selected_layers,
        )
        if not valid_layers:
            raise ValueError("유효한 레이어가 없습니다.")

        print(f"처리할 레이어: {valid_layers}")
        print(f"입력 파일: {in_jsonl}")
        print(f"출력 파일: {out_jsonl}")
        print(f"배치 크기: {batch_size}")

        processed_ids = set()
        file_mode = "w"

        if resume and os.path.exists(out_jsonl):
            processed_ids = self.get_processed_ids(out_jsonl)
            if processed_ids:
                file_mode = "a"
                print(f"기존에 처리된 샘플 {len(processed_ids)}개를 건너뜁니다.")
            else:
                print(
                    "기존 출력 파일이 비어있거나 유효한 데이터가 없습니다. 새로 시작합니다."
                )

        total_samples = self.count_total_samples(in_jsonl, limit)
        print(f"입력의 총 샘플 수: {total_samples}")
        written = 0
        skipped = 0

        with open(file=out_jsonl, mode=file_mode, encoding="utf-8") as fout:
            all_items = []
            for i, item in enumerate(self.read_jsonl(in_jsonl)):
                if limit and i >= limit:
                    break
                all_items.append(item)

            total_batches = (len(all_items) + batch_size - 1) // batch_size
            pbar = tqdm(
                self.create_batches(all_items, batch_size),
                desc="배치 처리 중",
                total=total_batches,
            )

            for batch_idx, batch_items in enumerate(pbar):
                try:
                    valid_batch_items = []
                    batch_skipped = 0

                    for item in batch_items:
                        if not isinstance(item, dict):
                            continue

                        sample_id = item.get("id")
                        if (
                            resume
                            and sample_id is not None
                            and str(sample_id) in processed_ids
                        ):
                            batch_skipped += 1
                            continue

                        q = item.get("question")
                        ctx = item.get("context")

                        if not q or not ctx:
                            continue

                        valid_batch_items.append(item)

                    skipped += batch_skipped

                    if not valid_batch_items:
                        pbar.set_postfix_str(f"written={written}, skipped={skipped}")
                        continue

                    questions = [item["question"] for item in valid_batch_items]
                    contexts = [item["context"] for item in valid_batch_items]

                    batch_results = self.hidden_collector.process_batch_samples(
                        questions=questions,
                        contexts=contexts,
                        eos_id=self.eos_id,
                        device=self.device,
                        max_len=max_len,
                        selected_layers=valid_layers,
                    )

                    for item, result in zip(valid_batch_items, batch_results):
                        out_item: Dict[str, Any] = {
                            "id": item.get("id"),
                            "selected_layers": result["selected_layers"],
                            "tok_len_q": result["tok_len_q"],
                            "tok_len_q_ctx": result["tok_len_q_ctx"],
                            "delta_vec": result["delta_vec"],
                        }
                        fout.write(json.dumps(out_item, ensure_ascii=False) + "\n")
                        written += 1

                    if written % flush_every == 0:
                        fout.flush()
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

                    pbar.set_postfix_str(f"written={written}, skipped={skipped}")

                except Exception as e:
                    tqdm.write(f"\n배치 {batch_idx} 처리 중 오류: {e}")
                    continue

        print(f"\n처리 완료: {written}개 새로 작성, {skipped}개 건너뜀")
        return written

    def get_processed_ids(self, out_jsonl: str) -> Set[str]:
        processed_ids = set()

        if not os.path.exists(out_jsonl):
            return processed_ids

        try:
            with open(out_jsonl, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if "id" in data and data["id"] is not None:
                            processed_ids.add(str(data["id"]))
                    except json.JSONDecodeError:
                        print(
                            f"경고: 출력 파일의 {line_num}번째 줄을 파싱할 수 없습니다."
                        )
                        continue
        except Exception as e:
            print(f"기존 출력 파일 읽기 중 오류: {e}")

        return processed_ids

    def count_total_samples(self, in_jsonl: str, limit: int) -> int:
        count = 0
        try:
            with open(in_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
                        if limit > 0 and count >= limit:
                            break
        except Exception:
            return 0
        return count

    def create_batches(
        self, items: List[Dict], batch_size: int
    ) -> Iterator[List[Dict]]:
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    def read_jsonl(self, path: str) -> Iterator[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
