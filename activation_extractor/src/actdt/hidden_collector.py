import re
import torch
from typing import Dict, List
from transformers import PreTrainedTokenizer, BatchEncoding, PreTrainedModel

from actdt.template_builder import PromptTemplateBuilder
from actdt.layer_processor import LayerProcessor


class HiddenStateCollector:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        layer_processor: LayerProcessor,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_processor = layer_processor

    def smart_truncate_template(self, text: str, max_tokens: int):
        full_encoding: BatchEncoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            padding=False,
        )

        full_length = full_encoding.input_ids.size(1)

        if full_length < max_tokens:
            return text

        main_pattern = r"<MAIN>(.*?)</MAIN>"
        match = re.search(main_pattern, text, re.DOTALL)

        if not match:
            raise ValueError("주어진 입력이 템플릿과 맞지 않아 처리할 수 없습니다.")

        main_content = match.group(1).strip()
        before_main = text[: match.start()]
        after_main = text[match.end() :]

        lines = main_content.split("\n")
        question_part = lines[0]
        context_part = lines[1:]

        if len(context_part) != 0:
            context_text = "\n".join(context_part)
            context_encoding: BatchEncoding = self.tokenizer(
                context_text,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=False,
                padding=False,
            )
            context_tokens = context_encoding.input_ids[0]

            fixed_parts = f"{before_main}<MAIN>\n{question_part}\n </MAIN>"
            fixed_encoding = self.tokenizer(
                fixed_parts,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=False,
                padding=False,
            )
            fixed_length = fixed_encoding.input_ids.size(1)

            available_tokens = max_tokens - fixed_length - 10

            if available_tokens > 0:
                truncated_context_tokens = context_tokens[-available_tokens:]
                truncated_context = self.tokenizer.decode(
                    truncated_context_tokens, skip_special_tokens=True
                )
                result = f"{before_main}<MAIN>\n{question_part}\n{truncated_context}\n</MAIN>{after_main}"
                return result

        raise ValueError("주어진 입력이 템플릿과 맞지 않아 처리할 수 없습니다.")

    @torch.inference_mode()
    def collect_last_token_hidden_across_layers(
        self,
        inputs: Dict[str, torch.Tensor],
        last_idx: torch.Tensor,
        selected_layers: List[int],
    ) -> Dict[int, torch.Tensor]:
        out = self.model(**inputs, output_hidden_states=True, return_dict=True)

        hs_all = out.hidden_states

        num_hs = len(hs_all)
        t_idx = int(last_idx.item())

        result = {}

        for li in selected_layers:
            mapped_idx = self.layer_processor.map_layer_to_hidden_state_index(
                layer_idx=li, num_hidden_states=num_hs
            )

            v = hs_all[mapped_idx][0, t_idx, :].detach().to(torch.float16).cpu()

            result[li] = v

        return result

    @torch.inference_mode()
    def collect_last_token_hidden_across_layers_batch(
        self,
        inputs: Dict[str, torch.Tensor],
        last_indices: torch.Tensor,
        selected_layers: List[int],
    ) -> List[Dict[int, torch.Tensor]]:
        out = self.model(**inputs, output_hidden_states=True, return_dict=True)

        hs_all = out.hidden_states
        num_hs = len(hs_all)
        batch_size = inputs["input_ids"].size(0)

        batch_results = []

        for batch_idx in range(batch_size):
            t_idx = int(last_indices[batch_idx].item())
            result = {}

            for li in selected_layers:
                mapped_idx = self.layer_processor.map_layer_to_hidden_state_index(
                    layer_idx=li, num_hidden_states=num_hs
                )

                v = (
                    hs_all[mapped_idx][batch_idx, t_idx, :]
                    .detach()
                    .to(torch.float16)
                    .cpu()
                )
                result[li] = v

            batch_results.append(result)

        return batch_results

    def compute_delta_vector(
        self,
        h_q: Dict[int, torch.Tensor],
        h_q_ctx: Dict[int, torch.Tensor],
        selected_layers: List[int],
    ) -> torch.Tensor:

        delta_parts = []

        for li in selected_layers:
            if li not in h_q or li not in h_q_ctx:
                continue
            vq = h_q[li]
            vc = h_q_ctx[li]
            dv = vc - vq

            delta_parts.append(dv)

        delta_vec: torch.Tensor = torch.cat(delta_parts, dim=0)

        return delta_vec

    def encode_ensure_eos_at_last(
        self, text: str, eos_id: int, max_len: int, device: str
    ) -> Dict[str, torch.Tensor]:

        processed_text = self.smart_truncate_template(text=text, max_tokens=max_len)

        enc = self.tokenizer(
            processed_text,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        input_ids = enc.input_ids[0]

        if input_ids[-1].item() != eos_id:
            if len(input_ids) < max_len:
                input_ids = torch.cat(
                    [input_ids, torch.tensor([eos_id], dtype=input_ids.dtype)]
                )
            else:
                input_ids[-1] = eos_id

        assert input_ids[-1].item() == eos_id, "EOS가 마지막 토큰이 아닙니다."

        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids.unsqueeze(0).to(device=device),
            "attention_mask": attention_mask.unsqueeze(0).to(device=device),
        }

    def encode_batch_ensure_eos_at_last(
        self, texts: List[str], eos_id: int, max_len: int, device: str
    ) -> Dict[str, torch.Tensor]:
        processed_texts = []
        for text in texts:
            processed_text = self.smart_truncate_template(text=text, max_tokens=max_len)
            processed_texts.append(processed_text)

        # 배치 인코딩 (padding=True로 설정하여 같은 길이로 맞춤)
        enc = self.tokenizer(
            processed_texts,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
            padding=True,
        )

        input_ids = enc.input_ids  # (batch_size, seq_len)
        attention_mask = enc.attention_mask  # (batch_size, seq_len)

        # 각 샘플의 실제 마지막 토큰 위치에 EOS 추가
        for i in range(input_ids.size(0)):
            # 현재 샘플의 실제 길이 찾기 (패딩 제외)
            actual_length = attention_mask[i].sum().item()

            if actual_length < max_len:
                # 마지막 토큰이 EOS가 아니면 EOS 추가
                if input_ids[i, actual_length - 1].item() != eos_id:
                    if actual_length < input_ids.size(1):
                        input_ids[i, actual_length] = eos_id
                        attention_mask[i, actual_length] = 1
                    else:
                        input_ids[i, actual_length - 1] = eos_id
            else:
                # 최대 길이에 도달한 경우 마지막 토큰을 EOS로 설정
                input_ids[i, max_len - 1] = eos_id

        return {
            "input_ids": input_ids.to(device=device),
            "attention_mask": attention_mask.to(device=device),
        }

    def process_batch_samples(
        self,
        questions: List[str],
        contexts: List[str],
        eos_id: int,
        max_len: int,
        device: str,
        selected_layers: List[int],
    ) -> List[Dict]:
        if len(questions) != len(contexts):
            raise ValueError("questions와 contexts의 갯수가 같아야 합니다.")

        batch_size = len(questions)
        builder = PromptTemplateBuilder()

        # 모든 질문과 컨텍스트에 대해 템플릿 생성
        q_only_texts = []
        q_ctx_texts = []

        for question, context in zip(questions, contexts):
            templates = builder.build_inputs_with_template(question, context)
            q_only_texts.append(templates["q_only"])
            q_ctx_texts.append(templates["q_ctx"])

        # 배치 인코딩
        input_q_batch = self.encode_batch_ensure_eos_at_last(
            texts=q_only_texts, device=device, eos_id=eos_id, max_len=max_len
        )
        input_q_ctx_batch = self.encode_batch_ensure_eos_at_last(
            texts=q_ctx_texts, device=device, eos_id=eos_id, max_len=max_len
        )

        # 각 샘플의 마지막 토큰 인덱스 계산
        last_indices_q = input_q_batch["attention_mask"].sum(dim=1) - 1
        last_indices_q_ctx = input_q_ctx_batch["attention_mask"].sum(dim=1) - 1

        # 배치로 hidden states 수집
        h_q_batch = self.collect_last_token_hidden_across_layers_batch(
            inputs=input_q_batch,
            last_indices=last_indices_q,
            selected_layers=selected_layers,
        )

        h_q_ctx_batch = self.collect_last_token_hidden_across_layers_batch(
            inputs=input_q_ctx_batch,
            last_indices=last_indices_q_ctx,
            selected_layers=selected_layers,
        )

        # 각 샘플별로 delta vector 계산
        results = []
        for i in range(batch_size):
            delta_vec = self.compute_delta_vector(
                h_q=h_q_batch[i],
                h_q_ctx=h_q_ctx_batch[i],
                selected_layers=selected_layers,
            )

            result = {
                "delta_vec": delta_vec.cpu().tolist(),
                "tok_len_q": int(last_indices_q[i].item()) + 1,
                "tok_len_q_ctx": int(last_indices_q_ctx[i].item()) + 1,
                "selected_layers": selected_layers,
            }
            results.append(result)

        return results

    def process_single_sample(
        self,
        question: str,
        context: str,
        eos_id: int,
        max_len: int,
        device: str,
        selected_layers: List[int],
    ) -> Dict:

        builder = PromptTemplateBuilder()
        templates = builder.build_inputs_with_template(question, context)

        input_q = self.encode_ensure_eos_at_last(
            text=templates["q_only"], device=device, eos_id=eos_id, max_len=max_len
        )
        input_q_ctx = self.encode_ensure_eos_at_last(
            text=templates["q_ctx"], device=device, eos_id=eos_id, max_len=max_len
        )

        last_idx_q = input_q["attention_mask"].sum(dim=1) - 1
        last_idx_q_ctx = input_q_ctx["attention_mask"].sum(dim=1) - 1

        h_q = self.collect_last_token_hidden_across_layers(
            inputs=input_q,
            last_idx=last_idx_q[0],
            selected_layers=selected_layers,
        )

        h_q_ctx = self.collect_last_token_hidden_across_layers(
            inputs=input_q_ctx,
            last_idx=last_idx_q_ctx[0],
            selected_layers=selected_layers,
        )

        delta_vec = self.compute_delta_vector(
            h_q=h_q,
            h_q_ctx=h_q_ctx,
            selected_layers=selected_layers,
        )

        return {
            "delta_vec": delta_vec.cpu().tolist(),
            "tok_len_q": int(last_idx_q.item()) + 1,
            "tok_len_q_ctx": int(last_idx_q_ctx.item()) + 1,
            "selected_layers": selected_layers,
        }
