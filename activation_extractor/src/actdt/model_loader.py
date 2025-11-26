import sys
import torch
import torch.nn as nn

from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from typing import cast, Tuple


def load_model_tokenizer(
    model_id, device, load_in_8bit, compute_dtpye, eos_token
) -> Tuple[PreTrainedTokenizer, PreTrainedModel, int]:
    if compute_dtpye is None:
        compute_dtpye = torch.float16

    bnb_cfg = None
    if load_in_8bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

    tok = AutoTokenizer.from_pretrained(
        model_id, use_fast=False, local_files_only=False
    )

    tok.padding_side = "right"
    if hasattr(tok, "truncation_side"):
        tok.truncation_side = "left"

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    added = False
    eos_id = tok.convert_tokens_to_ids(eos_token)
    if eos_id is None or eos_id == tok.unk_token_id:
        tok.add_special_tokens({"additional_special_tokens": [eos_token]})
        added = True
        eos_id = tok.convert_tokens_to_ids(eos_token)

    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": device},
        quantization_config=bnb_cfg,
        dtype=compute_dtpye if load_in_8bit else None,
        local_files_only=False,
    )

    mdl.eval()

    if hasattr(mdl, "config"):
        mdl.config.use_cache = False
        if mdl.config.pad_token_id is None:
            mdl.config.pad_token_id = tok.pad_token_id

    if added:
        mdl.resize_token_embeddings(len(tok))
        with torch.no_grad():
            original_eos_id: int = int(tok.eos_token_id)
            eos_id = int(eos_id)

            emb_layer = cast(nn.Embedding, mdl.get_input_embeddings())

            weight = emb_layer.weight
            weight[eos_id].copy_(weight[original_eos_id])

    if next(mdl.parameters()).device.type != "cuda":
        print("모델이 GPU에 로드되지 않았습니다.")
        sys.exit(1)

    return tok, mdl, eos_id
