import torch.nn as nn
from typing import List, Optional
from transformers import PreTrainedTokenizer, PreTrainedModel


class LayerProcessor:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._layers_list: Optional[List[nn.Module]] = None

    def layers_list(self) -> List[nn.Module]:
        if self._layers_list is None:
            self._layers_list = self._get_layers_list()
        return self._layers_list

    def _get_layers_list(self) -> List[nn.Module]:
        candidates = "model.layers"  # 그 외 "model.decoder.layers", "transformer.h"

        def _resolve_path(root, dotted: str) -> list[nn.Module] | None:
            cur = root
            for attr in dotted.split("."):
                if hasattr(cur, attr):
                    cur = getattr(cur, attr)
                else:
                    return None
            print(f"Resolved path: {dotted} -> {cur}")
            return cur

        for path in [candidates]:
            cur = _resolve_path(self.model, path)
            if cur is None:
                continue

            if isinstance(cur, (list, tuple, nn.ModuleList)) or (
                hasattr(cur, "__getitem__") and hasattr(cur, "__len__")
            ):
                return list(cur)

        raise AttributeError("레이어 시퀀스를 찾을 수 없습니다.")

    def get_layer_count(self) -> int:
        return len(self.layers_list())

    def validate_selected_layers(self, selected_layers: List[int]) -> List[int]:
        total_layers = len(self.layers_list())
        valid_layers = []

        for idx in selected_layers:
            if idx < 0:
                valid_idx = total_layers + idx
            else:
                valid_idx = idx

            if 0 <= valid_idx < total_layers:
                valid_layers.append(idx)
            else:
                print(
                    f"레이어 인덱스 {idx}는 유효하지 않습니다. 총 레이어 수: {total_layers}"
                )

        return valid_layers

    def map_layer_to_hidden_state_index(
        self, layer_idx: int, num_hidden_states: int
    ) -> int:
        if layer_idx < 0:
            mapped_idx = num_hidden_states + layer_idx
        else:
            mapped_idx = layer_idx + 1

        if mapped_idx <= 0 or mapped_idx >= num_hidden_states:
            raise IndexError(
                f"레이어 인덱스 {layer_idx}가 hidden_states 범위를 벗어났습니다."
                f"(총 hidden_states: {num_hidden_states})"
            )
        return mapped_idx
