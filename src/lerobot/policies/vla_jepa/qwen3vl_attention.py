# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3-VL text attention with a single fused QKV projection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers.cache_utils import Cache
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextAttention,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )
else:
    Cache = None
    ALL_ATTENTION_FUNCTIONS = None
    Qwen3VLTextAttention = nn.Module
    apply_rotary_pos_emb = None
    eager_attention_forward = None


class Qwen3VLFusedTextAttention(Qwen3VLTextAttention):
    """Qwen3-VL text attention that computes Q, K and V with one linear layer.

    Qwen3-VL uses grouped-query attention, so Q and K/V generally have different
    output sizes. The packed projection is consequently split by explicit sizes
    rather than reshaped into three equal chunks as in the vision attention.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)

        q_size = self.q_proj.out_features
        k_size = self.k_proj.out_features
        v_size = self.v_proj.out_features
        projections = (self.q_proj, self.k_proj, self.v_proj)
        bias_values = {projection.bias is not None for projection in projections}
        if len(bias_values) != 1:
            raise ValueError("Qwen3-VL Q, K and V projections must use the same bias setting")

        self.qkv_split_sizes = (q_size, k_size, v_size)
        self.qkv = nn.Linear(
            self.q_proj.in_features,
            q_size + k_size + v_size,
            bias=bias_values.pop(),
            device=self.q_proj.weight.device,
            dtype=self.q_proj.weight.dtype,
        )
        del self.q_proj
        del self.k_proj
        del self.v_proj

        # Existing VLA-JEPA checkpoints contain the upstream split projection
        # keys. Combine them transparently before strict state-dict loading.
        self.register_load_state_dict_pre_hook(self._fuse_legacy_qkv_state_dict)

    def _fuse_legacy_qkv_state_dict(
        self,
        module: nn.Module,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        del module, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        for suffix in ("weight", "bias"):
            fused_key = f"{prefix}qkv.{suffix}"
            split_keys = [f"{prefix}{projection}_proj.{suffix}" for projection in ("q", "k", "v")]
            present_split_keys = [key for key in split_keys if key in state_dict]
            if fused_key not in state_dict and len(present_split_keys) == len(split_keys):
                state_dict[fused_key] = torch.cat([state_dict.pop(key) for key in split_keys], dim=0)

    @classmethod
    def from_attention(cls, attention: Qwen3VLTextAttention) -> Qwen3VLFusedTextAttention:
        """Create a fused module with exactly the same parameters and mode."""
        projection_requires_grad = {
            attention.q_proj.weight.requires_grad,
            attention.k_proj.weight.requires_grad,
            attention.v_proj.weight.requires_grad,
        }
        if len(projection_requires_grad) != 1:
            raise ValueError("Qwen3-VL Q, K and V projections must share requires_grad")

        fused = cls(attention.config, attention.layer_idx)
        fused.to(device=attention.q_proj.weight.device, dtype=attention.q_proj.weight.dtype)
        fused.load_state_dict(attention.state_dict(), strict=True)
        fused.qkv.requires_grad_(projection_requires_grad.pop())

        old_parameters = dict(attention.named_parameters())
        for name, parameter in fused.named_parameters():
            if name == "qkv.weight" or name == "qkv.bias":
                continue
            old_parameter = old_parameters.get(name)
            if old_parameter is not None:
                parameter.requires_grad_(old_parameter.requires_grad)
        fused.train(attention.training)
        return fused

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        qkv = self.qkv(hidden_states)
        query_states, key_states, value_states = qkv.split(self.qkv_split_sizes, dim=-1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(key_states.view(hidden_shape)).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


def fuse_qwen3vl_text_qkv(language_model: nn.Module) -> int:
    """Replace every dense Qwen3-VL text attention with the fused variant."""
    fused_count = 0
    for layer in language_model.layers:
        attention = layer.self_attn
        if isinstance(attention, Qwen3VLFusedTextAttention):
            continue
        if not isinstance(attention, Qwen3VLTextAttention):
            raise TypeError(
                "Expected Qwen3VLTextAttention while fusing the Qwen3-VL language model, "
                f"got {type(attention).__name__}"
            )
        layer.self_attn = Qwen3VLFusedTextAttention.from_attention(attention)
        fused_count += 1
    return fused_count
