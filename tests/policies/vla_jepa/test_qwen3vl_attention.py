#!/usr/bin/env python

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

pytest.importorskip("transformers")

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig  # noqa: E402
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention  # noqa: E402

from lerobot.policies.vla_jepa.qwen3vl_attention import Qwen3VLFusedTextAttention  # noqa: E402


def _make_attention() -> Qwen3VLTextAttention:
    config = Qwen3VLTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        attention_bias=True,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return Qwen3VLTextAttention(config, layer_idx=0).eval()


def test_fused_qkv_matches_split_qkv_output_and_gradients() -> None:
    torch.manual_seed(0)
    split_attention = _make_attention()
    fused_attention = Qwen3VLFusedTextAttention.from_attention(deepcopy(split_attention))

    split_hidden = torch.randn(2, 5, 16, requires_grad=True)
    fused_hidden = split_hidden.detach().clone().requires_grad_()
    position_embeddings = (torch.randn(2, 5, 4), torch.randn(2, 5, 4))

    split_output, split_weights = split_attention(
        split_hidden, position_embeddings=position_embeddings, attention_mask=None
    )
    fused_output, fused_weights = fused_attention(
        fused_hidden, position_embeddings=position_embeddings, attention_mask=None
    )

    torch.testing.assert_close(fused_output, split_output)
    torch.testing.assert_close(fused_weights, split_weights)

    split_output.sum().backward()
    fused_output.sum().backward()
    torch.testing.assert_close(fused_hidden.grad, split_hidden.grad)
    torch.testing.assert_close(
        fused_attention.qkv.weight.grad,
        torch.cat(
            [
                split_attention.q_proj.weight.grad,
                split_attention.k_proj.weight.grad,
                split_attention.v_proj.weight.grad,
            ]
        ),
    )


def test_fused_qkv_uses_gqa_split_sizes_and_loads_legacy_state_dict() -> None:
    split_attention = _make_attention()
    legacy_state_dict = deepcopy(split_attention.state_dict())
    fused_attention = Qwen3VLFusedTextAttention.from_attention(split_attention)

    assert fused_attention.qkv_split_sizes == (16, 8, 8)
    assert fused_attention.qkv.out_features == 32
    assert not hasattr(fused_attention, "q_proj")
    assert not hasattr(fused_attention, "k_proj")
    assert not hasattr(fused_attention, "v_proj")
    assert set(fused_attention.state_dict()) == {
        "qkv.weight",
        "qkv.bias",
        "o_proj.weight",
        "o_proj.bias",
        "q_norm.weight",
        "k_norm.weight",
    }

    reloaded_attention = Qwen3VLFusedTextAttention(split_attention.config, layer_idx=0)
    reloaded_attention.load_state_dict(legacy_state_dict, strict=True)
    torch.testing.assert_close(reloaded_attention.qkv.weight, fused_attention.qkv.weight)
    torch.testing.assert_close(reloaded_attention.qkv.bias, fused_attention.qkv.bias)


def test_language_model_fusion_loads_prefixed_legacy_checkpoint_keys() -> None:
    from lerobot.policies.vla_jepa.qwen3vl_attention import fuse_qwen3vl_text_qkv

    language_model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.self_attn = _make_attention()
    language_model.layers = torch.nn.ModuleList([layer])

    legacy_state_dict = deepcopy(language_model.state_dict())
    expected_qkv_weight = torch.cat(
        [
            legacy_state_dict["layers.0.self_attn.q_proj.weight"],
            legacy_state_dict["layers.0.self_attn.k_proj.weight"],
            legacy_state_dict["layers.0.self_attn.v_proj.weight"],
        ]
    )

    assert fuse_qwen3vl_text_qkv(language_model) == 1
    language_model.load_state_dict(legacy_state_dict, strict=True)
    torch.testing.assert_close(language_model.layers[0].self_attn.qkv.weight, expected_qkv_weight)
    assert fuse_qwen3vl_text_qkv(language_model) == 0
