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

from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.fp8 import (
    _refresh_blockwise_te_quantizers,
    materialize_fp8_primary_weights,
    pad_batch_dim_for_fp8,
    replace_trainable_linear_layers,
)
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel


class FakeTELinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, params_dtype=None, device=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=params_dtype)
        self.input_shapes = []

    def forward(self, inputs):
        self.input_shapes.append(tuple(inputs.shape))
        return super().forward(inputs)

    def get_extra_state(self):
        return torch.tensor([], dtype=torch.uint8)

    def set_extra_state(self, state):
        self.extra_state = state


class FakeTELayerNormLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        eps=1e-5,
        bias=True,
        normalization="LayerNorm",
        return_layernorm_output=False,
        zero_centered_gamma=False,
        params_dtype=None,
        device=None,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=params_dtype))
        self.bias = (
            nn.Parameter(torch.empty(out_features, device=device, dtype=params_dtype)) if bias else None
        )
        self.layer_norm_weight = nn.Parameter(
            torch.empty(in_features, device=device, dtype=params_dtype)
        )
        self.layer_norm_bias = (
            nn.Parameter(torch.empty(in_features, device=device, dtype=params_dtype))
            if normalization == "LayerNorm"
            else None
        )
        self.eps = eps
        self.normalization = normalization
        self.return_layernorm_output = return_layernorm_output
        self.zero_centered_gamma = zero_centered_gamma
        self.input_shapes = []

    def forward(self, inputs):
        self.input_shapes.append(tuple(inputs.shape))
        if self.normalization == "RMSNorm":
            normalized = inputs.float() * torch.rsqrt(inputs.float().pow(2).mean(-1, keepdim=True) + self.eps)
            gamma = (
                1 + self.layer_norm_weight.float()
                if self.zero_centered_gamma
                else self.layer_norm_weight.float()
            )
            normalized = (normalized * gamma).to(inputs.dtype)
        else:
            normalized = F.layer_norm(
                inputs,
                (inputs.shape[-1],),
                self.layer_norm_weight,
                self.layer_norm_bias,
                self.eps,
            )
        output = F.linear(normalized, self.weight, self.bias)
        return (output, normalized) if self.return_layernorm_output else output

    def get_extra_state(self):
        return torch.tensor([], dtype=torch.uint8)

    def set_extra_state(self, state):
        self.extra_state = state


class FakePrimaryWeightQuantizer:
    def __init__(self):
        self.internal = True
        self.rowwise_usage = None
        self.columnwise_usage = None
        self.force_pow_2_scales = False
        self.amax_epsilon = None
        self.block_scaling_dim = None

    def set_usage(self, *, rowwise, columnwise):
        self.rowwise_usage = rowwise
        self.columnwise_usage = columnwise

    def __call__(self, tensor):
        return tensor.clone()


class FakeQuantizedWeight:
    def __init__(self, *, rowwise=True, columnwise=False):
        self.rowwise = rowwise
        self.columnwise = columnwise
        self.quantizer = None

    def get_usages(self):
        return {"rowwise": self.rowwise, "columnwise": self.columnwise}

    def update_quantizer(self, quantizer):
        self.quantizer = quantizer


class FakePrimaryWeightTELinear(FakeTELinear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.primary_weights_in_fp8 = False
        self.fp8_meta = {}
        self.weight_names = ["weight"]
        self.param_init_meta = {"weight": SimpleNamespace(fp8_meta_index=1)}
        self.weight_quantizer = FakePrimaryWeightQuantizer()
        self.quantizers = {"scaling_fwd": [None, self.weight_quantizer], "scaling_bwd": []}
        self._fp8_workspaces = {"old": object()}
        self.updated_weight_quantizers = False

    def init_fp8_meta_tensors(self, recipe):
        self.fp8_meta["recipe"] = recipe

    def module_setattr(self, name, value):
        setattr(self, name, value)

    def _update_weight_quantizers(self):
        self.updated_weight_quantizers = True


class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, inputs):
        normalized = inputs.float() * torch.rsqrt(inputs.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (normalized * (1 + self.weight.float())).to(inputs.dtype)


class FakeGemmaMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(16, 32, bias=False)
        self.up_proj = nn.Linear(16, 32, bias=False)
        self.down_proj = nn.Linear(32, 16, bias=False)
        self.act_fn = F.silu

    def forward(self, inputs):
        return self.down_proj(self.act_fn(self.gate_proj(inputs)) * self.up_proj(inputs))


class FakeGemmaLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = Gemma3RMSNorm(16)
        self.self_attn = nn.Module()
        self.self_attn.head_dim = 4
        self.self_attn.q_proj = nn.Linear(16, 16, bias=False)
        self.self_attn.k_proj = nn.Linear(16, 16, bias=False)
        self.self_attn.v_proj = nn.Linear(16, 16, bias=False)
        self.post_attention_layernorm = Gemma3RMSNorm(16)
        self.mlp = FakeGemmaMLP()


def fake_te_namespace():
    return SimpleNamespace(Linear=FakeTELinear, LayerNormLinear=FakeTELayerNormLinear)


def test_replace_trainable_linear_layers_preserves_weights_and_frozen_layers():
    model = nn.ModuleDict(
        {
            "trainable": nn.Linear(16, 32),
            "frozen": nn.Linear(16, 32),
        }
    )
    model["frozen"].requires_grad_(False)
    expected_weight = model["trainable"].weight.detach().clone()
    expected_bias = model["trainable"].bias.detach().clone()
    checkpoint = model["trainable"].state_dict()

    replaced = replace_trainable_linear_layers(model, SimpleNamespace(Linear=FakeTELinear))

    assert replaced == ["trainable"]
    assert type(model["trainable"]) is FakeTELinear
    assert type(model["frozen"]) is nn.Linear
    torch.testing.assert_close(model["trainable"].weight, expected_weight)
    torch.testing.assert_close(model["trainable"].bias, expected_bias)
    model["trainable"].load_state_dict(checkpoint, strict=True)


def test_layernormlinear_fusion_is_optional_and_remaining_linears_use_te_linear():
    pure_linear_model = FakeGemmaLayer()
    replace_trainable_linear_layers(pure_linear_model, fake_te_namespace(), fuse_layernorm=False)
    assert type(pure_linear_model.input_layernorm) is Gemma3RMSNorm
    assert type(pure_linear_model.self_attn.q_proj) is FakeTELinear

    fused_model = FakeGemmaLayer()
    replace_trainable_linear_layers(fused_model, fake_te_namespace(), fuse_layernorm=True)
    assert type(fused_model.input_layernorm) is nn.Identity
    assert type(fused_model.self_attn.q_proj) is FakeTELayerNormLinear
    assert type(fused_model.self_attn.k_proj) is FakeTELinear
    assert type(fused_model.self_attn.v_proj) is FakeTELinear
    assert type(fused_model.post_attention_layernorm) is nn.Identity
    assert type(fused_model.mlp.gate_proj) is FakeTELayerNormLinear
    assert type(fused_model.mlp.up_proj) is FakeTELinear
    assert type(fused_model.mlp.down_proj) is FakeTELinear


def test_layernormlinear_fusion_keeps_frozen_layers_in_high_precision():
    model = nn.ModuleDict(
        {
            "trainable": FakeGemmaLayer(),
            "frozen": FakeGemmaLayer(),
        }
    )
    model["frozen"].requires_grad_(False)

    replaced = replace_trainable_linear_layers(
        model,
        fake_te_namespace(),
        fuse_layernorm=True,
    )

    assert any(name.startswith("trainable.") for name in replaced)
    assert not any(name.startswith("frozen.") for name in replaced)
    assert type(model["trainable"].self_attn.q_proj) is FakeTELayerNormLinear
    assert type(model["frozen"].input_layernorm) is Gemma3RMSNorm
    assert type(model["frozen"].self_attn.q_proj) is nn.Linear
    assert type(model["frozen"].mlp.gate_proj) is nn.Linear

    dispatcher = SmolVLMWithExpertModel.__new__(SmolVLMWithExpertModel)
    nn.Module.__init__(dispatcher)
    dispatcher.vlm = SimpleNamespace(
        model=SimpleNamespace(text_model=SimpleNamespace(layers=list(model.values())))
    )
    dispatcher.lm_expert = SimpleNamespace(layers=[])
    dispatcher.configure_input_norm_and_query_dispatch()

    inputs = torch.randn(2, 4, 16)
    fused_norm, fused_query = dispatcher._input_norm_and_query(model["trainable"], inputs)
    expected_frozen_norm = model["frozen"].input_layernorm(inputs)
    expected_frozen_query = model["frozen"].self_attn.q_proj(expected_frozen_norm)
    frozen_norm, frozen_query = dispatcher._input_norm_and_query(model["frozen"], inputs)

    assert fused_norm.shape == inputs.shape
    assert fused_query.shape == inputs.shape
    torch.testing.assert_close(frozen_norm, expected_frozen_norm)
    torch.testing.assert_close(frozen_query, expected_frozen_query)


def test_layernormlinear_fusion_preserves_gemma_forward_and_loads_legacy_checkpoint():
    torch.manual_seed(0)
    model = FakeGemmaLayer()
    inputs = torch.randn(2, 4, 16)
    expected_norm = model.input_layernorm(inputs)
    expected_query = model.self_attn.q_proj(expected_norm)
    mlp_inputs = torch.randn(2, 4, 16)
    expected_mlp = model.mlp(model.post_attention_layernorm(mlp_inputs))
    legacy_checkpoint = model.state_dict()

    replace_trainable_linear_layers(model, fake_te_namespace(), fuse_layernorm=True)
    model.load_state_dict(legacy_checkpoint, strict=True)
    actual_norm, actual_query = SmolVLMWithExpertModel._fused_input_norm_and_query(model, inputs)
    actual_mlp = model.mlp(mlp_inputs)

    torch.testing.assert_close(actual_norm, expected_norm)
    torch.testing.assert_close(actual_query, expected_query)
    torch.testing.assert_close(actual_mlp, expected_mlp)
    assert "self_attn.q_proj.layer_norm_weight" in model.state_dict()
    assert "input_layernorm.weight" not in model.state_dict()


def test_layernormlinear_local_padding_restores_linear_and_norm_outputs():
    model = FakeGemmaLayer()
    replace_trainable_linear_layers(model, fake_te_namespace(), fuse_layernorm=True)
    model.eval()

    normalized, query = SmolVLMWithExpertModel._fused_input_norm_and_query(model, torch.randn(1, 16))

    assert model.self_attn.q_proj.input_shapes == [(8, 16)]
    assert normalized.shape == (1, 16)
    assert query.shape == (1, 16)


def test_pad_batch_dim_for_fp8_appends_zero_samples():
    inputs = torch.randn(57, 32)
    padded = pad_batch_dim_for_fp8(inputs, 7)

    assert padded.shape == (64, 32)
    torch.testing.assert_close(padded[:57], inputs)
    assert torch.count_nonzero(padded[57:]) == 0


def test_refresh_blockwise_te_quantizers_uses_checkpoint_recipe_state():
    weight_quantizer = FakePrimaryWeightQuantizer()
    expected_fwd = [SimpleNamespace(force_pow_2_scales=True), weight_quantizer]
    expected_bwd = [SimpleNamespace(force_pow_2_scales=True)]
    weight = FakeQuantizedWeight(rowwise=True, columnwise=False)
    module = SimpleNamespace(
        fp8_meta={
            "recipe": SimpleNamespace(float8_block_scaling=lambda: True),
            "scaling_fwd": SimpleNamespace(make_quantizers=lambda: expected_fwd),
            "scaling_bwd": SimpleNamespace(make_quantizers=lambda: expected_bwd),
        },
        quantizers={"scaling_fwd": [], "scaling_bwd": []},
        _fp8_workspaces={"stale": object()},
        weight_names=["weight"],
        weight=weight,
        param_init_meta={"weight": SimpleNamespace(fp8_meta_index=1)},
    )

    _refresh_blockwise_te_quantizers(module, incompatible_keys=None)

    assert module.quantizers == {"scaling_fwd": expected_fwd, "scaling_bwd": expected_bwd}
    assert module._fp8_workspaces == {}
    assert weight.quantizer is weight_quantizer
    assert weight_quantizer.rowwise_usage
    assert not weight_quantizer.columnwise_usage
    assert not weight_quantizer.internal


def test_replace_trainable_linear_layers_rejects_unsupported_shape():
    model = nn.Sequential(nn.Linear(15, 32))

    with pytest.raises(ValueError, match="divisible by 16"):
        replace_trainable_linear_layers(model, SimpleNamespace(Linear=FakeTELinear))


def test_materialize_fp8_primary_weights_quantizes_loaded_weight_once():
    model = nn.Sequential(FakePrimaryWeightTELinear(16, 32))
    expected = model[0].weight.detach().clone()
    recipe = SimpleNamespace(
        fp8_quant_fwd_weight=SimpleNamespace(power_2_scale=True, amax_epsilon=0.25),
        w_block_scaling_dim=2,
    )
    fp8 = SimpleNamespace(
        te=SimpleNamespace(
            QuantizedTensor=(),
            Float8BlockQuantizer=FakePrimaryWeightQuantizer,
        ),
        recipe=recipe,
        primary_weights=False,
    )

    converted = materialize_fp8_primary_weights(model, fp8)

    assert converted == ["0.weight"]
    torch.testing.assert_close(model[0].weight, expected)
    assert not model[0].weight.requires_grad
    assert model[0].primary_weights_in_fp8
    assert model[0].weight_quantizer.rowwise_usage
    assert not model[0].weight_quantizer.columnwise_usage
    assert model[0].weight_quantizer.force_pow_2_scales
    assert model[0].weight_quantizer.amax_epsilon == 0.25
    assert model[0].weight_quantizer.block_scaling_dim == 2
    assert model[0]._fp8_workspaces == {}
    assert fp8.primary_weights


@pytest.mark.parametrize(
    ("input_shape", "te_input_shape", "output_shape"),
    [
        ((1, 16), (8, 16), (1, 32)),
        ((2, 3, 16), (8, 16), (2, 3, 32)),
        ((2, 4, 16), (2, 4, 16), (2, 4, 32)),
    ],
)
def test_replaced_te_linear_pads_only_flattened_rows(input_shape, te_input_shape, output_shape):
    model = nn.Sequential(nn.Linear(16, 32))
    replace_trainable_linear_layers(model, SimpleNamespace(Linear=FakeTELinear))
    model.eval()

    output = model(torch.randn(input_shape))

    assert model[0].input_shapes == [te_input_shape]
    assert output.shape == output_shape


@pytest.mark.parametrize("fp8_format", ["invalid", "HYBRID"])
def test_smolvla_fp8_rejects_invalid_format(fp8_format):
    with pytest.raises(ValueError, match="fp8_format"):
        SmolVLAConfig(fp8_format=fp8_format)


def test_smolvla_fp8_rejects_invalid_recipe():
    with pytest.raises(ValueError, match="fp8_recipe"):
        SmolVLAConfig(fp8_recipe="invalid")


def test_smolvla_fp8_rejects_non_positive_amax_history():
    with pytest.raises(ValueError, match="fp8_amax_history_len"):
        SmolVLAConfig(fp8_amax_history_len=0)


def test_smolvla_fp8_allows_torch_compile():
    config = SmolVLAConfig(use_fp8=True, compile_model=True)

    assert config.use_fp8
    assert config.compile_model


def test_smolvla_fp8_primary_weights_require_fp8():
    with pytest.raises(ValueError, match="requires use_fp8"):
        SmolVLAConfig(fp8_primary_weights=True)


def test_smolvla_fp8_primary_weights_reject_torch_compile():
    with pytest.raises(ValueError, match="cannot use torch.compile"):
        SmolVLAConfig(use_fp8=True, fp8_primary_weights=True, compile_model=True)


def test_sample_actions_keeps_inference_batch_unchanged():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model._fp8 = SimpleNamespace(autocast=lambda: nullcontext())
    captured = {}

    def fake_sample_actions(
        self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs
    ):
        captured["batch_sizes"] = [
            images[0].shape[0],
            img_masks[0].shape[0],
            lang_tokens.shape[0],
            lang_masks.shape[0],
            state.shape[0],
            noise.shape[0],
            kwargs["prev_chunk_left_over"].shape[0],
        ]
        return torch.ones(state.shape[0], 2, 32)

    model._sample_actions = MethodType(fake_sample_actions, model)
    actions = model.sample_actions(
        images=[torch.randn(1, 3, 4, 4)],
        img_masks=[torch.ones(1, dtype=torch.bool)],
        lang_tokens=torch.ones(1, 4, dtype=torch.long),
        lang_masks=torch.ones(1, 4, dtype=torch.bool),
        state=torch.randn(1, 32),
        noise=torch.randn(1, 2, 32),
        prev_chunk_left_over=torch.randn(1, 2, 32),
    )

    assert captured["batch_sizes"] == [1] * 7
    assert actions.shape == (1, 2, 32)
