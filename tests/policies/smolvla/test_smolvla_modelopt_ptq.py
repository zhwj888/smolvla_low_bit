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

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.smolvla.modelopt_ptq import (
    LinearPTQDecision,
    build_autoquant_formats,
    build_fixed_nvfp4_config,
)

NVFP4_RECIPE = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": (2, 1)}},
        {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": (2, 1)}},
    ],
    "algorithm": "max",
}
FP8_RECIPE = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": (4, 3)}},
        {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": (4, 3)}},
    ],
    "algorithm": "max",
}


def _modelopt_stub():
    return SimpleNamespace(
        NVFP4_DEFAULT_CFG=deepcopy(NVFP4_RECIPE),
        FP8_DEFAULT_CFG=deepcopy(FP8_RECIPE),
    )


def _decision(name: str, eligible: bool) -> LinearPTQDecision:
    return LinearPTQDecision(
        name=name,
        in_features=32,
        out_features=32,
        trainable=True,
        eligible=eligible,
        reason="eligible" if eligible else "user_excluded",
    )


def test_fixed_nvfp4_rht_rotates_both_operands_and_clears_excluded_layers():
    modelopt = _modelopt_stub()
    config = build_fixed_nvfp4_config(
        modelopt,
        [_decision("body", True), _decision("head", False)],
        calibration_algorithm="mse",
        enable_rht=True,
        rht_block_size=16,
        rht_rotate_fp32=True,
    )

    expected_rotation = {"enable": True, "rotate_fp32": True, "block_size": 16}
    by_name = {entry["quantizer_name"]: entry for entry in config["quant_cfg"]}
    assert by_name["*input_quantizer"]["cfg"]["rotate"] == expected_rotation
    assert by_name["*weight_quantizer"]["cfg"]["rotate"] == expected_rotation
    for entry in config["quant_cfg"]:
        if entry["quantizer_name"] in ("*input_quantizer", "*weight_quantizer"):
            if entry.get("parent_class") == "nn.Linear":
                assert entry["cfg"]["rotate"] == expected_rotation
            else:
                assert entry["cfg"]["rotate"] is False
    assert by_name["head.*"] == {
        "quantizer_name": "head.*",
        "enable": False,
        "cfg": {"rotate": False},
    }
    assert config["algorithm"] == "mse"
    assert "rotate" not in modelopt.NVFP4_DEFAULT_CFG["quant_cfg"][1]["cfg"]


def test_autoquant_applies_rht_only_to_nvfp4_candidate():
    nvfp4, fp8 = build_autoquant_formats(
        _modelopt_stub(),
        calibration_algorithm="max",
        enable_rht=True,
        rht_block_size=None,
    )

    for entry in nvfp4["quant_cfg"]:
        if entry.get("parent_class") == "nn.Linear":
            assert entry["cfg"]["rotate"] == {
                "enable": True,
                "rotate_fp32": False,
                "block_size": None,
            }
    assert all("rotate" not in entry.get("cfg", {}) for entry in fp8["quant_cfg"])


@pytest.mark.parametrize("autoquant", [False, True])
def test_rht_runtime_preserves_non_linear_modules_and_restore(tmp_path, autoquant):
    mtq = pytest.importorskip("modelopt.torch.quantization")
    mto = pytest.importorskip("modelopt.torch.opt")

    def make_model():
        return nn.ModuleDict(
            {
                "body": nn.Linear(32, 32),
                "head": nn.Linear(32, 32),
                "vision_model_norm": nn.LayerNorm(32),
                "embedding": nn.Embedding(16, 32),
                "vision_model_conv": nn.Conv2d(3, 4, 2),
            }
        )

    model = make_model().eval()
    reference = deepcopy(model)
    if autoquant:
        config = build_autoquant_formats(mtq, calibration_algorithm="max", enable_rht=True)[0]
    else:
        config = build_fixed_nvfp4_config(
            mtq,
            [_decision("body", True), _decision("head", False)],
            calibration_algorithm="max",
            enable_rht=True,
        )
    config["algorithm"] = None  # Test conversion/rotation independently of calibration.
    model = mtq.quantize(model, config)
    expected = {"body.input_quantizer", "body.weight_quantizer"}
    if autoquant:
        expected |= {"head.input_quantizer", "head.weight_quantizer"}

    def check(candidate):
        rotating = {
            name
            for name, q in candidate.named_modules()
            if isinstance(q, mtq.nn.TensorQuantizer) and q.rotate_is_enabled
        }
        assert rotating == expected
        inputs = {
            "vision_model_norm": torch.randn(2, 32),
            "embedding": torch.tensor([0, 3]),
            "vision_model_conv": torch.randn(2, 3, 8, 8),
        }
        if not autoquant:
            inputs["head"] = torch.randn(2, 32)
        with torch.no_grad():
            for name, x in inputs.items():
                torch.testing.assert_close(candidate[name](x), reference[name](x), rtol=0, atol=0)

    check(model)
    checkpoint = tmp_path / "rht.pt"
    mto.save(model, checkpoint)
    check(mto.restore(make_model(), checkpoint))


def test_rht_disabled_preserves_original_recipe():
    modelopt = _modelopt_stub()
    nvfp4, fp8 = build_autoquant_formats(modelopt, calibration_algorithm="max", enable_rht=False)
    assert nvfp4 == modelopt.NVFP4_DEFAULT_CFG
    assert fp8 == modelopt.FP8_DEFAULT_CFG


@pytest.mark.parametrize("sweep", [False, True])
def test_sweep_selects_static_weights_and_preserves_fp8_default(sweep):
    modelopt = _modelopt_stub()
    for entry in modelopt.NVFP4_DEFAULT_CFG["quant_cfg"]:
        if entry["quantizer_name"] in ("*weight_quantizer", "*input_quantizer"):
            entry["cfg"]["block_sizes"] = {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}
    original = deepcopy(modelopt.NVFP4_DEFAULT_CFG)
    nvfp4, fp8 = build_autoquant_formats(
        modelopt, calibration_algorithm="mse", fp8_scale_sweep=sweep, enable_rht=True
    )
    for entry in nvfp4["quant_cfg"]:
        name = entry["quantizer_name"]
        if name in ("*weight_quantizer", "*input_quantizer"):
            expected = "static" if sweep and name == "*weight_quantizer" else "dynamic"
            assert entry["cfg"]["block_sizes"]["type"] == expected
    assert nvfp4["algorithm"] == ({"method": "mse", "fp8_scale_sweep": True} if sweep else "mse")
    assert fp8["algorithm"] == ("max" if sweep else "mse")
    assert modelopt.NVFP4_DEFAULT_CFG == original
    if sweep:
        assert fp8 == modelopt.FP8_DEFAULT_CFG


def test_sweep_rejects_max_calibration():
    with pytest.raises(ValueError, match="requires calibration_algorithm='mse'"):
        build_autoquant_formats(_modelopt_stub(), calibration_algorithm="max", fp8_scale_sweep=True)


@pytest.mark.parametrize("autoquant", [False, True])
def test_sweep_cuda_calibration_backward_and_restore(tmp_path, autoquant):
    if not torch.cuda.is_available():
        pytest.skip("NVFP4 calibration requires CUDA")
    mtq = pytest.importorskip("modelopt.torch.quantization")
    mto = pytest.importorskip("modelopt.torch.opt")

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Linear(32, 32)
            self.head = nn.Linear(32, 32)

        def forward(self, x):
            return self.head(self.body(x).tanh())

    torch.manual_seed(17)
    model = TinyModel().cuda().eval()
    x = torch.randn(16, 32, device="cuda")
    head_weight = model.head.weight.detach().clone()
    if autoquant:
        model, _ = mtq.auto_quantize(
            model,
            constraints={"effective_bits": 10.0},
            quantization_formats=build_autoquant_formats(
                mtq, calibration_algorithm="mse", fp8_scale_sweep=True, enable_rht=True
            ),
            disabled_layers=["head"],
            data_loader=[x],
            forward_step=lambda m, batch: m(batch).square().mean(),
            loss_func=lambda output, batch: output,
            num_calib_steps=1,
            num_score_steps=1,
            method="gradient",
        )
    else:
        config = build_fixed_nvfp4_config(
            mtq, [_decision("body", True), _decision("head", False)],
            calibration_algorithm="mse", fp8_scale_sweep=True, enable_rht=True,
        )
        model = mtq.quantize(model, config, forward_loop=lambda m: m(x))

    weight_q = model.body.weight_quantizer
    assert weight_q.is_nvfp4_static
    assert weight_q.amax.numel() == 64
    assert weight_q.global_amax.numel() == 1
    assert model.body.input_quantizer.block_sizes["type"] == "dynamic"
    assert weight_q.rotate_is_enabled and model.body.input_quantizer.rotate_is_enabled
    assert not model.head.weight_quantizer.is_enabled
    assert not model.head.weight_quantizer.rotate_is_enabled
    torch.testing.assert_close(model.head.weight, head_weight, rtol=0, atol=0)
    scales = weight_q.amax.detach().clone()
    before = model.body.weight.detach().clone()
    model.zero_grad(set_to_none=True)
    loss = model(x).square().mean()
    loss.backward()
    assert torch.isfinite(model.body.weight.grad).all()
    assert model.body.weight.grad.abs().sum() > 0
    torch.optim.SGD(model.body.parameters(), lr=0.01).step()
    assert not torch.equal(model.body.weight, before)
    torch.testing.assert_close(weight_q.amax, scales, rtol=0, atol=0)

    checkpoint = tmp_path / "sweep.pt"
    mto.save(model, checkpoint)
    restored = mto.restore(TinyModel().cuda().eval(), checkpoint)
    assert restored.body.weight_quantizer.is_nvfp4_static
    torch.testing.assert_close(restored.body.weight_quantizer.amax, scales, rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)
