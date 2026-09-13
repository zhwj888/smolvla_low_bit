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
from torch import nn

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching


def test_fp8_and_nvfp4_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        SmolVLAConfig(use_fp8=True, use_nvfp4=True)


def test_nvfp4_training_batch_is_padded_to_runtime_alignment():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model._fp8 = None
    model._nvfp4 = SimpleNamespace(row_alignment=32, autocast=lambda: nullcontext())
    captured = {}

    def fake_forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
    ):
        captured["batch_sizes"] = [
            images[0].shape[0],
            img_masks[0].shape[0],
            lang_tokens.shape[0],
            lang_masks.shape[0],
            state.shape[0],
            actions.shape[0],
            noise.shape[0],
            time.shape[0],
        ]
        return torch.ones(actions.shape[0], 2, 32)

    model._forward = MethodType(fake_forward, model)
    losses = model.forward(
        images=[torch.randn(17, 3, 4, 4)],
        img_masks=[torch.ones(17, dtype=torch.bool)],
        lang_tokens=torch.ones(17, 4, dtype=torch.long),
        lang_masks=torch.ones(17, 4, dtype=torch.bool),
        state=torch.randn(17, 32),
        actions=torch.randn(17, 2, 32),
        noise=torch.randn(17, 2, 32),
        time=torch.randn(17),
    )

    assert captured["batch_sizes"] == [32] * 8
    assert losses.shape == (17, 2, 32)


def test_nvfp4_config_keeps_gb10_compatible_recipe_defaults():
    config = SmolVLAConfig(use_nvfp4=True)

    assert config.nvfp4_disable_stochastic_rounding
    assert not config.nvfp4_disable_rht
    assert not config.nvfp4_disable_2d_quantization
