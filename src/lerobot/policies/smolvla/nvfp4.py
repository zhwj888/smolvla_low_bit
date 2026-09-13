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

"""Transformer Engine NVFP4 integration for SmolVLA.

This module deliberately reuses the already-tested Transformer Engine module conversion from the
FP8 integration without changing FP8's recipe selection, serialization, or execution path.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import MethodType, ModuleType

import torch
from torch import nn

from lerobot.utils.import_utils import require_package

from .configuration_smolvla import SmolVLAConfig
from .fp8 import replace_trainable_linear_layers

NVFP4_DEFAULT_ROW_ALIGNMENT = 16


def pad_batch_dim_for_nvfp4(tensor: torch.Tensor, padding: int) -> torch.Tensor:
    """Append zero samples so flattened training GEMMs meet the runtime row alignment."""
    return torch.cat((tensor, tensor.new_zeros((padding, *tensor.shape[1:]))), dim=0)


def _te_linear_forward_with_nvfp4_row_padding(
    module: nn.Module,
    inp: torch.Tensor,
    *args,
    **kwargs,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Pad an individual inference GEMM to the NVFP4 block alignment and restore its shape."""
    unpadded_forward = module._te_unpadded_forward
    if module._nvfp4_force_bf16_input and inp.is_floating_point() and inp.dtype is not torch.bfloat16:
        inp = inp.to(dtype=torch.bfloat16)
    row_alignment = module._nvfp4_row_alignment
    if module.training or (inp.ndim > 1 and inp.shape[0] % row_alignment == 0):
        return unpadded_forward(inp, *args, **kwargs)

    input_shape = inp.shape
    rows = inp.numel() // input_shape[-1]
    row_padding = (-rows) % row_alignment
    if not row_padding:
        return unpadded_forward(inp, *args, **kwargs)

    flattened = inp.reshape(rows, input_shape[-1])
    padded = torch.cat(
        (flattened, flattened.new_zeros((row_padding, *flattened.shape[1:]))),
        dim=0,
    )
    output = unpadded_forward(padded, *args, **kwargs)
    padded_rows = rows + row_padding

    def restore_shape(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 0 or tensor.numel() // tensor.shape[-1] != padded_rows:
            return tensor
        return tensor.reshape(padded_rows, tensor.shape[-1])[:rows].reshape(
            *input_shape[:-1], tensor.shape[-1]
        )

    if isinstance(output, tuple):
        return tuple(restore_shape(item) if isinstance(item, torch.Tensor) else item for item in output)
    return restore_shape(output)


def _enable_nvfp4_row_padding(module: nn.Module) -> None:
    """Replace FP8's local wrapper with the NVFP4 runtime-alignment wrapper."""
    module.forward = MethodType(_te_linear_forward_with_nvfp4_row_padding, module)


@dataclass
class TransformerEngineNVFP4:
    """Own the NVFP4 recipe and the converted Transformer Engine modules."""

    te: ModuleType
    recipe: object
    replaced_modules: list[str]
    fused_modules: list[str]
    row_alignment: int
    skipped_modules: list[str]

    def autocast(self) -> AbstractContextManager:
        return self.te.autocast(enabled=True, recipe=self.recipe)


def setup_transformer_engine_nvfp4(
    module: nn.Module,
    config: SmolVLAConfig,
) -> TransformerEngineNVFP4:
    """Fuse trainable Norm→Linear paths and configure eligible GEMMs for NVFP4 training."""
    require_package("transformer-engine", extra="smolvla-fp8", import_name="transformer_engine")

    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import NVFP4BlockScaling

    nvfp4_available, reason = te.is_nvfp4_available(return_reason=True)
    if not nvfp4_available:
        raise RuntimeError(f"Transformer Engine NVFP4 is not available on this system: {reason}")

    compute_capability = None
    if torch.cuda.is_available():
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] == 12 and not config.nvfp4_disable_stochastic_rounding:
            raise RuntimeError(
                "NVFP4 stochastic rounding is not valid on SM12x with Transformer Engine 2.16. "
                "Set nvfp4_disable_stochastic_rounding=True."
            )

    recipe = NVFP4BlockScaling(
        disable_rht=config.nvfp4_disable_rht,
        disable_stochastic_rounding=config.nvfp4_disable_stochastic_rounding,
        disable_2d_quantization=config.nvfp4_disable_2d_quantization,
    )
    # TE 2.16 reports NVFP4 availability on SM12x, but GB10's cuBLASLt has no NVFP4 algorithm for
    # feature dimensions that are only 16-aligned (SmolVLA's 720-wide expert is one such case).
    # Temporarily hide those modules from the shared converter, then restore their trainability.
    feature_alignment = (
        32
        if compute_capability is not None and compute_capability[0] == 12
        else NVFP4_DEFAULT_ROW_ALIGNMENT
    )
    row_alignment = feature_alignment
    skipped_modules: list[str] = []
    temporarily_frozen: list[tuple[nn.Linear, list[bool]]] = []
    seen_linears: set[int] = set()
    for module_name, child in module.named_modules():
        if type(child) is not nn.Linear or id(child) in seen_linears:
            continue
        seen_linears.add(id(child))
        if not any(parameter.requires_grad for parameter in child.parameters(recurse=False)):
            continue
        if child.in_features % feature_alignment or child.out_features % feature_alignment:
            requires_grad = [
                parameter.requires_grad for parameter in child.parameters(recurse=False)
            ]
            temporarily_frozen.append((child, requires_grad))
            child.requires_grad_(False)
            skipped_modules.append(module_name)

    try:
        replaced_modules = replace_trainable_linear_layers(
            module,
            te,
            include_frozen=False,
            primary_weights=False,
            recipe=recipe,
            materialize_primary_weights=False,
            fuse_layernorm=True,
        )
    finally:
        for child, requires_grad in temporarily_frozen:
            for parameter, original_requires_grad in zip(
                child.parameters(recurse=False), requires_grad, strict=True
            ):
                parameter.requires_grad_(original_requires_grad)
    if not replaced_modules:
        raise RuntimeError("SmolVLA NVFP4 was enabled, but no eligible Linear modules were found.")

    for module_name in replaced_modules:
        converted_module = module.get_submodule(module_name)
        converted_module._nvfp4_row_alignment = row_alignment
        converted_module._nvfp4_force_bf16_input = not config.nvfp4_disable_rht
        _enable_nvfp4_row_padding(converted_module)

    fused_modules = [
        name
        for name, child in module.named_modules()
        if getattr(child, "_te_fused_norm_linear", False)
    ]
    if not fused_modules:
        raise RuntimeError("SmolVLA NVFP4 requires LayerNormLinear fusion, but no fused modules were found.")

    return TransformerEngineNVFP4(
        te=te,
        skipped_modules=skipped_modules,
        recipe=recipe,
        row_alignment=row_alignment,
        replaced_modules=replaced_modules,
        fused_modules=fused_modules,
    )
