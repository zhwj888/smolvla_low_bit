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

"""Transformer Engine FP8 integration for SmolVLA."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import warnings
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import MethodType, ModuleType
from typing import Any

import torch
from torch import nn

from lerobot.utils.import_utils import require_package

from .configuration_smolvla import SmolVLAConfig


FP8_WEIGHTS_NAME = "model_fp8.pt"
FP8_MANIFEST_NAME = "fp8_manifest.json"
FP8_CHECKPOINT_FORMAT = "lerobot.smolvla.transformer_engine.fp8.v1"
FP8_DIAGNOSTICS_ENV = "LEROBOT_FP8_DIAGNOSTICS"


def _fp8_diagnostics_enabled() -> bool:
    """Return whether per-module FP8 replacement diagnostics are enabled."""
    return os.getenv(FP8_DIAGNOSTICS_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


def _synchronize_fp8_replacement(linear: nn.Linear) -> None:
    """Surface an asynchronous CUDA construction error at the module that launched it."""
    if _fp8_diagnostics_enabled() and linear.weight.device.type == "cuda":
        torch.cuda.synchronize(linear.weight.device)


def _supply_default_te_extra_state(
    module: nn.Module,
    state_dict: dict,
    prefix: str,
    *args,
) -> None:
    """Let a non-TE checkpoint initialize TE's FP8 metadata from the new module."""
    extra_state_key = f"{prefix}_extra_state"
    if extra_state_key not in state_dict:
        state_dict[extra_state_key] = module.get_extra_state()


def _migrate_legacy_norm_state(legacy_norm_path: str, fused_linear_path: str):
    """Build a load hook that moves an unfused norm state into LayerNormLinear."""

    def hook(module: nn.Module, state_dict: dict, prefix: str, *args) -> None:
        del module, args
        for legacy_name, fused_name in (
            ("weight", "layer_norm_weight"),
            ("bias", "layer_norm_bias"),
        ):
            legacy_key = f"{prefix}{legacy_norm_path}.{legacy_name}"
            fused_key = f"{prefix}{fused_linear_path}.{fused_name}"
            legacy_value = state_dict.pop(legacy_key, None)
            if fused_key not in state_dict and legacy_value is not None:
                state_dict[fused_key] = legacy_value

    return hook


def _synchronize_primary_weight_quantizers(module: nn.Module) -> None:
    """Attach rebuilt module quantizers to serialized FP8-primary weights.

    TE cannot obtain a weight quantizer through ``_update_weight_quantizers`` outside FP8
    autocast. Load-state post-hooks run outside autocast, so update the quantized storage directly
    while preserving whether its rowwise and columnwise payloads are actually present.
    """
    for weight_name in getattr(module, "weight_names", ()):
        weight = getattr(module, weight_name, None)
        if not hasattr(weight, "get_usages") or not hasattr(weight, "update_quantizer"):
            continue

        init_meta = module.param_init_meta.get(weight_name)
        quantizer_index = None if init_meta is None else init_meta.fp8_meta_index
        if quantizer_index is None:
            continue
        quantizer = module.quantizers["scaling_fwd"][quantizer_index]
        if quantizer is None:
            continue

        usages = weight.get_usages()
        quantizer.set_usage(
            rowwise=bool(usages["rowwise"]),
            columnwise=bool(usages["columnwise"]),
        )
        quantizer.internal = False
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Quantizer is being updated, this may affect model behavior",
            )
            weight.update_quantizer(quantizer)


def _refresh_blockwise_te_quantizers(module: nn.Module, incompatible_keys) -> None:
    """Rebuild stateless blockwise quantizers after TE restores checkpoint metadata.

    Transformer Engine 2.16 can restore a blockwise recipe whose power-of-two scale settings do
    not propagate to the quantizers already cached by ``Linear``. Blackwell emulates this recipe
    with MXFP8 kernels and rejects those stale quantizers. Rebuilding from the restored recipe
    state is lossless because blockwise quantizers do not carry amax history.
    """
    del incompatible_keys
    fp8_meta = getattr(module, "fp8_meta", None)
    if not fp8_meta:
        return
    recipe = fp8_meta.get("recipe")
    if recipe is None or not recipe.float8_block_scaling():
        return

    for key in ("scaling_fwd", "scaling_bwd"):
        recipe_state = fp8_meta.get(key)
        if recipe_state is not None:
            module.quantizers[key] = recipe_state.make_quantizers()
    module._fp8_workspaces.clear()
    _synchronize_primary_weight_quantizers(module)


def pad_batch_dim_for_fp8(tensor: torch.Tensor, padding: int) -> torch.Tensor:
    """Append zero samples so the batch dimension meets TE FP8 alignment."""
    return torch.cat((tensor, tensor.new_zeros((padding, *tensor.shape[1:]))), dim=0)


def _te_linear_forward_with_row_padding(
    module: nn.Module,
    inp: torch.Tensor,
    *args,
    **kwargs,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Pad only a Linear's flattened row dimension for TE FP8 execution.

    Transformer Engine flattens every leading input dimension into the GEMM ``M`` dimension and
    requires that dimension to be divisible by 8. Padding the policy batch globally satisfies that
    constraint, but makes batch-1 inference run the entire vision-language model eight times. This
    wrapper confines the padding to an individual GEMM and immediately removes the padded rows.
    """
    unpadded_forward = module._te_unpadded_forward
    # Training already pads the policy batch once, which makes every flattened GEMM row dimension
    # aligned. The common inference case with a batch divisible by 8 is aligned for the same reason.
    # Bypass all shape bookkeeping on those hot paths.
    if module.training or (inp.ndim > 1 and inp.shape[0] % 8 == 0):
        return unpadded_forward(inp, *args, **kwargs)

    input_shape = inp.shape
    rows = inp.numel() // input_shape[-1]
    row_padding = (-rows) % 8

    if not row_padding:
        return unpadded_forward(inp, *args, **kwargs)

    flattened = inp.reshape(rows, input_shape[-1])
    padded = torch.cat(
        (flattened, flattened.new_zeros((row_padding, input_shape[-1]))),
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


def _enable_te_linear_row_padding(linear: nn.Module) -> None:
    """Install local FP8 row padding without changing module parameters or state-dict keys."""
    linear._te_unpadded_forward = linear.forward
    linear.forward = MethodType(_te_linear_forward_with_row_padding, linear)


def _make_te_linear(
    linear: nn.Linear,
    te: ModuleType,
    *,
    primary_weights: bool = False,
    recipe: object | None = None,
    materialize_primary_weights: bool = False,
) -> nn.Module:
    """Create a Transformer Engine Linear with the same parameters as a torch Linear."""
    quantized_init = (
        te.quantized_model_init(enabled=True, recipe=recipe) if primary_weights else nullcontext()
    )
    # no_grad makes TE allocate only the row-wise representation required by inference. Enabling
    # gradients here would also allocate a column-wise copy for weight-gradient GEMMs.
    with torch.no_grad(), quantized_init:
        replacement = te.Linear(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            params_dtype=linear.weight.dtype,
            device=linear.weight.device if not primary_weights or materialize_primary_weights else "meta",
        )
        # An FP8-only module is always followed by loading either the source safetensors during
        # export or model_fp8.pt during inference. Avoid quantizing the temporary initialization a
        # second time here; TE 2.16 can race on that redundant in-place update for large tied heads.
        if not primary_weights:
            replacement.weight.copy_(linear.weight)
        if linear.bias is not None and (not primary_weights or materialize_primary_weights):
            replacement.bias.copy_(linear.bias)

    replacement.weight.requires_grad_(linear.weight.requires_grad and not primary_weights)
    if linear.bias is not None:
        replacement.bias.requires_grad_(linear.bias.requires_grad and not primary_weights)
    replacement.train(linear.training)
    _enable_te_linear_row_padding(replacement)
    replacement.register_load_state_dict_pre_hook(_supply_default_te_extra_state)
    replacement.register_load_state_dict_post_hook(_refresh_blockwise_te_quantizers)
    return replacement


def _te_norm_spec(norm: nn.Module) -> tuple[str, bool] | None:
    """Return the TE normalization name and gamma convention for supported norms."""
    norm_name = type(norm).__name__
    if norm_name == "Gemma3RMSNorm":
        # Gemma3 evaluates RMSNorm(x) * (1 + weight), so its gamma is zero-centered.
        return "RMSNorm", True
    if norm_name == "LlamaRMSNorm":
        return "RMSNorm", False
    if isinstance(norm, nn.RMSNorm):
        return "RMSNorm", False
    if isinstance(norm, nn.LayerNorm):
        return "LayerNorm", False
    return None


def _make_te_layernorm_linear(
    norm: nn.Module,
    linear: nn.Linear,
    te: ModuleType,
    *,
    primary_weights: bool = False,
    recipe: object | None = None,
    materialize_primary_weights: bool = False,
) -> nn.Module:
    """Fuse a supported norm and Linear while preserving their learned parameters."""
    norm_spec = _te_norm_spec(norm)
    if norm_spec is None:
        raise TypeError(f"Unsupported normalization for TE fusion: {type(norm).__name__}.")
    normalization, zero_centered_gamma = norm_spec
    quantized_init = (
        te.quantized_model_init(enabled=True, recipe=recipe) if primary_weights else nullcontext()
    )
    device = linear.weight.device if not primary_weights or materialize_primary_weights else "meta"
    eps = float(getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-5)))
    with torch.no_grad(), quantized_init:
        replacement = te.LayerNormLinear(
            linear.in_features,
            linear.out_features,
            eps=eps,
            bias=linear.bias is not None,
            normalization=normalization,
            return_layernorm_output=True,
            zero_centered_gamma=zero_centered_gamma,
            params_dtype=linear.weight.dtype,
            device=device,
        )
        if not primary_weights:
            replacement.weight.copy_(linear.weight)
        if linear.bias is not None and (not primary_weights or materialize_primary_weights):
            replacement.bias.copy_(linear.bias)
        if not primary_weights or materialize_primary_weights:
            replacement.layer_norm_weight.copy_(norm.weight)
            norm_bias = getattr(norm, "bias", None)
            if norm_bias is not None:
                replacement.layer_norm_bias.copy_(norm_bias)

    replacement.weight.requires_grad_(linear.weight.requires_grad and not primary_weights)
    if linear.bias is not None:
        replacement.bias.requires_grad_(linear.bias.requires_grad and not primary_weights)
    replacement.layer_norm_weight.requires_grad_(norm.weight.requires_grad and not primary_weights)
    norm_bias = getattr(norm, "bias", None)
    if norm_bias is not None:
        replacement.layer_norm_bias.requires_grad_(norm_bias.requires_grad and not primary_weights)
    replacement.train(linear.training)
    replacement._te_fused_norm_linear = True
    _enable_te_linear_row_padding(replacement)
    replacement.register_load_state_dict_pre_hook(_supply_default_te_extra_state)
    replacement.register_load_state_dict_post_hook(_refresh_blockwise_te_quantizers)
    return replacement


def _fused_mlp_forward(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Run Gemma/SigLIP MLPs whose first projection includes the preceding norm."""
    if getattr(getattr(module, "gate_proj", None), "_te_fused_norm_linear", False):
        hidden_states = hidden_states.to(module.gate_proj.layer_norm_weight.dtype)
        gate, normalized = module.gate_proj(hidden_states)
        return module.down_proj(module.act_fn(gate) * module.up_proj(normalized))
    hidden_states = hidden_states.to(module.fc1.layer_norm_weight.dtype)
    projected, _ = module.fc1(hidden_states)
    return module.fc2(module.activation_fn(projected))


def _siglip2_attention_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Preserve SigLIP2 attention while obtaining Q and normalized K/V input from one fused op."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.siglip2.modeling_siglip2 import eager_attention_forward

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    hidden_states = hidden_states.to(module.q_proj.layer_norm_weight.dtype)
    queries, normalized = module.q_proj(hidden_states)
    queries = queries.view(hidden_shape).transpose(1, 2)
    keys = module.k_proj(normalized).view(hidden_shape).transpose(1, 2)
    values = module.v_proj(normalized).view(hidden_shape).transpose(1, 2)
    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        module.config._attn_implementation, eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        module,
        queries,
        keys,
        values,
        attention_mask,
        is_causal=module.is_causal,
        scaling=module.scale,
        dropout=0.0 if not module.training else module.dropout,
        **kwargs,
    )
    attn_output = module.out_proj(attn_output.reshape(*input_shape, -1).contiguous())
    return attn_output, attn_weights


def _replace_norm_linear_pairs(
    module: nn.Module,
    te: ModuleType,
    *,
    include_frozen: bool,
    primary_weights: bool,
    recipe: object | None,
    materialize_primary_weights: bool,
) -> list[str]:
    """Fuse every supported Norm→Linear pair whose forward path is explicitly adapted."""
    replaced: list[str] = []
    fusion_specs = (
        ("input_layernorm", "self_attn", "q_proj"),
        ("post_attention_layernorm", "mlp", "gate_proj"),
        ("layer_norm1", "self_attn", "q_proj"),
        ("layer_norm2", "mlp", "fc1"),
        ("layernorm", "mlp", "fc1"),
    )
    for layer_name, layer in list(module.named_modules()):
        for norm_name, projection_parent_name, projection_name in fusion_specs:
            norm = getattr(layer, norm_name, None)
            projection_parent = getattr(layer, projection_parent_name, None)
            linear = getattr(projection_parent, projection_name, None)
            if _te_norm_spec(norm) is None or type(linear) is not nn.Linear:
                continue
            if not include_frozen and not any(
                parameter.requires_grad for parameter in linear.parameters(recurse=False)
            ):
                continue
            qualified_linear_name = ".".join(
                part for part in (layer_name, projection_parent_name, projection_name) if part
            )
            if linear.in_features % 16 != 0 or linear.out_features % 16 != 0:
                raise ValueError(
                    "Transformer Engine FP8 requires LayerNormLinear in_features and out_features "
                    f"to be divisible by 16, but {qualified_linear_name} has shape "
                    f"({linear.out_features}, {linear.in_features})."
                )
            if _fp8_diagnostics_enabled():
                logging.info(
                    "Creating TE LayerNormLinear %s with weight_shape=(%d, %d), dtype=%s, "
                    "device=%s, primary_weights=%s, materialize_primary_weights=%s.",
                    qualified_linear_name,
                    linear.out_features,
                    linear.in_features,
                    linear.weight.dtype,
                    linear.weight.device,
                    primary_weights,
                    materialize_primary_weights,
                )
            try:
                replacement = _make_te_layernorm_linear(
                    norm,
                    linear,
                    te,
                    primary_weights=primary_weights,
                    recipe=recipe,
                    materialize_primary_weights=materialize_primary_weights,
                )
                _synchronize_fp8_replacement(linear)
            except Exception as error:
                raise RuntimeError(
                    f"Failed to replace Norm->Linear {qualified_linear_name} with weight shape "
                    f"({linear.out_features}, {linear.in_features}), dtype={linear.weight.dtype}, "
                    f"device={linear.weight.device}, primary_weights={primary_weights}, "
                    f"materialize_primary_weights={materialize_primary_weights}."
                ) from error
            fused_linear_path = f"{projection_parent_name}.{projection_name}"
            layer.register_load_state_dict_pre_hook(
                _migrate_legacy_norm_state(norm_name, fused_linear_path)
            )
            setattr(projection_parent, projection_name, replacement)
            setattr(layer, norm_name, nn.Identity())
            if projection_name in ("gate_proj", "fc1"):
                projection_parent.forward = MethodType(_fused_mlp_forward, projection_parent)
            elif norm_name == "layer_norm1":
                projection_parent.forward = MethodType(_siglip2_attention_forward, projection_parent)
            replaced.append(qualified_linear_name)
    return replaced


def replace_trainable_linear_layers(
    module: nn.Module,
    te: ModuleType,
    *,
    include_frozen: bool = False,
    primary_weights: bool = False,
    recipe: object | None = None,
    materialize_primary_weights: bool = False,
    fuse_layernorm: bool = False,
) -> list[str]:
    """Replace supported Norm→Linear pairs and all remaining Linear modules with TE modules.

    Transformer Engine FP8 GEMMs require both feature dimensions to be divisible by 16. A clear
    error is raised instead of silently leaving part of the trainable model outside FP8.
    """
    replaced = (
        _replace_norm_linear_pairs(
            module,
            te,
            include_frozen=include_frozen,
            primary_weights=primary_weights,
            recipe=recipe,
            materialize_primary_weights=materialize_primary_weights,
        )
        if fuse_layernorm
        else []
    )
    replacements: dict[int, nn.Module] = {}

    def visit(parent: nn.Module, prefix: str) -> None:
        for child_name, child in list(parent._modules.items()):
            if child is None:
                continue
            qualified_name = f"{prefix}.{child_name}" if prefix else child_name

            if id(child) in replacements:
                parent._modules[child_name] = replacements[id(child)]
                continue

            is_linear = type(child) is nn.Linear
            is_unused_lm_head = qualified_name.endswith("lm_head")
            should_replace = is_linear and not is_unused_lm_head and (
                include_frozen
                or any(parameter.requires_grad for parameter in child.parameters(recurse=False))
            )
            if should_replace:
                if child.in_features % 16 != 0 or child.out_features % 16 != 0:
                    raise ValueError(
                        "Transformer Engine FP8 requires Linear in_features and out_features to be "
                        f"divisible by 16, but {qualified_name} has shape "
                        f"({child.out_features}, {child.in_features})."
                    )
                try:
                    replacement = _make_te_linear(
                        child,
                        te,
                        primary_weights=primary_weights,
                        recipe=recipe,
                        materialize_primary_weights=materialize_primary_weights,
                    )
                except Exception as error:
                    raise RuntimeError(
                        f"Failed to replace Linear {qualified_name} with shape "
                        f"({child.out_features}, {child.in_features}) using Transformer Engine."
                    ) from error
                replacements[id(child)] = replacement
                parent._modules[child_name] = replacement
                replaced.append(qualified_name)
                continue

            visit(child, qualified_name)

    visit(module, "")
    return replaced


@dataclass
class TransformerEngineFP8:
    """Own the TE recipe and create one FP8 autocast region per training forward."""

    te: ModuleType
    recipe: object
    replaced_modules: list[str]
    fused_modules: list[str]
    primary_weights: bool = False

    def autocast(self) -> AbstractContextManager:
        return self.te.autocast(enabled=True, recipe=self.recipe)


def setup_transformer_engine_fp8(module: nn.Module, config: SmolVLAConfig) -> TransformerEngineFP8:
    """Validate FP8 support, replace trainable Linear modules, and construct the recipe."""
    require_package("transformer-engine", extra="smolvla-fp8", import_name="transformer_engine")

    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import (
        DelayedScaling,
        Float8BlockScaling,
        Float8CurrentScaling,
        Format,
        MXFP8BlockScaling,
    )

    availability_check = {
        "delayed": te.is_fp8_available,
        "current": te.is_fp8_available,
        "blockwise": te.is_fp8_block_scaling_available,
        "mxfp8": te.is_mxfp8_available,
    }[config.fp8_recipe]
    fp8_available, reason = availability_check(return_reason=True)
    if not fp8_available:
        raise RuntimeError(
            f"Transformer Engine FP8 recipe {config.fp8_recipe!r} is not available on this system: {reason}"
        )

    fp8_format = Format.HYBRID if config.fp8_format == "hybrid" else Format.E4M3
    if config.fp8_recipe == "delayed":
        recipe = DelayedScaling(
            fp8_format=fp8_format,
            amax_history_len=config.fp8_amax_history_len,
            amax_compute_algo="max",
        )
    elif config.fp8_recipe == "current":
        recipe = Float8CurrentScaling(fp8_format=fp8_format)
    elif config.fp8_recipe == "blockwise":
        recipe = Float8BlockScaling(fp8_format=fp8_format)
    else:
        recipe = MXFP8BlockScaling(fp8_format=fp8_format)

    replaced_modules = replace_trainable_linear_layers(
        module,
        te,
        # Fusion changes only the implementation of eligible trainable Norm->Linear pairs. Keep
        # frozen modules in their original high precision so fused and pure-Linear modes have the
        # same FP8 coverage and inference matches the precision used for those modules in training.
        # The unused lm_head is always skipped by the replacement function as well.
        include_frozen=False,
        primary_weights=config.fp8_primary_weights,
        recipe=recipe,
        materialize_primary_weights=bool(
            getattr(config, "_fp8_materialize_primary_weights", False)
        ),
        fuse_layernorm=config.fp8_fuse_layernorm,
    )
    if not replaced_modules:
        raise RuntimeError("SmolVLA FP8 was enabled, but no eligible Linear modules were found.")
    fused_modules = [
        name
        for name, child in module.named_modules()
        if getattr(child, "_te_fused_norm_linear", False)
    ]
    return TransformerEngineFP8(
        te=te,
        recipe=recipe,
        replaced_modules=replaced_modules,
        fused_modules=fused_modules,
        primary_weights=config.fp8_primary_weights,
    )


def materialize_fp8_primary_weights(
    module: nn.Module,
    fp8: TransformerEngineFP8,
) -> list[str]:
    """Convert loaded high-precision TE weights to FP8-primary parameters once.

    Exporters must call this only after the final high-precision checkpoint has been loaded. This
    deferred path avoids asking ``quantized_model_init`` to quantize temporary random parameters
    while the model is being constructed, and mirrors TE's own parameter materialization logic.
    Inference only needs the row-wise representation used by the forward GEMM.
    """
    converted: list[str] = []
    quantized_tensor_type = fp8.te.QuantizedTensor

    with torch.no_grad():
        for module_name, child in module.named_modules():
            if not hasattr(child, "primary_weights_in_fp8"):
                continue

            # Ensure the weight quantizers correspond to the recipe selected for the export,
            # rather than relying on whichever extra-state happened to be present in the source.
            child.fp8_meta["recipe"] = fp8.recipe
            child.init_fp8_meta_tensors(fp8.recipe)

            for weight_name in child.weight_names:
                weight = getattr(child, weight_name)
                qualified_name = f"{module_name}.{weight_name}" if module_name else weight_name
                if isinstance(weight, quantized_tensor_type):
                    converted.append(qualified_name)
                    continue

                init_meta = child.param_init_meta.get(weight_name)
                quantizer_index = None if init_meta is None else init_meta.fp8_meta_index
                if quantizer_index is None:
                    raise RuntimeError(
                        f"Transformer Engine module {module_name!r} has no FP8 quantizer index "
                        f"for parameter {weight_name!r}."
                    )
                quantizer = child.quantizers["scaling_fwd"][quantizer_index]
                if quantizer is None:
                    raise RuntimeError(
                        f"Transformer Engine module {module_name!r} did not initialize its "
                        f"weight quantizer for recipe {type(fp8.recipe).__name__}."
                    )

                # TE 2.16 may retain Qparams from a checkpoint's cached quantizer even after the
                # module recipe has been replaced. Keep the actual weight quantizer aligned with
                # the selected export recipe. This is mandatory on Blackwell, where blockwise FP8
                # is represented through an MXFP8-compatible layout and scales must be powers of
                # two. Hopper accepts either setting, so synchronizing here also makes exports
                # architecture-independent.
                blockwise_quantizer_type = getattr(fp8.te, "Float8BlockQuantizer", ())
                if isinstance(quantizer, blockwise_quantizer_type):
                    weight_qparams = fp8.recipe.fp8_quant_fwd_weight
                    quantizer.force_pow_2_scales = weight_qparams.power_2_scale
                    quantizer.amax_epsilon = weight_qparams.amax_epsilon
                    quantizer.block_scaling_dim = fp8.recipe.w_block_scaling_dim

                # H20 training exercises the rowwise+columnwise square-blockwise kernel, while
                # TE 2.16's FP8-primary constructor requests rowwise-only under no_grad. Materialize
                # through the training-proven path on Hopper, then discard the columnwise copy so
                # the exported inference payload still uses one byte per weight element.
                hopper_blockwise = (
                    weight.device.type == "cuda"
                    and torch.cuda.get_device_capability(weight.device)[0] == 9
                    and isinstance(quantizer, fp8.te.Float8BlockQuantizer)
                )
                quantizer.set_usage(rowwise=True, columnwise=hopper_blockwise)
                quantizer.internal = False
                try:
                    quantized_weight = quantizer(weight.detach())
                    if hopper_blockwise:
                        quantized_weight.update_usage(
                            rowwise_usage=True,
                            columnwise_usage=False,
                        )
                    if weight.device.type == "cuda":
                        torch.cuda.synchronize(weight.device)
                except Exception as error:
                    raise RuntimeError(
                        f"Failed to materialize FP8-primary weight {qualified_name} with shape "
                        f"{tuple(weight.shape)}, dtype={weight.dtype}, device={weight.device}."
                    ) from error

                quantized_weight = nn.Parameter(quantized_weight, requires_grad=False)
                child.module_setattr(weight_name, quantized_weight)
                converted.append(qualified_name)

            child.primary_weights_in_fp8 = True
            child._fp8_workspaces.clear()
            _synchronize_primary_weight_quantizers(child)

    if not converted:
        raise RuntimeError("No Transformer Engine weights were found to materialize as FP8-primary.")
    fp8.primary_weights = True
    return converted


def _version_major_minor(version: str) -> str:
    """Return the compatibility-significant major.minor portion of a package version."""
    return ".".join(version.split(".")[:2])


def _build_fp8_manifest(module: nn.Module, config: SmolVLAConfig) -> dict[str, Any]:
    """Describe and validate a SmolVLA model whose TE parameters are stored in FP8."""
    import transformer_engine
    import transformer_engine.pytorch as te

    te_modules = [
        (name, child)
        for name, child in module.named_modules()
        if hasattr(child, "primary_weights_in_fp8")
    ]
    if not te_modules:
        raise RuntimeError("Cannot export FP8-only checkpoint: no Transformer Engine modules were found.")
    fused_modules = [
        name for name, child in te_modules if getattr(child, "_te_fused_norm_linear", False)
    ]

    non_quantized_te_modules = [
        name for name, child in te_modules if not bool(child.primary_weights_in_fp8)
    ]
    if non_quantized_te_modules:
        preview = ", ".join(non_quantized_te_modules[:5])
        raise RuntimeError(
            "Cannot export FP8-only checkpoint because some TE modules still have high-precision "
            f"primary weights: {preview}."
        )

    quantized_parameters = [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if isinstance(parameter, te.QuantizedTensor)
    ]
    if not quantized_parameters:
        raise RuntimeError("Cannot export FP8-only checkpoint: no quantized parameters were found.")

    remaining_torch_linears = [
        name for name, child in module.named_modules() if type(child) is nn.Linear
    ]

    return {
        "format": FP8_CHECKPOINT_FORMAT,
        "policy_type": "smolvla",
        "torch_version": torch.__version__,
        "transformer_engine_version": transformer_engine.__version__,
        "recipe": config.fp8_recipe,
        "fp8_format": config.fp8_format,
        "fuse_layernorm": config.fp8_fuse_layernorm,
        "primary_weights": True,
        "te_module_count": len(te_modules),
        "layernormlinear_module_count": len(fused_modules),
        "layernormlinear_module_names": fused_modules,
        "quantized_parameter_count": len(quantized_parameters),
        "quantized_parameter_names": [name for name, _ in quantized_parameters],
        "high_precision_linear_count": len(remaining_torch_linears),
        "high_precision_linear_names": remaining_torch_linears,
    }


def save_fp8_checkpoint(module: nn.Module, directory: str | Path, config: SmolVLAConfig) -> Path:
    """Save actual TE quantized weights and their scale metadata for inference."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = _build_fp8_manifest(module, config)
    payload = {
        "format": FP8_CHECKPOINT_FORMAT,
        "manifest": manifest,
        "state_dict": module.state_dict(),
    }

    # torch.save invokes TE QuantizedTensor.__reduce_ex__, which serializes the raw uint8 payload,
    # scale tensors, format and quantizer metadata without dequantizing the weight.
    with tempfile.NamedTemporaryFile(dir=directory, prefix=".model_fp8-", suffix=".tmp", delete=False) as f:
        temporary_path = Path(f.name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, directory / FP8_WEIGHTS_NAME)
    finally:
        temporary_path.unlink(missing_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", dir=directory, prefix=".fp8_manifest-", suffix=".tmp", delete=False
    ) as f:
        json.dump(manifest, f, indent=2)
        manifest_temporary_path = Path(f.name)
    try:
        os.replace(manifest_temporary_path, directory / FP8_MANIFEST_NAME)
    finally:
        manifest_temporary_path.unlink(missing_ok=True)
    return directory / FP8_WEIGHTS_NAME


def load_fp8_checkpoint(
    module: nn.Module,
    checkpoint_file: str | Path,
    config: SmolVLAConfig,
    *,
    map_location: str | torch.device,
) -> nn.Module:
    """Load a trusted, local FP8-only SmolVLA checkpoint into quantized TE parameters."""
    checkpoint_file = Path(checkpoint_file)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(
            f"FP8-only config requires {FP8_WEIGHTS_NAME}, but it was not found at {checkpoint_file}."
        )

    # TE QuantizedTensor uses a custom pickle reducer. weights_only=False is therefore required;
    # callers must only load artifacts they trust, just like any conventional .pt checkpoint.
    payload = torch.load(checkpoint_file, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != FP8_CHECKPOINT_FORMAT:
        raise RuntimeError(f"{checkpoint_file} is not a supported SmolVLA FP8-only checkpoint.")
    manifest = payload.get("manifest")
    state_dict = payload.get("state_dict")
    if not isinstance(manifest, dict) or not isinstance(state_dict, dict):
        raise RuntimeError(f"{checkpoint_file} is missing its FP8 manifest or state_dict.")

    import transformer_engine

    expected = (config.fp8_recipe, config.fp8_format, config.fp8_fuse_layernorm)
    actual = (
        manifest.get("recipe"),
        manifest.get("fp8_format"),
        bool(manifest.get("fuse_layernorm", False)),
    )
    if actual != expected:
        raise RuntimeError(
            f"FP8 recipe/fusion mismatch: checkpoint uses {actual}, but config requests {expected}."
        )
    saved_te_version = str(manifest.get("transformer_engine_version", ""))
    if _version_major_minor(saved_te_version) != _version_major_minor(transformer_engine.__version__):
        raise RuntimeError(
            "Transformer Engine major/minor mismatch for FP8-only checkpoint: "
            f"saved with {saved_te_version}, running {transformer_engine.__version__}."
        )

    # FP8-only inference modules are constructed on meta so TE does not quantize temporary random
    # weights. assign=True installs the serialized QuantizedTensor objects and their CUDA payloads.
    incompatible = module.load_state_dict(state_dict, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"FP8-only checkpoint has incompatible keys: {incompatible}.")
    _build_fp8_manifest(module, config)
    return module
