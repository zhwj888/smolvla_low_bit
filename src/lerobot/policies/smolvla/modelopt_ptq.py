#!/usr/bin/env python

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

"""ModelOpt PTQ helpers for a high-precision SmolVLA model.

This module is deliberately independent from SmolVLA's Transformer Engine integrations. It
creates ModelOpt fake-quantized modules for PTQ/AutoQuant and preserves high-precision master
weights so the resulting checkpoint can be used as the student of a later QAD run.
"""

from __future__ import annotations

import fnmatch
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from torch import nn

MODELOPT_PTQ_CHECKPOINT_NAME = "modelopt_ptq.pt"
MODELOPT_STATE_NAME = "modelopt_state.pth"
MODELOPT_PTQ_MANIFEST_NAME = "modelopt_ptq_manifest.json"
MODELOPT_AUTOQUANT_STATE_NAME = "modelopt_autoquant_state.pt"
MODELOPT_PTQ_FORMAT = "lerobot-smolvla-modelopt-fake-quant-v1"

_VISION_NAME_TOKENS = (
    "embed_vision",
    "vision_model",
    "vision_tower",
    "visual",
)

_OUTPUT_HEAD_NAME_SUFFIXES = (
    "action_out_proj",
    "lm_head",
)


@dataclass(frozen=True)
class LinearPTQDecision:
    """PTQ eligibility decision made on the original BF16 module graph."""

    name: str
    in_features: int
    out_features: int
    trainable: bool
    eligible: bool
    reason: str


def analyze_linear_eligibility(
    model: nn.Module,
    *,
    feature_alignment: int,
    trainable_only: bool,
    keep_vision_high_precision: bool,
    keep_output_heads_high_precision: bool,
    excluded_module_patterns: list[str] | None = None,
) -> list[LinearPTQDecision]:
    """Classify BF16 Linear modules before ModelOpt replaces them.

    ``feature_alignment`` is a deployment constraint, not a fake-quantization requirement. On
    GB10/SM12x it should be 32 to keep the search result executable by the existing TE 2.16
    NVFP4 path; other Blackwell variants may use 16.
    """

    if feature_alignment <= 0:
        raise ValueError(f"feature_alignment must be positive, got {feature_alignment}.")

    excluded_module_patterns = excluded_module_patterns or []
    decisions: list[LinearPTQDecision] = []
    seen_modules: set[int] = set()

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or id(module) in seen_modules:
            continue
        seen_modules.add(id(module))

        trainable = any(parameter.requires_grad for parameter in module.parameters(recurse=False))
        reason = "eligible"
        eligible = True

        if any(fnmatch.fnmatch(name, pattern) for pattern in excluded_module_patterns):
            eligible = False
            reason = "user_excluded"
        elif keep_output_heads_high_precision and name.endswith(_OUTPUT_HEAD_NAME_SUFFIXES):
            eligible = False
            reason = "output_head_high_precision"
        elif keep_vision_high_precision and any(token in name.lower() for token in _VISION_NAME_TOKENS):
            eligible = False
            reason = "vision_high_precision"
        elif trainable_only and not trainable:
            eligible = False
            reason = "frozen"
        elif module.in_features % feature_alignment or module.out_features % feature_alignment:
            eligible = False
            reason = f"feature_shape_not_{feature_alignment}_aligned"

        decisions.append(
            LinearPTQDecision(
                name=name,
                in_features=module.in_features,
                out_features=module.out_features,
                trainable=trainable,
                eligible=eligible,
                reason=reason,
            )
        )

    return decisions


def disabled_layer_names(decisions: list[LinearPTQDecision]) -> list[str]:
    """Return exact module names that must stay outside the quantization search space."""

    return [decision.name for decision in decisions if not decision.eligible]


def _configure_nvfp4_rht(
    config: dict[str, Any],
    *,
    enable_rht: bool,
    rht_block_size: int | None,
    rht_rotate_fp32: bool,
) -> None:
    """Configure matching activation/weight rotations so they cancel inside Linear."""

    if not enable_rht:
        return

    rotate = {
        "enable": True,
        "rotate_fp32": rht_rotate_fp32,
        "block_size": rht_block_size,
    }
    found: set[str] = set()
    entries: list[dict[str, Any]] = []
    for entry in config["quant_cfg"]:
        entries.append(entry)
        quantizer_name = entry.get("quantizer_name")
        if quantizer_name not in ("*input_quantizer", "*weight_quantizer"):
            continue
        quantizer_config = entry.get("cfg")
        if not isinstance(quantizer_config, dict):
            raise TypeError(
                f"Expected a dictionary config for {quantizer_name}, got {type(quantizer_config).__name__}."
            )
        # Rotation runs even when a TensorQuantizer is disabled. Keep the original
        # quantization rule, but only add paired rotations to Linear operands;
        # LayerNorm, Embedding and Conv2d must not inherit this transform.
        quantizer_config["rotate"] = False
        linear_entry = deepcopy(entry)
        linear_entry["parent_class"] = "nn.Linear"
        linear_entry["cfg"]["rotate"] = deepcopy(rotate)
        entries.append(linear_entry)
        found.add(quantizer_name)

    expected = {"*input_quantizer", "*weight_quantizer"}
    if found != expected:
        raise KeyError(f"NVFP4 recipe is missing RHT targets: {sorted(expected - found)}.")
    config["quant_cfg"] = entries


def _configure_nvfp4_calibration(
    config: dict[str, Any], *, calibration_algorithm: str, fp8_scale_sweep: bool
) -> None:
    """Use static weight blocks for the NVFP4 MSE FP8-scale sweep preset."""
    if not fp8_scale_sweep:
        config["algorithm"] = calibration_algorithm
        return
    if calibration_algorithm != "mse":
        raise ValueError("fp8_scale_sweep requires calibration_algorithm='mse'.")

    found = False
    for entry in config["quant_cfg"]:
        if entry.get("quantizer_name") == "*weight_quantizer":
            weight_cfg = entry.get("cfg", {})
            block_sizes = weight_cfg.get("block_sizes")
            if not isinstance(block_sizes, dict):
                raise ValueError("NVFP4 FP8-scale sweep requires blockwise weight quantization.")
            block_sizes["type"] = "static"
            found = True
    if not found:
        raise ValueError("NVFP4 recipe is missing its weight quantizer for FP8-scale sweep.")
    config["algorithm"] = {"method": "mse", "fp8_scale_sweep": True}


def build_fixed_nvfp4_config(
    modelopt_quantization: Any,
    decisions: list[LinearPTQDecision],
    *,
    calibration_algorithm: str,
    fp8_scale_sweep: bool = False,
    enable_rht: bool = False,
    rht_block_size: int | None = 16,
    rht_rotate_fp32: bool = False,
) -> dict[str, Any]:
    """Build a fixed NVFP4 recipe with deployment-incompatible modules disabled last."""

    config = deepcopy(modelopt_quantization.NVFP4_DEFAULT_CFG)
    _configure_nvfp4_calibration(
        config, calibration_algorithm=calibration_algorithm, fp8_scale_sweep=fp8_scale_sweep
    )
    _configure_nvfp4_rht(
        config,
        enable_rht=enable_rht,
        rht_block_size=rht_block_size,
        rht_rotate_fp32=rht_rotate_fp32,
    )
    for name in disabled_layer_names(decisions):
        # Last-match-wins in ModelOpt. Match the module's input/weight/output quantizers without
        # accidentally disabling a sibling whose name merely shares a prefix.
        entry: dict[str, Any] = {"quantizer_name": f"{name}.*", "enable": False}
        if enable_rht:
            # A toggle-only disable preserves earlier attributes in ModelOpt. Explicitly replace
            # the config so excluded BF16 modules do not still execute an unnecessary rotation.
            entry["cfg"] = {"rotate": False}
        config["quant_cfg"].append(entry)
    return config


def build_autoquant_formats(
    modelopt_quantization: Any,
    *,
    calibration_algorithm: str,
    fp8_scale_sweep: bool = False,
    enable_rht: bool = False,
    rht_block_size: int | None = 16,
    rht_rotate_fp32: bool = False,
) -> list[dict[str, Any]]:
    """Return independent NVFP4 and FP8 candidate recipes for AutoQuant."""

    nvfp4 = deepcopy(modelopt_quantization.NVFP4_DEFAULT_CFG)
    fp8 = deepcopy(modelopt_quantization.FP8_DEFAULT_CFG)
    _configure_nvfp4_calibration(
        nvfp4, calibration_algorithm=calibration_algorithm, fp8_scale_sweep=fp8_scale_sweep
    )
    # The sweep preset refines NVFP4 weights only; retain the original FP8 recipe.
    if not fp8_scale_sweep:
        fp8["algorithm"] = calibration_algorithm
    _configure_nvfp4_rht(
        nvfp4,
        enable_rht=enable_rht,
        rht_block_size=rht_block_size,
        rht_rotate_fp32=rht_rotate_fp32,
    )
    return [nvfp4, fp8]


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, float, int, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _quantizer_enabled(quantizer: nn.Module | None) -> bool:
    return quantizer is not None and not bool(getattr(quantizer, "_disabled", True))


def _quantizer_num_bits(quantizer: nn.Module | None) -> Any:
    if quantizer is None:
        return None
    return getattr(quantizer, "num_bits", getattr(quantizer, "_num_bits", None))


def _quantizer_rotation(quantizer: nn.Module | None) -> dict[str, Any]:
    """Return the effective ModelOpt Hadamard rotation settings for reporting."""

    if quantizer is None:
        return {"enabled": False, "block_size": None, "rotate_fp32": False}
    return {
        "enabled": bool(getattr(quantizer, "rotate_is_enabled", False)),
        "block_size": getattr(quantizer, "rotate_block_size", None),
        "rotate_fp32": bool(getattr(quantizer, "rotate_is_fp32", False)),
    }


def _precision_from_quantizers(module: nn.Module) -> str:
    input_quantizer = getattr(module, "input_quantizer", None)
    weight_quantizer = getattr(module, "weight_quantizer", None)
    if not (_quantizer_enabled(input_quantizer) and _quantizer_enabled(weight_quantizer)):
        return "bf16"

    num_bits = _quantizer_num_bits(weight_quantizer)
    normalized_bits = tuple(num_bits) if isinstance(num_bits, (list, tuple)) else num_bits
    if normalized_bits == (2, 1):
        return "nvfp4"
    if normalized_bits == (4, 3):
        return "fp8"
    return "quantized_other"


def build_quantized_linear_report(model: nn.Module) -> dict[str, Any]:
    """Describe the selected precision of every Linear after PTQ/AutoQuant."""

    modules: list[dict[str, Any]] = []
    seen_modules: set[int] = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or id(module) in seen_modules:
            continue
        seen_modules.add(id(module))

        input_quantizer = getattr(module, "input_quantizer", None)
        weight_quantizer = getattr(module, "weight_quantizer", None)
        precision = _precision_from_quantizers(module)
        modules.append(
            {
                "name": name,
                "module_class": type(module).__name__,
                "in_features": module.in_features,
                "out_features": module.out_features,
                "weight_dtype": str(module.weight.dtype),
                "trainable": any(
                    parameter.requires_grad for parameter in module.parameters(recurse=False)
                ),
                "precision": precision,
                "input_quantizer": {
                    "enabled": _quantizer_enabled(input_quantizer),
                    "num_bits": _json_value(_quantizer_num_bits(input_quantizer)),
                    "block_sizes": _json_value(getattr(input_quantizer, "block_sizes", None)),
                    "rotation": _quantizer_rotation(input_quantizer),
                },
                "weight_quantizer": {
                    "enabled": _quantizer_enabled(weight_quantizer),
                    "num_bits": _json_value(_quantizer_num_bits(weight_quantizer)),
                    "block_sizes": _json_value(getattr(weight_quantizer, "block_sizes", None)),
                    "rotation": _quantizer_rotation(weight_quantizer),
                },
            }
        )

    precision_bits = {"bf16": 16, "fp8": 8, "nvfp4": 4}
    total_weight_elements = sum(
        module["in_features"] * module["out_features"] for module in modules
    )
    weighted_effective_bits = None
    if total_weight_elements and all(module["precision"] in precision_bits for module in modules):
        weighted_effective_bits = sum(
            module["in_features"]
            * module["out_features"]
            * precision_bits[module["precision"]]
            for module in modules
        ) / total_weight_elements

    counts = Counter(module["precision"] for module in modules)
    return {
        "linear_count": len(modules),
        "total_weight_elements": total_weight_elements,
        "weighted_effective_bits": weighted_effective_bits,
        "precision_counts": dict(sorted(counts.items())),
        "modules": modules,
    }


def serialize_decisions(decisions: list[LinearPTQDecision]) -> list[dict[str, Any]]:
    return [asdict(decision) for decision in decisions]
