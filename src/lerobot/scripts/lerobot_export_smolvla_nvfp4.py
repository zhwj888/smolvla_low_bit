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
"""Export a trained SmolVLA checkpoint with NVFP4-primary Transformer Engine weights."""

import json
import logging
import os
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from torch import nn

from lerobot.configs import parser
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import PRETRAINED_MODEL_DIR
from lerobot.utils.utils import init_logging

NVFP4_WEIGHTS_NAME = "model_nvfp4.pt"
NVFP4_MANIFEST_NAME = "nvfp4_manifest.json"
NVFP4_CHECKPOINT_FORMAT = "lerobot-smolvla-nvfp4-primary-v1"


@dataclass
class ExportSmolVLANVFP4Config:
    """Options for converting a trained SmolVLA checkpoint to NVFP4-primary weights."""

    # A checkpoint step directory containing pretrained_model/, or pretrained_model itself.
    checkpoint_dir: Path
    # New output directory containing model_nvfp4.pt and the source preprocessing assets.
    output_dir: Path
    device: str = "cuda"
    overwrite: bool = False


def _locate_pretrained_dir(checkpoint_dir: Path) -> Path:
    nested = checkpoint_dir / PRETRAINED_MODEL_DIR
    return nested if nested.is_dir() else checkpoint_dir


def _prepare_output_dir(source_dir: Path, output_dir: Path, overwrite: bool) -> None:
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("output_dir must differ from checkpoint_dir; the training checkpoint is preserved.")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{output_dir} is not empty. Choose a new directory or pass --overwrite=true."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    excluded = {
        SAFETENSORS_SINGLE_FILE,
        NVFP4_WEIGHTS_NAME,
        NVFP4_MANIFEST_NAME,
    }
    for source in source_dir.iterdir():
        if source.name in excluded:
            continue
        destination = output_dir / source.name
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)


def _atomic_torch_save(payload: dict, output_file: Path) -> None:
    with tempfile.NamedTemporaryFile(
        dir=output_file.parent,
        prefix=".model_nvfp4-",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, output_file)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_json_save(payload: dict, output_file: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=output_file.parent,
        prefix=".nvfp4_manifest-",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2)
        temporary_path = Path(handle.name)
    try:
        os.replace(temporary_path, output_file)
    finally:
        temporary_path.unlink(missing_ok=True)


def _synchronize_nvfp4_primary_weight_quantizers(module: nn.Module, te) -> None:
    """Attach each module NVFP4Quantizer to its serialized NVFP4Tensor parameter."""
    for weight_name in getattr(module, "weight_names", ()):
        weight = getattr(module, weight_name, None)
        if not isinstance(weight, te.NVFP4Tensor):
            continue

        init_meta = module.param_init_meta.get(weight_name)
        quantizer_index = None if init_meta is None else init_meta.fp8_meta_index
        if quantizer_index is None:
            raise RuntimeError(
                f"Transformer Engine module {module.name!r} has no quantizer index for "
                f"NVFP4 parameter {weight_name!r}."
            )
        quantizer = module.quantizers["scaling_fwd"][quantizer_index]
        if not isinstance(quantizer, te.NVFP4Quantizer):
            raise RuntimeError(
                f"Expected NVFP4Quantizer for {module.name}.{weight_name}, got "
                f"{type(quantizer).__name__}."
            )

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


def _materialize_nvfp4_primary_weights(module: nn.Module, nvfp4) -> list[str]:
    """Convert loaded BF16 TE weights to NVFP4Tensor primary parameters exactly once."""
    te = nvfp4.te
    converted: list[str] = []

    with torch.no_grad():
        for module_name, child in module.named_modules():
            if not hasattr(child, "primary_weights_in_fp8"):
                continue

            child.fp8_meta["recipe"] = nvfp4.recipe
            child.init_fp8_meta_tensors(nvfp4.recipe)

            for weight_name in child.weight_names:
                weight = getattr(child, weight_name)
                qualified_name = f"{module_name}.{weight_name}" if module_name else weight_name
                if isinstance(weight, te.NVFP4Tensor):
                    converted.append(qualified_name)
                    continue
                if isinstance(weight, te.QuantizedTensor):
                    raise RuntimeError(
                        f"Expected a high-precision or NVFP4 weight for {qualified_name}, got "
                        f"{type(weight).__name__}."
                    )

                init_meta = child.param_init_meta.get(weight_name)
                quantizer_index = None if init_meta is None else init_meta.fp8_meta_index
                if quantizer_index is None:
                    raise RuntimeError(
                        f"Transformer Engine module {module_name!r} has no NVFP4 quantizer index "
                        f"for parameter {weight_name!r}."
                    )
                quantizer = child.quantizers["scaling_fwd"][quantizer_index]
                if not isinstance(quantizer, te.NVFP4Quantizer):
                    raise RuntimeError(
                        f"Expected NVFP4Quantizer for {qualified_name}, got "
                        f"{type(quantizer).__name__}."
                    )
                compatible_recipe = quantizer._get_compatible_recipe()
                if compatible_recipe is None or not isinstance(nvfp4.recipe, compatible_recipe):
                    raise RuntimeError(
                        f"NVFP4 quantizer for {qualified_name} is not compatible with "
                        f"{type(nvfp4.recipe).__name__}."
                    )

                quantizer.set_usage(rowwise=True, columnwise=False)
                quantizer.internal = False
                try:
                    quantized_weight = quantizer(weight.detach())
                    if weight.device.type == "cuda":
                        torch.cuda.synchronize(weight.device)
                except Exception as error:
                    raise RuntimeError(
                        f"Failed to materialize NVFP4-primary weight {qualified_name} with shape "
                        f"{tuple(weight.shape)}, dtype={weight.dtype}, device={weight.device}."
                    ) from error
                if not isinstance(quantized_weight, te.NVFP4Tensor):
                    raise RuntimeError(
                        f"NVFP4Quantizer returned {type(quantized_weight).__name__} for "
                        f"{qualified_name}; expected NVFP4Tensor."
                    )

                child.module_setattr(
                    weight_name,
                    nn.Parameter(quantized_weight, requires_grad=False),
                )
                converted.append(qualified_name)

            # TE retains the historical field name for all quantized primary-weight formats,
            # including NVFP4. It does not imply that these parameters use Float8Tensor.
            child.primary_weights_in_fp8 = True
            child._fp8_workspaces.clear()
            _synchronize_nvfp4_primary_weight_quantizers(child, te)

    if not converted:
        raise RuntimeError("No Transformer Engine weights were found to materialize as NVFP4.")
    return converted


@parser.wrap()
def export_smolvla_nvfp4(cfg: ExportSmolVLANVFP4Config) -> Path:
    """Materialize final checkpoint weights as NVFP4 tensors and save a trusted local PT payload."""
    if not torch.cuda.is_available() or not cfg.device.startswith("cuda"):
        raise RuntimeError("SmolVLA NVFP4-primary export requires a CUDA device supported by TE.")

    import transformer_engine
    import transformer_engine.pytorch as te

    source_dir = _locate_pretrained_dir(cfg.checkpoint_dir)
    model_file = source_dir / SAFETENSORS_SINGLE_FILE
    if not model_file.is_file():
        raise FileNotFoundError(f"Standard source checkpoint not found: {model_file}")
    _prepare_output_dir(source_dir, cfg.output_dir, cfg.overwrite)

    config = SmolVLAConfig.from_pretrained(source_dir)
    if not config.use_nvfp4:
        logging.warning(
            "The source was not marked as NVFP4-trained; its final master weights will still be "
            "quantized using the configured NVFP4 recipe."
        )
    config.device = cfg.device
    config.use_fp8 = False
    config.use_nvfp4 = True
    config.compile_model = False
    config.load_vlm_weights = False
    config.pretrained_path = None

    logging.info("Building SmolVLA NVFP4 export model on %s.", config.device)
    policy = SmolVLAPolicy(config)
    PreTrainedPolicy._load_as_safetensor(
        policy,
        str(model_file),
        config.device,
        strict=True,
    )
    policy.to(torch.device(config.device))
    policy.eval()
    policy.requires_grad_(False)

    converted = _materialize_nvfp4_primary_weights(policy, policy.model._nvfp4)
    quantized_parameters = [
        name
        for name, parameter in policy.named_parameters()
        if isinstance(parameter, te.NVFP4Tensor)
    ]
    fallback_linears = [
        name for name, child in policy.named_modules() if type(child) is nn.Linear
    ]
    if len(quantized_parameters) != len(converted):
        raise RuntimeError(
            "NVFP4 export parameter count mismatch: "
            f"materialized {len(converted)}, found {len(quantized_parameters)} quantized parameters."
        )

    manifest = {
        "format": NVFP4_CHECKPOINT_FORMAT,
        "policy_type": "smolvla",
        "torch_version": torch.__version__,
        "transformer_engine_version": transformer_engine.__version__,
        "recipe": "NVFP4BlockScaling",
        "disable_stochastic_rounding": config.nvfp4_disable_stochastic_rounding,
        "disable_rht": config.nvfp4_disable_rht,
        "disable_2d_quantization": config.nvfp4_disable_2d_quantization,
        "quantized_parameter_count": len(quantized_parameters),
        "quantized_parameter_names": quantized_parameters,
        "bf16_fallback_linear_count": len(fallback_linears),
        "bf16_fallback_linear_names": fallback_linears,
        "source_checkpoint": str(source_dir),
    }
    payload = {
        "format": NVFP4_CHECKPOINT_FORMAT,
        "manifest": manifest,
        "config": json.loads((source_dir / "config.json").read_text()),
        "state_dict": policy.state_dict(),
    }

    output_file = cfg.output_dir / NVFP4_WEIGHTS_NAME
    _atomic_torch_save(payload, output_file)
    _atomic_json_save(manifest, cfg.output_dir / NVFP4_MANIFEST_NAME)
    logging.info(
        "Exported %d NVFP4 parameters and retained %d BF16 Linear modules in %s.",
        len(quantized_parameters),
        len(fallback_linears),
        output_file,
    )
    return output_file


def main() -> None:
    init_logging()
    export_smolvla_nvfp4()


if __name__ == "__main__":
    main()
