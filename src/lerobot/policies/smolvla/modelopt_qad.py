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

"""Independent ModelOpt QAD support for SmolVLA's continuous flow predictions.

Restore the PTQ quantizers without recalibration or layer selection. The student keeps
FP32 optimizer/master parameters and uses BF16 autocast with fake quantization in forward.
The frozen BF16 teacher predicts the velocity at the same noisy action and timestep.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import require_package

from .configuration_smolvla import SmolVLAConfig
from .modeling_smolvla import SmolVLAPolicy
from .modelopt_ptq import MODELOPT_PTQ_CHECKPOINT_NAME, MODELOPT_PTQ_FORMAT, MODELOPT_PTQ_MANIFEST_NAME
from .qad_safetensors import (
    QAD_SAFE_FORMAT,
    QAD_SAFE_MANIFEST_NAME,
    QAD_SAFE_STATE_NAME,
    QAD_SAFE_WEIGHTS_NAME,
    read_modelopt_state,
)

QAD_FORMAT = "lerobot-smolvla-modelopt-qad-v1"
QAD_WEIGHTS_NAME = "modelopt_qad.pt"
QAD_MANIFEST_NAME = "modelopt_qad_manifest.json"
QAD_TRAINING_STATE_NAME = "qad_training_state.pt"


def modelopt_modules():
    """Load ModelOpt only when running QAD; no changes to shared optional dependencies."""
    try:
        require_package("nvidia-modelopt", extra="smolvla", import_name="modelopt")
    except ImportError as exc:
        raise ImportError("QAD requires nvidia-modelopt[torch] in the SmolVLA Python environment.") from exc
    return (
        importlib.import_module("modelopt.torch.opt"),
        importlib.import_module("modelopt.torch.quantization.nn"),
    )


def read_student_manifest(path: str | Path) -> tuple[dict[str, Any], str]:
    path = Path(path)
    for name, expected, weights in (
        (QAD_SAFE_MANIFEST_NAME, QAD_SAFE_FORMAT, QAD_SAFE_WEIGHTS_NAME),
        (QAD_MANIFEST_NAME, QAD_FORMAT, QAD_WEIGHTS_NAME),
        (MODELOPT_PTQ_MANIFEST_NAME, MODELOPT_PTQ_FORMAT, MODELOPT_PTQ_CHECKPOINT_NAME),
    ):
        if (path / name).is_file():
            manifest = json.loads((path / name).read_text())
            if manifest.get("format") != expected or not manifest.get("fake_quant"):
                raise ValueError(f"Unsupported fake-quant checkpoint manifest: {path / name}")
            if manifest.get("compressed") or manifest.get("transformer_engine_used"):
                raise ValueError(
                    "QAD requires ModelOpt fake quantization with high-precision master weights."
                )
            if not (path / weights).is_file():
                raise FileNotFoundError(path / weights)
            return manifest, weights
    raise FileNotFoundError(f"No ModelOpt PTQ/QAD manifest found in {path}.")


def _plain_config(path: str | Path, device: str) -> SmolVLAConfig:
    config = SmolVLAConfig.from_pretrained(path)
    if any(getattr(config, key) for key in ("use_fp8", "use_nvfp4", "fp8_primary_weights")):
        raise ValueError("QAD only accepts the plain SmolVLA architecture, without a TE runtime.")
    if config.use_peft or config.adapt_to_pi_aloha:
        raise ValueError("This QAD entry point requires a full checkpoint and adapt_to_pi_aloha=False.")
    config.device = device
    config.load_vlm_weights = False
    config.compile_model = False
    config.use_amp = False
    config.pretrained_path = None
    config.push_to_hub = False
    return config


def quantizer_signature(model: nn.Module) -> dict[str, Any]:
    """Include inactive quantizers so accidental changes to the PTQ assignment are detectable."""
    _, quant_nn = modelopt_modules()
    return {
        name: {
            "enabled": not quantizer._disabled,
            "num_bits": quantizer.num_bits,
            "block_sizes": quantizer.block_sizes,
            "fake_quant": quantizer.fake_quant,
        }
        for name, quantizer in model.named_modules()
        if isinstance(quantizer, quant_nn.TensorQuantizer)
    }


def load_qad_student(path: str | Path, device: str = "cuda") -> SmolVLAPolicy:
    """Restore PTQ or QAD state for training/inference under BF16 autocast.

    Local ModelOpt .pt checkpoints must be trusted, like the existing PTQ artifacts.
    Construct FP32 masters BEFORE restore; casting the whole restored model would also
    cast its calibrated scale buffers. No quantizer assignment or scale is recalibrated.
    """
    mto, quant_nn = modelopt_modules()
    manifest, weights = read_student_manifest(path)
    policy = SmolVLAPolicy(_plain_config(path, device)).float()
    if manifest["format"] == QAD_SAFE_FORMAT:
        state = read_modelopt_state(Path(path) / QAD_SAFE_STATE_NAME)
        policy = mto.restore_from_modelopt_state(policy, state)
        policy.load_state_dict(load_file(Path(path) / weights, device="cpu"), strict=True)
    else:
        policy = mto.restore(policy, Path(path) / weights)
    policy.to(device=device)
    active = []
    for name, quantizer in policy.named_modules():
        if not isinstance(quantizer, quant_nn.TensorQuantizer):
            continue
        quantizer.disable_calib()
        if quantizer._disabled:
            continue
        if not quantizer.fake_quant:
            raise ValueError(f"QAD requires fake quantization: {name}")
        quantizer.enable_quant()
        if not hasattr(quantizer, "_amax") or not torch.isfinite(quantizer._amax).all():
            raise ValueError(f"Missing/nonfinite PTQ calibration for {name}")
        if not (quantizer._amax > 0).all():
            raise ValueError(f"Nonpositive PTQ calibration for {name}")
        active.append(name)
    if not active:
        raise ValueError("Student has no enabled ModelOpt quantizers.")
    # QAD checkpoints record trainability explicitly; PTQ follows its original config.
    if "trainable_parameter_names" in manifest:
        trainable = set(manifest["trainable_parameter_names"])
        for name, parameter in policy.named_parameters():
            parameter.requires_grad_(name in trainable)
    return policy.eval()


def load_bf16_teacher(path: str | Path, student_config: SmolVLAConfig, device: str) -> SmolVLAPolicy:
    config = _plain_config(path, device)
    # Compare semantics as well as architecture, allowing training/performance flags to differ.
    for key in (
        "input_features",
        "output_features",
        "chunk_size",
        "max_state_dim",
        "max_action_dim",
        "normalization_mapping",
        "resize_imgs_with_padding",
        "empty_cameras",
        "vlm_model_name",
        "attention_mode",
        "num_vlm_layers",
        "num_expert_layers",
        "self_attn_every_n_layers",
        "expert_width_multiplier",
        "add_image_special_tokens",
        "prefix_length",
        "min_period",
        "max_period",
    ):
        if getattr(config, key) != getattr(student_config, key):
            raise ValueError(f"Teacher/student config mismatch: {key}")
    teacher = SmolVLAPolicy.from_pretrained(path, config=config, strict=True)
    return teacher.to(device=device, dtype=torch.bfloat16).requires_grad_(False).eval()


def masked_velocity_mse(student: Tensor, teacher: Tensor, action_dim: int, is_pad: Tensor | None) -> Tensor:
    """Exclude padded action coordinates and episode-tail frames from distillation."""
    if student.shape != teacher.shape or student.ndim != 3:
        raise ValueError("Teacher/student velocities must have the same (batch, chunk, action) shape.")
    if not 0 < action_dim <= student.shape[-1]:
        raise ValueError("Invalid physical action dimension.")
    error = (student[..., :action_dim].float() - teacher[..., :action_dim].detach().float()).square()
    if is_pad is None:
        return error.mean()
    if is_pad.shape != student.shape[:2]:
        raise ValueError("action_is_pad must have shape (batch, chunk).")
    valid = ~is_pad.to(device=error.device, dtype=torch.bool)
    # masked_fill avoids NaN/Inf in padded entries leaking through a multiplication by zero.
    return error.masked_fill(~valid.unsqueeze(-1), 0).sum() / (valid.sum() * action_dim).clamp_min(1)


def _velocity_forward(policy: nn.Module, batch: dict[str, Any], noise: Tensor, time: Tensor):
    predictions = []

    def capture(_module, _args, output):
        predictions.append(output)

    handle = policy.model.action_out_proj.register_forward_hook(capture)
    try:
        # SmolVLA may assign state/action entries in forward; keep teacher/student batches separate.
        local_batch = dict(batch)
        for key in (OBS_STATE, ACTION):
            if key in local_batch:
                local_batch[key] = local_batch[key].clone()
        output = policy(local_batch, noise=noise, time=time)
    finally:
        handle.remove()
    if len(predictions) != 1:
        raise RuntimeError("Expected exactly one action velocity prediction from SmolVLA forward.")
    task_loss = output[0] if isinstance(output, tuple) else output
    return predictions[0], task_loss


def qad_forward(
    student: nn.Module,
    teacher: nn.Module,
    batch: dict[str, Any],
    noise: Tensor,
    time: Tensor,
    *,
    distillation_weight: float = 1.0,
    task_weight: float = 0.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Distill velocities, not scalar task losses or categorical language logits.

    Caller owns autocast. Both models must receive the same normalized batch, noise and
    timestep. Teacher is always evaluated without gradients; only student is optimized.
    """
    teacher.eval()
    with torch.no_grad():
        target, _ = _velocity_forward(teacher, batch, noise, time)
    prediction, task_loss = _velocity_forward(student, batch, noise, time)
    kd_loss = masked_velocity_mse(
        prediction, target, student.config.action_feature.shape[0], batch.get("action_is_pad")
    )
    loss = distillation_weight * kd_loss
    if task_weight:
        loss = loss + task_weight * task_loss
    return loss, {
        "loss": loss.detach(),
        "distillation_loss": kd_loss.detach(),
        "task_loss": task_loss.detach(),
    }
