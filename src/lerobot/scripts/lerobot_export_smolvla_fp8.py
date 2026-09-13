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
"""Export a trained SmolVLA checkpoint with actual Transformer Engine FP8-only weights."""

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

# TE 2.16's Float8BlockScaling initialization can race internally when hundreds of primary FP8
# weights are materialized in one process. Restrict blocking launches to this one-time exporter;
# inference loads serialized FP8 payloads through meta tensors and does not need this workaround.
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import torch
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE

from lerobot.configs import parser
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.fp8 import (
    FP8_MANIFEST_NAME,
    FP8_WEIGHTS_NAME,
    materialize_fp8_primary_weights,
    save_fp8_checkpoint,
)
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import PRETRAINED_MODEL_DIR
from lerobot.utils.utils import init_logging


@dataclass
class ExportSmolVLAFP8Config:
    """Options for converting a standard SmolVLA checkpoint to FP8-only inference weights."""

    # A checkpoint step directory containing pretrained_model/, or pretrained_model itself.
    checkpoint_dir: Path
    # New self-contained directory that lerobot-eval will use as --policy.path.
    output_dir: Path
    device: str = "cuda"
    overwrite: bool = False
    # None inherits the source checkpoint setting. Set true/false to choose the exported runtime
    # structure independently of how the high-precision training checkpoint was produced.
    fp8_fuse_layernorm: bool | None = None


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
        "config.json",
        SAFETENSORS_SINGLE_FILE,
        FP8_WEIGHTS_NAME,
        FP8_MANIFEST_NAME,
    }
    for source in source_dir.iterdir():
        if source.name in excluded:
            continue
        destination = output_dir / source.name
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)


@parser.wrap()
def export_smolvla_fp8(cfg: ExportSmolVLAFP8Config) -> Path:
    """Quantize an existing SmolVLA checkpoint once and save its true FP8 weight payloads."""
    if not torch.cuda.is_available() or not cfg.device.startswith("cuda"):
        raise RuntimeError("SmolVLA FP8-only export requires a CUDA device supported by Transformer Engine.")

    source_dir = _locate_pretrained_dir(cfg.checkpoint_dir)
    model_file = source_dir / SAFETENSORS_SINGLE_FILE
    if not model_file.is_file():
        raise FileNotFoundError(f"Standard source checkpoint not found: {model_file}")
    _prepare_output_dir(source_dir, cfg.output_dir, cfg.overwrite)

    config = SmolVLAConfig.from_pretrained(source_dir)
    if not config.use_fp8:
        logging.warning(
            "The source was not marked as FP8-trained; its high-precision weights will still be "
            "quantized for FP8-only inference."
        )
    config.device = cfg.device
    config.use_fp8 = True
    # Build ordinary high-precision TE parameters first. Constructing hundreds of FP8-primary
    # modules would quantize temporary random weights before the checkpoint is loaded; TE 2.16's
    # Hopper square-blockwise initialization kernel can fail in that redundant pass. The final
    # checkpoint weights are converted exactly once after safetensors loading below.
    config.fp8_primary_weights = False
    config.compile_model = False
    if cfg.fp8_fuse_layernorm is not None:
        config.fp8_fuse_layernorm = cfg.fp8_fuse_layernorm
    # The exported state dict is complete, so loading the original SmolVLM weights first would only
    # waste startup time and memory before the FP8 checkpoint overwrites them.
    config.load_vlm_weights = False
    config.pretrained_path = None

    logging.info(
        "Building deferred FP8-only SmolVLA with recipe=%s format=%s LayerNormLinear=%s on %s.",
        config.fp8_recipe,
        config.fp8_format,
        config.fp8_fuse_layernorm,
        config.device,
    )
    policy = SmolVLAPolicy(config)
    # Explicitly call the base safetensors loader: the SmolVLA override is reserved for loading the
    # already-exported model_fp8.pt format.
    PreTrainedPolicy._load_as_safetensor(policy, str(model_file), config.device, strict=True)
    # load_state_dict copies tensors into the module's existing parameters; its map location does
    # not move those CPU parameters. TE's CUDA quantizers require the source weights on CUDA.
    policy.to(torch.device(config.device))
    policy.eval()
    policy.requires_grad_(False)

    converted = materialize_fp8_primary_weights(policy, policy.model._fp8)
    config.fp8_primary_weights = True
    logging.info("Materialized %d final checkpoint weights as FP8-primary parameters.", len(converted))

    fp8_file = save_fp8_checkpoint(policy, cfg.output_dir, config)
    config._save_pretrained(cfg.output_dir)
    logging.info("Exported true FP8-only SmolVLA checkpoint to %s", fp8_file)
    return fp8_file


def main() -> None:
    init_logging()
    export_smolvla_fp8()


if __name__ == "__main__":
    main()
