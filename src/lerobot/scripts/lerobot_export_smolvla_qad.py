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

"""Export a trusted PTQ/QAD .pt to safetensors + JSON, retaining NVFP4 fake quantization."""

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save_file

from lerobot.configs import parser
from lerobot.policies.smolvla.modelopt_ptq import MODELOPT_PTQ_FORMAT
from lerobot.policies.smolvla.modelopt_qad import QAD_FORMAT, read_student_manifest
from lerobot.policies.smolvla.qad_safetensors import (
    QAD_SAFE_FORMAT,
    QAD_SAFE_MANIFEST_NAME,
    QAD_SAFE_STATE_NAME,
    QAD_SAFE_WEIGHTS_NAME,
    encode_state,
)
from lerobot.utils.utils import init_logging


@dataclass
class QADExportConfig:
    checkpoint: Path
    output_dir: Path


def export_qad(cfg: QADExportConfig) -> Path:
    source, destination = cfg.checkpoint, cfg.output_dir
    if destination.exists():
        raise FileExistsError(f"Choose a new export directory: {destination}")
    manifest, weights = read_student_manifest(source)
    if manifest["format"] not in (QAD_FORMAT, MODELOPT_PTQ_FORMAT):
        raise ValueError("Export requires a ModelOpt PTQ or QAD .pt checkpoint.")
    # Only the conversion of the existing trusted .pt uses pickle; exported loading does not.
    checkpoint = torch.load(source / weights, map_location="cpu", weights_only=False, mmap=True)
    state_json = json.dumps(
        {"format": QAD_SAFE_FORMAT, "state": encode_state(checkpoint["modelopt_state"])}, allow_nan=False
    )
    # Clone to break shared storage without dropping keys or changing dtype/calibration scales.
    tensors = {k: v.detach().cpu().contiguous().clone() for k, v in checkpoint["model_state_dict"].items()}
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    save_file(tensors, temporary / QAD_SAFE_WEIGHTS_NAME, metadata={"format": "pt"})
    (temporary / QAD_SAFE_STATE_NAME).write_text(state_json + "\n")
    for name in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
        shutil.copy2(source / name, temporary / name)
    for file in source.glob("policy_*processor*.safetensors"):
        shutil.copy2(file, temporary / file.name)
    for name in ("tokenizer", "qad_sampling.json"):
        if (source / name).is_dir():
            shutil.copytree(source / name, temporary / name)
        elif (source / name).is_file():
            shutil.copy2(source / name, temporary / name)
    exported = {
        **manifest,
        "format": QAD_SAFE_FORMAT,
        ("source_qad" if manifest["format"] == QAD_FORMAT else "source_ptq"): str(source.resolve()),
        "weights_file": QAD_SAFE_WEIGHTS_NAME,
        "modelopt_state_file": QAD_SAFE_STATE_NAME,
        "tensor_count": len(tensors),
    }
    (temporary / QAD_SAFE_MANIFEST_NAME).write_text(json.dumps(exported, indent=2) + "\n")
    temporary.rename(destination)
    logging.info("Exported %d tensors without dtype conversion to %s", len(tensors), destination)
    return destination


@parser.wrap()
def export_main(cfg: QADExportConfig):
    export_qad(cfg)


if __name__ == "__main__":
    init_logging()
    export_main()
