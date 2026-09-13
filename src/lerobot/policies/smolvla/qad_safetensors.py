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

"""Lossless JSON encoding for ModelOpt metadata alongside QAD safetensors weights."""

import json
from pathlib import Path
from typing import Any

import torch

QAD_SAFE_FORMAT = "lerobot-smolvla-modelopt-qad-safetensors-v1"
QAD_SAFE_MANIFEST_NAME = "modelopt_qad_safetensors_manifest.json"
QAD_SAFE_WEIGHTS_NAME = "model.safetensors"
QAD_SAFE_STATE_NAME = "modelopt_state.json"


def encode_state(value: Any) -> Any:
    # Tag all containers to preserve integer keys, tuples, and torch.Size exactly.
    if isinstance(value, torch.dtype):
        return {"type": "dtype", "value": str(value).removeprefix("torch.")}
    if isinstance(value, dict):
        return {"type": "dict", "value": [[encode_state(k), encode_state(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        kind = "size" if isinstance(value, torch.Size) else "tuple" if isinstance(value, tuple) else "list"
        return {"type": kind, "value": [encode_state(v) for v in value]}
    if value is None or type(value) in (bool, int, float, str):
        return value
    # Only allow this explicit ModelOpt type; never serialize arbitrary Python objects.
    if (
        type(value).__module__ == "modelopt.torch.quantization.config"
        and type(value).__name__ == "RotateConfig"
    ):
        return {"type": "rotate_config", "value": encode_state(value.model_dump(mode="python"))}
    raise TypeError(f"Unsupported ModelOpt metadata type: {type(value).__name__}")


def decode_state(value: Any) -> Any:
    if value is None or type(value) in (bool, int, float, str):
        return value
    if not isinstance(value, dict) or set(value) != {"type", "value"}:
        raise ValueError("Invalid tagged ModelOpt JSON metadata.")
    kind, data = value["type"], value["value"]
    if kind == "dtype":
        dtype = getattr(torch, data, None) if isinstance(data, str) else None
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unsupported torch dtype: {data}")
        return dtype
    if kind == "rotate_config":
        from modelopt.torch.quantization.config import RotateConfig

        return RotateConfig.model_validate(decode_state(data))
    if kind == "dict":
        return {decode_state(k): decode_state(v) for k, v in data}
    if kind in ("tuple", "list", "size"):
        values = [decode_state(v) for v in data]
        return {"tuple": tuple, "list": list, "size": torch.Size}[kind](values)
    raise ValueError(f"Unsupported ModelOpt JSON tag: {kind}")


def read_modelopt_state(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text())
    if payload.get("format") != QAD_SAFE_FORMAT:
        raise ValueError("Unsupported QAD ModelOpt JSON format.")
    return decode_state(payload["state"])
