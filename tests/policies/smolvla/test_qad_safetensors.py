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

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modelopt_ptq import (
    MODELOPT_PTQ_CHECKPOINT_NAME,
    MODELOPT_PTQ_FORMAT,
    MODELOPT_PTQ_MANIFEST_NAME,
)
from lerobot.policies.smolvla.modelopt_qad import QAD_FORMAT, QAD_MANIFEST_NAME, QAD_WEIGHTS_NAME
from lerobot.policies.smolvla.qad_safetensors import (
    QAD_SAFE_MANIFEST_NAME,
    QAD_SAFE_STATE_NAME,
    decode_state,
    encode_state,
    read_modelopt_state,
)
from lerobot.scripts import lerobot_eval_smolvla_qad as evaluation
from lerobot.scripts.lerobot_export_smolvla_qad import QADExportConfig, export_qad


def test_metadata_roundtrip_preserves_types_and_integer_keys():
    state = {"shape": torch.Size([4, 16]), "block": {-1: 16, "scale_bits": (4, 3)}, "dtype": torch.bfloat16}
    restored = decode_state(json.loads(json.dumps(encode_state(state))))
    assert restored == state
    assert type(restored["shape"]) is torch.Size
    assert type(restored["block"]["scale_bits"]) is tuple
    assert -1 in restored["block"]


def test_metadata_rejects_arbitrary_objects_and_unknown_types():
    with pytest.raises(TypeError):
        encode_state(object())
    with pytest.raises(ValueError):
        decode_state({"type": "dtype", "value": "load"})
    with pytest.raises(ValueError):
        decode_state({"type": "object", "value": "unknown"})


@pytest.mark.parametrize("with_rht", [False, True])
@pytest.mark.parametrize("source_kind", ["ptq", "qad"])
def test_export_preserves_shared_weights_scales_and_processors(tmp_path, with_rht, source_kind):
    source = tmp_path / "source"
    source.mkdir()
    fmt, manifest_name, weights_name = (
        (MODELOPT_PTQ_FORMAT, MODELOPT_PTQ_MANIFEST_NAME, MODELOPT_PTQ_CHECKPOINT_NAME)
        if source_kind == "ptq"
        else (QAD_FORMAT, QAD_MANIFEST_NAME, QAD_WEIGHTS_NAME)
    )
    (source / manifest_name).write_text(json.dumps({"format": fmt, "fake_quant": True}))
    for name in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
        (source / name).write_text("{}\n")
    (source / "policy_preprocessor_step_5_normalizer_processor.safetensors").write_bytes(b"stats unchanged")
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    tensors = {"weight": tensor, "shared": tensor, "scale": torch.tensor(1.5, dtype=torch.bfloat16)}
    metadata = {"block": {-1: 16}, "dtype": torch.float32}
    if with_rht:
        config = pytest.importorskip("modelopt.torch.quantization.config")
        metadata["rotation"] = config.RotateConfig(enable=True, rotate_fp32=False, block_size=16)
    torch.save({"modelopt_state": metadata, "model_state_dict": tensors}, source / weights_name)
    output = tmp_path / "export"
    export_qad(QADExportConfig(source, output))
    exported_manifest = json.loads((output / QAD_SAFE_MANIFEST_NAME).read_text())
    assert exported_manifest[f"source_{source_kind}"] == str(source.resolve())
    restored = load_file(output / "model.safetensors")
    assert restored.keys() == tensors.keys()
    for key in tensors:
        torch.testing.assert_close(restored[key], tensors[key], rtol=0, atol=0)
    assert read_modelopt_state(output / QAD_SAFE_STATE_NAME) == metadata
    assert (
        output / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    ).read_bytes() == b"stats unchanged"
    assert not list(output.glob("*.pt"))
    with pytest.raises(FileExistsError):
        export_qad(QADExportConfig(source, output))


@pytest.mark.parametrize("rename", [{}, {"old": "new"}])
def test_eval_preserves_saved_processors_unless_explicit_rename(monkeypatch, rename):
    calls = []
    monkeypatch.setattr(evaluation, "make_pre_post_processors", lambda **kwargs: calls.append(kwargs))
    policy = SimpleNamespace(config=SimpleNamespace(device="cuda"))
    evaluation.make_eval_processors(policy, "checkpoint", rename)
    expected = {"device_processor": {"device": "cuda"}}
    if rename:
        expected["rename_observations_processor"] = {"rename_map": rename}
    assert calls[0]["preprocessor_overrides"] == expected
    assert "dataset_stats" not in calls[0]


def test_eval_applies_action_steps_and_freezes_student(monkeypatch):
    saved = SmolVLAConfig(device="cuda")
    requested = SmolVLAConfig(device="cuda", use_amp=True, n_action_steps=1, num_steps=10)
    requested.pretrained_path = Path("checkpoint")
    monkeypatch.setattr(SmolVLAConfig, "from_pretrained", lambda *args: saved)
    policy = torch.nn.Linear(2, 2)
    policy.config = SmolVLAConfig(device="cuda")
    reset_calls = []
    policy.reset = lambda: reset_calls.append(policy.config.n_action_steps)
    monkeypatch.setattr(evaluation, "load_qad_student", lambda *args: policy)
    result = evaluation.load_eval_student(requested)
    assert result.config.n_action_steps == 1 and result.config.num_steps == 10
    assert reset_calls == [1]
    assert not result.training and all(not p.requires_grad for p in result.parameters())
    requested.chunk_size += 1
    with pytest.raises(ValueError, match="chunk_size"):
        evaluation.load_eval_student(requested)


@pytest.mark.parametrize("enable,fp32,block", [(True, False, 16), (True, True, None), (False, False, 32)])
def test_rotation_metadata_roundtrip_preserves_type_and_settings(enable, fp32, block):
    config = pytest.importorskip("modelopt.torch.quantization.config")
    rotation = config.RotateConfig(enable=enable, rotate_fp32=fp32, block_size=block)
    original = {"quantizer_state": {"input_quantizer": {"_rotate": rotation}}, "legacy_rotate": False}
    restored = decode_state(json.loads(json.dumps(encode_state(original), allow_nan=False)))
    actual = restored["quantizer_state"]["input_quantizer"]["_rotate"]
    assert type(actual) is config.RotateConfig
    assert actual.model_dump() == rotation.model_dump()
    assert restored == original


def test_rotation_metadata_rejects_invalid_block_size():
    pytest.importorskip("modelopt.torch.quantization.config")
    payload = {"type": "rotate_config", "value": encode_state({"enable": True, "block_size": -1})}
    with pytest.raises(ValueError):
        decode_state(payload)
