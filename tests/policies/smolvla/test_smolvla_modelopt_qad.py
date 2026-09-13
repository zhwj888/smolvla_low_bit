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
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs.default import DatasetConfig
from lerobot.policies.smolvla.modelopt_qad import masked_velocity_mse, qad_forward, read_student_manifest
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.scripts import lerobot_qad_smolvla as qad_script
from lerobot.scripts.lerobot_qad_smolvla import EpochBatchSampler, SmolVLAQADConfig


class ToyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.action_out_proj = nn.Linear(4, 4, bias=False)
        self.config = SimpleNamespace(action_feature=SimpleNamespace(shape=(2,)))
        self.seen = []

    def forward(self, batch, noise, time):
        self.seen.append((noise, time, torch.is_grad_enabled(), self.training))
        batch["observation.state"].add_(1)
        velocity = self.model.action_out_proj(noise + time[:, None, None])
        return velocity.square().mean(), {}


def test_distillation_excludes_padded_frames_and_action_dimensions():
    student = torch.tensor([[[1.0, 3.0, 100.0], [900.0, 900.0, 900.0]]], requires_grad=True)
    teacher = torch.zeros_like(student, requires_grad=True)
    loss = masked_velocity_mse(student, teacher, 2, torch.tensor([[False, True]]))
    assert loss.item() == 5
    loss.backward()
    torch.testing.assert_close(student.grad, torch.tensor([[[1.0, 3.0, 0.0], [0.0, 0.0, 0.0]]]))
    assert teacher.grad is None


def test_fully_padded_batch_has_zero_loss_and_grad():
    student = torch.ones(2, 3, 4, requires_grad=True)
    loss = masked_velocity_mse(student, torch.zeros_like(student), 2, torch.ones(2, 3, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0
    assert torch.count_nonzero(student.grad) == 0


@pytest.mark.parametrize("shape,pad", [((2, 3), None), ((1, 2, 3), torch.zeros(1, 3))])
def test_distillation_rejects_wrong_shapes(shape, pad):
    with pytest.raises(ValueError):
        masked_velocity_mse(torch.zeros(shape), torch.zeros(shape), 2, pad)


def test_teacher_is_no_grad_and_inputs_match_without_mutating_batch():
    student, teacher = ToyPolicy(), ToyPolicy()
    batch = {"observation.state": torch.zeros(2, 4)}
    noise, time = torch.randn(2, 3, 4), torch.rand(2)
    teacher.train()
    before = {k: v.clone() for k, v in teacher.state_dict().items()}
    loss, metrics = qad_forward(student, teacher, batch, noise, time)
    loss.backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert student.model.action_out_proj.weight.grad.abs().sum() > 0
    assert student.seen[0][0] is teacher.seen[0][0] is noise
    assert student.seen[0][1] is teacher.seen[0][1] is time
    assert teacher.seen[0][2:] == (False, False)
    assert torch.count_nonzero(batch["observation.state"]) == 0
    for k, v in teacher.state_dict().items():
        torch.testing.assert_close(v, before[k])
    assert not student.model.action_out_proj._forward_hooks
    assert not teacher.model.action_out_proj._forward_hooks
    torch.testing.assert_close(loss.detach(), metrics["distillation_loss"])


def test_optional_task_loss_is_weighted_separately():
    student, teacher = ToyPolicy(), ToyPolicy()
    loss, metrics = qad_forward(
        student,
        teacher,
        {"observation.state": torch.zeros(1, 4)},
        torch.ones(1, 3, 4),
        torch.ones(1),
        distillation_weight=2,
        task_weight=0.25,
    )
    torch.testing.assert_close(loss.detach(), 2 * metrics["distillation_loss"] + 0.25 * metrics["task_loss"])


def test_capture_hook_removed_when_forward_fails():
    student, teacher = ToyPolicy(), ToyPolicy()
    with pytest.raises(KeyError):
        qad_forward(student, teacher, {}, torch.ones(1, 3, 4), torch.ones(1))
    assert not student.model.action_out_proj._forward_hooks
    assert not teacher.model.action_out_proj._forward_hooks


def test_sampler_resume_uses_consumed_cursor_not_prefetched_cursor():
    sampler = EpochBatchSampler(11, 3, seed=42)
    original = list(sampler)  # Pretend all batches have been prefetched but only two consumed.
    resumed = list(EpochBatchSampler(11, 3, seed=42, offset=2))
    assert resumed == original[2:]
    assert sorted(x for batch in original for x in batch) == list(range(11))
    assert len(original[-1]) == 2
    assert list(EpochBatchSampler(11, 3, seed=42, epoch=1)) != original
    assert len(EpochBatchSampler(11, 3, seed=42, offset=4)) == 0


def test_processor_restore_only_overrides_device(monkeypatch):
    calls = []
    stats = {"observation.state": {"mean": [0.5], "std": [0.1]}, "action": {"mean": [0.0], "std": [1.0]}}
    normalizer = NormalizerProcessorStep(features={}, norm_map={}, stats=stats)
    pre = SimpleNamespace(steps=[normalizer])
    post = object()

    def factory(**kwargs):
        calls.append(kwargs)
        return pre, post

    monkeypatch.setattr(qad_script, "make_pre_post_processors", factory)
    policy = SimpleNamespace(config=object())
    actual = qad_script.make_qad_processors(policy, "student", policy, "teacher", "cuda:0")
    assert actual == (pre, post)
    assert len(calls) == 2
    for call in calls:
        assert call["preprocessor_overrides"] == {"device_processor": {"device": "cuda:0"}}
        assert "postprocessor_overrides" not in call
        assert "dataset_stats" not in call


def test_mismatched_teacher_normalization_is_rejected(monkeypatch):
    preprocessors = iter(
        [
            SimpleNamespace(
                steps=[NormalizerProcessorStep(features={}, norm_map={}, stats={"action": {"mean": [x]}})]
            )
            for x in [0.0, 1.0]
        ]
    )
    monkeypatch.setattr(qad_script, "make_pre_post_processors", lambda **kw: (next(preprocessors), None))
    policy = SimpleNamespace(config=object())
    with pytest.raises(ValueError, match="normalization differs"):
        qad_script.make_qad_processors(policy, "student", policy, "teacher", "cuda")


def test_manifest_rejects_compressed_checkpoint(tmp_path):
    (tmp_path / "modelopt_ptq_manifest.json").write_text(
        json.dumps(
            {
                "format": "lerobot-smolvla-modelopt-fake-quant-v1",
                "fake_quant": True,
                "compressed": True,
            }
        )
    )
    with pytest.raises(ValueError, match="master weights"):
        read_student_manifest(tmp_path)


@pytest.mark.parametrize(
    "kwargs", [{"task_weight": -1}, {"distillation_weight": 0}, {"learning_rate": float("nan")}]
)
def test_invalid_training_config(tmp_path, kwargs):
    with pytest.raises(ValueError):
        SmolVLAQADConfig(
            ptq_path=tmp_path, dataset=DatasetConfig(repo_id="test/data"), output_dir=tmp_path, **kwargs
        )
