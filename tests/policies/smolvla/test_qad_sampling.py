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
import random
from dataclasses import asdict

import pytest
import torch

from lerobot.configs.default import DatasetConfig
from lerobot.policies.smolvla.qad_sampling import QAD_SAMPLING_NAME, select_qad_samples
from lerobot.scripts import lerobot_qad_smolvla as trainer


class Metadata(dict):
    @property
    def column_names(self):
        return list(self)


class Dataset:
    def __init__(self, task_episode_lengths):
        self.hf_dataset = Metadata({k: [] for k in ("task_index", "episode_index", "frame_index", "index")})
        for task, lengths in enumerate(task_episode_lengths):
            for episode, length in enumerate(lengths):
                for frame in range(length):
                    for key, value in zip(
                        self.hf_dataset, (task, episode, frame, 1000 + len(self)), strict=True
                    ):
                        self.hf_dataset[key].append(value)

    def __len__(self):
        return len(self.hf_dataset["index"])

    def __getitem__(self, index):
        raise AssertionError("Selection must not decode dataset observations.")


def test_task_episode_stage_balance_and_no_duplicates():
    ds = Dataset([[90] * 4, [30] * 4, [60] * 4])
    indices, report = select_qad_samples(ds, 0.1, 42)
    assert len(indices) == len(set(indices)) == 72
    assert report["selected_tasks"] == 3
    assert report["selected_episodes"] == 12
    for task in report["task_counts"].values():
        assert task["selected_samples"] == 24
        assert set(task["episode_counts"].values()) == {6}
        assert task["stage_counts"] == {"early": 8, "middle": 8, "late": 8}
    assert all(r["source_index"] == r["dataset_index"] + 1000 for r in report["samples"])


def test_remainders_and_short_episodes_are_redistributed_without_replacement():
    ds = Dataset([[1], [2], [30]])
    indices, report = select_qad_samples(ds, 0.5, 3)
    assert len(indices) == len(set(indices)) == 17
    assert [t["selected_samples"] for t in report["task_counts"].values()] == [1, 2, 14]


def test_seed_reproducibility_without_changing_global_rng():
    ds = Dataset([[90] * 3, [90] * 3])
    before = random.getstate()
    a = select_qad_samples(ds, 0.2, 17)
    assert a == select_qad_samples(ds, 0.2, 17)
    assert a[0] != select_qad_samples(ds, 0.2, 18)[0]
    assert random.getstate() == before


def test_episode_filter_local_indices_and_fingerprint():
    ds = Dataset([[30] * 2])
    ds.hf_dataset = Metadata({k: v[30:] for k, v in ds.hf_dataset.items()})
    indices, report = select_qad_samples(ds, 0.2, 42)
    assert all(0 <= i < 30 for i in indices)
    assert all(
        r["episode_index"] == 1 and r["source_index"] == r["dataset_index"] + 1030 for r in report["samples"]
    )
    ds.hf_dataset["index"][0] += 10000
    assert report["source_fingerprint"] != select_qad_samples(ds, 0.2, 42)[1]["source_fingerprint"]


def test_full_unbalanced_selection_preserves_legacy_index_order():
    ds = Dataset([[3], [9]])
    assert select_qad_samples(ds, 1, 42, balanced=False)[0] == list(range(len(ds)))
    assert sorted(select_qad_samples(ds, 1, 42)[0]) == list(range(len(ds)))


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1, float("nan")])
def test_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="sample_fraction"):
        select_qad_samples(Dataset([[30]]), fraction, 42)


def test_no_silent_random_fallback_and_too_small_budget():
    ds = Dataset([[3], [3], [3]])
    with pytest.raises(ValueError, match="cover all"):
        select_qad_samples(ds, 0.1, 42)
    del ds.hf_dataset["task_index"]
    with pytest.raises(ValueError, match="metadata"):
        select_qad_samples(ds, 0.5, 42)


def test_resume_validates_exact_selection(monkeypatch, tmp_path):
    cfg = trainer.SmolVLAQADConfig(
        ptq_path=tmp_path,
        dataset=DatasetConfig(repo_id="test/data"),
        output_dir=tmp_path / "out",
        resume_from=tmp_path,
        steps=2,
    )
    _, report = select_qad_samples(Dataset([[30] * 3]), cfg.sample_fraction, cfg.seed)
    (tmp_path / QAD_SAMPLING_NAME).write_text(json.dumps(report))
    state = {"step": 1, "dataset_size": report["selected_samples"], "config": asdict(cfg)}
    torch.save(state, tmp_path / trainer.QAD_TRAINING_STATE_NAME)
    monkeypatch.setattr(
        trainer,
        "read_student_manifest",
        lambda p: ({"format": trainer.QAD_FORMAT, "teacher_path": "teacher"}, "unused"),
    )
    assert trainer._resume_state(cfg, "teacher", report["selected_samples"], report)["step"] == 1
    report["samples"][0]["dataset_index"] += 1
    with pytest.raises(ValueError, match="sample selection differs"):
        trainer._resume_state(cfg, "teacher", report["selected_samples"], report)


def test_qad_defaults_select_ten_percent_for_one_pass(tmp_path):
    cfg = trainer.SmolVLAQADConfig(
        ptq_path=tmp_path, dataset=DatasetConfig(repo_id="test/data"), output_dir=tmp_path / "out"
    )
    assert cfg.balanced_sampling and cfg.sample_fraction == 0.1 and cfg.steps == 0
