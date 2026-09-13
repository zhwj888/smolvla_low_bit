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

"""Reproducible, without-replacement QAD subsets; no changes to dataset statistics."""

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from typing import Any

QAD_SAMPLING_NAME = "qad_sampling.json"
STAGES = ("early", "middle", "late")


def _allocate(capacities: dict, count: int, order: list) -> dict:
    """Equal quotas, redistributing a depleted group's unused quota to other groups."""
    quotas = dict.fromkeys(capacities, 0)
    pending = [key for key in order if capacities[key] > 0]
    while count and pending:
        for key in pending:
            quotas[key] += 1
            count -= 1
            if not count:
                break
        pending = [key for key in pending if quotas[key] < capacities[key]]
    if count:
        raise ValueError("Requested more unique samples than available.")
    return quotas


def select_qad_samples(
    dataset, fraction: float, seed: int, balanced: bool = True
) -> tuple[list[int], dict[str, Any]]:
    """Balance tasks, then episodes within each task, then temporal thirds per episode.

    Selection uses metadata only, never decoding images or recomputing mean/std.
    Quotas differ by at most one where capacity permits. Stage remainder priority
    rotates across episodes to avoid always giving extra samples to early frames.
    The returned indices are local to the already episode-filtered training dataset.
    """
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("sample_fraction must be in (0, 1].")
    if not len(dataset):
        raise ValueError("QAD training dataset is empty.")
    count = math.ceil(len(dataset) * fraction)
    hf = getattr(dataset, "hf_dataset", None)
    columns = set(getattr(hf, "column_names", []))
    required = {"task_index", "episode_index", "frame_index"}
    if not required.issubset(columns):
        raise ValueError("QAD sampling requires task_index, episode_index and frame_index metadata.")
    keys = ["task_index", "episode_index", "frame_index"]
    if "index" in columns:
        keys.append("index")
    # Avoid per-row tensor formatting and loading observation/video columns.
    data = hf.select_columns(keys).with_format(None).to_dict() if hasattr(hf, "select_columns") else hf
    if any(len(data[key]) != len(dataset) for key in keys):
        raise ValueError("QAD metadata length does not match the selected training dataset.")
    rows = list(zip(*(map(int, data[key]) for key in keys), strict=True))
    fingerprint = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
    groups = defaultdict(lambda: defaultdict(list))
    records = {}
    for index, row in enumerate(rows):
        task, episode, frame = row[:3]
        groups[task][episode].append((frame, index))
    for task, episodes in groups.items():
        for episode, values in episodes.items():
            values.sort()
            for position, (frame, index) in enumerate(values):
                records[index] = {
                    "dataset_index": index,
                    "source_index": rows[index][3] if len(keys) == 4 else index,
                    "task_index": task,
                    "episode_index": episode,
                    "frame_index": frame,
                    "stage": STAGES[position * 3 // len(values)],
                }
    rng = random.Random(seed)
    indices = []
    if balanced:
        if count < len(groups):
            raise ValueError(
                f"Selected {count} samples cannot cover all {len(groups)} tasks; increase sample_fraction."
            )
        tasks = sorted(groups)
        rng.shuffle(tasks)
        task_quotas = _allocate({t: sum(map(len, groups[t].values())) for t in tasks}, count, tasks)
        for task_position, task in enumerate(tasks):
            episodes = sorted(groups[task])
            rng.shuffle(episodes)
            quotas = _allocate({e: len(groups[task][e]) for e in episodes}, task_quotas[task], episodes)
            cursor = task_position % 3
            for episode in episodes:
                pools = {stage: [] for stage in STAGES}
                for _, index in groups[task][episode]:
                    pools[records[index]["stage"]].append(index)
                order = list(STAGES[cursor:] + STAGES[:cursor])
                stage_quotas = _allocate({s: len(pools[s]) for s in STAGES}, quotas[episode], order)
                cursor = (cursor + quotas[episode]) % 3
                for stage in STAGES:
                    indices.extend(rng.sample(pools[stage], stage_quotas[stage]))
    else:
        indices = (
            rng.sample(range(len(dataset)), count) if count < len(dataset) else list(range(len(dataset)))
        )
    samples = [records[index] for index in indices]
    task_counts = {}
    for task in sorted(groups):
        selected = [r for r in samples if r["task_index"] == task]
        episode_counts = Counter(r["episode_index"] for r in selected)
        task_counts[str(task)] = {
            "available_samples": sum(map(len, groups[task].values())),
            "selected_samples": len(selected),
            "available_episodes": len(groups[task]),
            "selected_episodes": len(episode_counts),
            "episode_counts": dict(sorted(episode_counts.items())),
            "stage_counts": {s: sum(r["stage"] == s for r in selected) for s in STAGES},
        }
    return indices, {
        "version": 1,
        "strategy": "task_episode_stage_balanced" if balanced else "random",
        "sample_fraction": fraction,
        "seed": seed,
        "source_fingerprint": fingerprint,
        "available_samples": len(dataset),
        "selected_samples": len(indices),
        "available_tasks": len(groups),
        "selected_tasks": sum(bool(t["selected_samples"]) for t in task_counts.values()),
        "selected_episodes": len({(r["task_index"], r["episode_index"]) for r in samples}),
        "stage_counts": {s: sum(r["stage"] == s for r in samples) for s in STAGES},
        "task_counts": task_counts,
        "samples": samples,
    }
