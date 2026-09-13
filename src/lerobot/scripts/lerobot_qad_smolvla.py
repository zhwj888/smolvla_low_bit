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

"""Single-GPU SmolVLA ModelOpt QAD: python -m lerobot.scripts.lerobot_qad_smolvla.

This standalone trainer leaves PTQ and both Transformer Engine training paths untouched.
"""

import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Sampler, Subset

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modelopt_qad import (
    QAD_FORMAT,
    QAD_MANIFEST_NAME,
    QAD_TRAINING_STATE_NAME,
    QAD_WEIGHTS_NAME,
    load_bf16_teacher,
    load_qad_student,
    modelopt_modules,
    qad_forward,
    quantizer_signature,
    read_student_manifest,
)
from lerobot.policies.smolvla.qad_sampling import QAD_SAMPLING_NAME, select_qad_samples
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.random_utils import deserialize_rng_state, serialize_rng_state, set_seed
from lerobot.utils.utils import init_logging


@dataclass
class SmolVLAQADConfig:
    ptq_path: Path
    dataset: DatasetConfig
    output_dir: Path
    teacher_path: str | None = None  # Defaults to the PTQ manifest's original BF16 source.
    resume_from: Path | None = None  # A numbered QAD checkpoint; output_dir must be new.
    steps: int = 0  # 0 = one pass over the selected subset; otherwise total optimizer steps.
    sample_fraction: float = 0.1
    balanced_sampling: bool = True
    batch_size: int = 4
    num_workers: int = 4
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    distillation_weight: float = 1.0
    task_weight: float = 0.0
    save_freq: int = 250
    log_freq: int = 10
    seed: int = 1000
    device: str = "cuda"

    def __post_init__(self):
        for key in ("batch_size", "save_freq", "log_freq"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive.")
        if self.steps < 0:
            raise ValueError("steps must be nonnegative (0 means one pass over the selected subset).")
        if not math.isfinite(self.sample_fraction) or not 0 < self.sample_fraction <= 1:
            raise ValueError("sample_fraction must be in (0, 1].")
        if self.num_workers < 0 or self.warmup_steps < 0:
            raise ValueError("num_workers and warmup_steps must be nonnegative.")
        for key in ("learning_rate", "grad_clip_norm", "distillation_weight"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive.")
        for key in ("weight_decay", "task_weight"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative.")
        if not self.device.startswith("cuda"):
            raise ValueError("ModelOpt NVFP4 QAD requires CUDA.")
        if self.dataset.streaming or self.dataset.image_transforms.enable:
            raise ValueError("This QAD trainer requires a map-style dataset without random image transforms.")


class EpochBatchSampler(Sampler[list[int]]):
    """Deterministic shuffle with an explicit consumed-batch cursor, safe against prefetch."""

    def __init__(self, size: int, batch_size: int, seed: int, epoch: int = 0, offset: int = 0):
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.epoch, self.offset = epoch, offset

    def __iter__(self):
        indices = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + self.epoch))
        for start in range(self.offset * self.batch_size, self.size, self.batch_size):
            yield indices[start : start + self.batch_size].tolist()

    def __len__(self):
        return max(0, math.ceil(self.size / self.batch_size) - self.offset)


def make_qad_processors(student, student_path, teacher, teacher_path, device):
    """Preserve saved stats and rename_map; change only the runtime device."""
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=student.config,
        pretrained_path=str(student_path),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    teacher_pre, _ = make_pre_post_processors(
        policy_cfg=teacher.config,
        pretrained_path=str(teacher_path),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    student_norm = next(s for s in preprocessor.steps if isinstance(s, NormalizerProcessorStep))
    teacher_norm = next(s for s in teacher_pre.steps if isinstance(s, NormalizerProcessorStep))
    for feature in (OBS_STATE, ACTION):
        a, b = student_norm.stats.get(feature, {}), teacher_norm.stats.get(feature, {})
        if a.keys() != b.keys() or any(
            not torch.equal(torch.as_tensor(a[k]).cpu().float(), torch.as_tensor(b[k]).cpu().float())
            for k in a
        ):
            raise ValueError(
                f"PTQ/teacher normalization differs for {feature}; shared QAD inputs would be invalid."
            )
    return preprocessor, postprocessor


def _write_json(path: Path, value: Any):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def _save_checkpoint(
    cfg, student, pre, post, optimizer, step, epoch, offset, teacher_path, dataset_size, selection
):
    mto, _ = modelopt_modules()
    destination = cfg.output_dir / "checkpoints" / f"{step:06d}"
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    _write_json(temporary / QAD_SAMPLING_NAME, selection)
    mto.save(student, temporary / QAD_WEIGHTS_NAME)
    student.config._save_pretrained(temporary)
    pre.save_pretrained(temporary)
    post.save_pretrained(temporary)
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "batch_offset": offset,
            "optimizer": optimizer.state_dict(),
            "rng": serialize_rng_state(),
            "config": asdict(cfg),
            "dataset_size": dataset_size,
        },
        temporary / QAD_TRAINING_STATE_NAME,
    )
    _write_json(
        temporary / QAD_MANIFEST_NAME,
        {
            "format": QAD_FORMAT,
            "fake_quant": True,
            "compressed": False,
            "transformer_engine_used": False,
            "source_ptq": str(cfg.ptq_path.resolve()),
            "teacher_path": teacher_path,
            "step": step,
            "student_master_dtype": "float32",
            "forward_autocast_dtype": "bfloat16",
            "distillation_target": "action_velocity",
            "distillation_weight": cfg.distillation_weight,
            "task_weight": cfg.task_weight,
            "quantizers": quantizer_signature(student),
            "trainable_parameter_names": [n for n, p in student.named_parameters() if p.requires_grad],
        },
    )
    temporary.rename(destination)
    logging.info("Saved QAD checkpoint %s", destination)
    return destination


def _resume_state(cfg, teacher_path, dataset_size, selection):
    if cfg.resume_from is None:
        return None
    manifest, _ = read_student_manifest(cfg.resume_from)
    if manifest["format"] != QAD_FORMAT or manifest["teacher_path"] != teacher_path:
        raise ValueError("Resume requires a QAD checkpoint with the same teacher.")
    state = torch.load(cfg.resume_from / QAD_TRAINING_STATE_NAME, map_location="cpu", weights_only=False)
    sampling_path = cfg.resume_from / QAD_SAMPLING_NAME
    if sampling_path.is_file():
        saved_selection = json.loads(sampling_path.read_text())
        for key in ("version", "strategy", "sample_fraction", "seed", "source_fingerprint", "samples"):
            if saved_selection[key] != selection[key]:
                raise ValueError(f"Resume sample selection differs: {key}")
    elif cfg.balanced_sampling or cfg.sample_fraction != 1.0:
        raise ValueError("Legacy QAD resume requires balanced_sampling=false and sample_fraction=1.0.")
    for key, legacy_default in (("sample_fraction", 1.0), ("balanced_sampling", False)):
        if getattr(cfg, key) != state["config"].get(key, legacy_default):
            raise ValueError(f"Resume configuration differs: {key}")
    if state["dataset_size"] != dataset_size:
        raise ValueError("Resume dataset size differs.")
    for key in (
        "dataset",
        "batch_size",
        "seed",
        "learning_rate",
        "warmup_steps",
        "weight_decay",
        "grad_clip_norm",
        "distillation_weight",
        "task_weight",
        "ptq_path",
    ):
        if asdict(cfg)[key] != state["config"][key]:
            raise ValueError(f"Resume configuration differs: {key}")
    if state["step"] >= cfg.steps:
        raise ValueError("steps must exceed the resumed optimizer step.")
    return state


def run_qad(cfg: SmolVLAQADConfig) -> Path:
    """Train and save only the student; teacher and source checkpoints are read-only."""
    init_logging()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use this QAD entry point as a single process on one GPU.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NVFP4 QAD.")
    if cfg.output_dir.exists():
        raise FileExistsError(f"Choose a new QAD output directory: {cfg.output_dir}")
    set_seed(cfg.seed)
    device_index = torch.device(cfg.device).index
    torch.cuda.set_device(torch.cuda.current_device() if device_index is None else device_index)
    manifest, _ = read_student_manifest(cfg.ptq_path)
    teacher_path = cfg.teacher_path or manifest.get("source_policy") or manifest.get("teacher_path")
    if not teacher_path:
        raise ValueError("Specify teacher_path: the source manifest has no BF16 teacher path.")
    student_path = cfg.resume_from or cfg.ptq_path
    student = load_qad_student(student_path, cfg.device)
    teacher = load_bf16_teacher(teacher_path, student.config, cfg.device)
    preprocessor, postprocessor = make_qad_processors(
        student, student_path, teacher, teacher_path, cfg.device
    )
    pipeline_cfg = TrainPipelineConfig(dataset=cfg.dataset, policy=student.config, output_dir=cfg.output_dir)
    dataset, _ = make_train_eval_datasets(pipeline_cfg)
    if len(dataset) == 0:
        raise ValueError("QAD training dataset is empty.")
    dataset_meta = dataset.meta
    indices, selection = select_qad_samples(dataset, cfg.sample_fraction, cfg.seed, cfg.balanced_sampling)
    dataset = Subset(dataset, indices)
    if cfg.steps == 0:
        cfg.steps = math.ceil(len(dataset) / cfg.batch_size)
    state = _resume_state(cfg, teacher_path, len(dataset), selection)
    parameters = [p for p in student.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable student parameters.")
    optimizer = torch.optim.AdamW(parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    step, epoch, offset = 0, 0, 0
    if state:
        optimizer.load_state_dict(state["optimizer"])
        step, epoch, offset = state["step"], state["epoch"], state["batch_offset"]
        deserialize_rng_state(state["rng"])
    sampler = EpochBatchSampler(len(dataset), cfg.batch_size, cfg.seed, epoch, offset)
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=lerobot_collate_fn if dataset_meta.has_language_columns else None,
        persistent_workers=cfg.num_workers > 0,
        multiprocessing_context="spawn" if cfg.num_workers > 0 else None,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    signature = quantizer_signature(student)
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(cfg.output_dir / "qad_config.json", asdict(cfg))
    _write_json(cfg.output_dir / QAD_SAMPLING_NAME, selection)
    logging.info(
        "QAD selection: %d/%d samples, %d tasks, %d episodes, stages=%s, steps=%d",
        len(dataset),
        selection["available_samples"],
        selection["selected_tasks"],
        selection["selected_episodes"],
        selection["stage_counts"],
        cfg.steps,
    )
    logging.info("QAD: %d dataset samples, teacher=%s, student=%s", len(dataset), teacher_path, student_path)
    student.train()
    checkpoint = None
    with (cfg.output_dir / "metrics.jsonl").open("w") as log:
        while step < cfg.steps:
            sampler.epoch, sampler.offset = epoch, offset
            for batch in dataloader:
                for key in dataset_meta.camera_keys:
                    if key in batch and batch[key].dtype == torch.uint8:
                        batch[key] = batch[key].float() / 255.0
                batch = preprocessor(batch)
                actions = student.prepare_action(batch)
                if actions.ndim != 3:
                    raise ValueError("QAD requires batched action chunks with shape (B, T, D).")
                noise = student.model.sample_noise(actions.shape, actions.device)
                time = student.model.sample_time(actions.shape[0], actions.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, metrics = qad_forward(
                        student,
                        teacher,
                        batch,
                        noise,
                        time,
                        distillation_weight=cfg.distillation_weight,
                        task_weight=cfg.task_weight,
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite QAD loss at step {step + 1}.")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, cfg.grad_clip_norm, error_if_nonfinite=True)
                lr = cfg.learning_rate * min(1.0, (step + 1) / max(1, cfg.warmup_steps))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
                step += 1
                offset += 1
                row = {
                    "step": step,
                    "learning_rate": lr,
                    "grad_norm": float(norm),
                    **{k: float(v) for k, v in metrics.items()},
                }
                log.write(json.dumps(row) + "\n")
                log.flush()
                if step == 1 or step % cfg.log_freq == 0:
                    logging.info("QAD %s", row)
                if step % cfg.save_freq == 0 or step == cfg.steps:
                    if quantizer_signature(student) != signature:
                        raise RuntimeError("The PTQ quantizer assignment changed during QAD.")
                    checkpoint = _save_checkpoint(
                        cfg,
                        student,
                        preprocessor,
                        postprocessor,
                        optimizer,
                        step,
                        epoch,
                        offset,
                        teacher_path,
                        len(dataset),
                        selection,
                    )
                if step >= cfg.steps:
                    break
            else:
                epoch += 1
                offset = 0
    return checkpoint


@parser.wrap()
def qad_smolvla(cfg: SmolVLAQADConfig):
    return run_qad(cfg)


if __name__ == "__main__":
    qad_smolvla()
