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

"""Create a QAD-ready ModelOpt PTQ checkpoint from an original BF16 SmolVLA policy.

This command intentionally does not enter either Transformer Engine path. The source must be a
standard BF16 SmolVLA checkpoint (``use_fp8=false`` and ``use_nvfp4=false``), and the output keeps
BF16 master weights plus ModelOpt fake quantizers. Use ``--quantization_mode=auto`` to perform a
hardware-constrained NVFP4/FP8 sensitivity search, or ``nvfp4`` for a fixed NVFP4 assignment.
"""

import datetime as dt
import importlib.util
import json
import logging
import random
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.factory import ProcessorConfigKwargs
from lerobot.policies.smolvla.autoquant_budget import auto_quantize_with_budget, validate_exclusions
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modelopt_ptq import (
    MODELOPT_AUTOQUANT_STATE_NAME,
    MODELOPT_PTQ_CHECKPOINT_NAME,
    MODELOPT_PTQ_FORMAT,
    MODELOPT_PTQ_MANIFEST_NAME,
    MODELOPT_STATE_NAME,
    analyze_linear_eligibility,
    build_autoquant_formats,
    build_fixed_nvfp4_config,
    build_quantized_linear_report,
    disabled_layer_names,
    serialize_decisions,
)
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

MODELOPT_AUTOQUANT_CHECKPOINT_NAME = "modelopt_autoquant_checkpoint.pt"


@dataclass
class SmolVLAModelOptPTQConfig:
    """Configuration for BF16 SmolVLA PTQ and optional AutoQuant sensitivity search."""

    policy_path: str
    dataset: DatasetConfig
    output_dir: Path
    rename_map: dict[str, str] = field(default_factory=dict)

    # ``auto`` searches NVFP4/FP8/no-quant per eligible Linear. ``nvfp4`` uses a fixed recipe.
    quantization_mode: str = "auto"
    calibration_algorithm: str = "max"
    # MSE FP8-scale sweep uses static NVFP4 weights and keeps activations dynamic.
    fp8_scale_sweep: bool = False
    effective_bits: float = 11.25
    # Exclude fixed high-precision modules from the bit budget (auto mode only).
    cost_excluded_layers: list[str] = field(default_factory=list)
    calibration_batches: int = 32
    score_batches: int = 16

    # ModelOpt Hadamard rotation is applied to both Linear operands before NVFP4 quantization.
    enable_rht: bool = True
    rht_block_size: int | None = 16
    rht_rotate_fp32: bool = False

    batch_size: int = 8
    num_workers: int = 4
    seed: int = 1000
    device: str = "cuda"

    # TE 2.16 on GB10/SM12x requires feature dimensions divisible by 32 for real NVFP4 GEMM.
    feature_alignment: int = 32
    trainable_only: bool = True
    keep_vision_high_precision: bool = True
    keep_output_heads_high_precision: bool = True
    excluded_module_patterns: list[str] = field(default_factory=list)
    precompile_extensions: bool = False

    def __post_init__(self) -> None:
        if self.cost_excluded_layers and self.quantization_mode != "auto":
            raise ValueError("cost_excluded_layers is only supported in auto mode.")
        if self.quantization_mode not in ("auto", "nvfp4"):
            raise ValueError(
                "quantization_mode must be 'auto' or 'nvfp4', "
                f"got {self.quantization_mode!r}."
            )
        if self.calibration_algorithm not in ("max", "mse"):
            raise ValueError(
                "calibration_algorithm must be 'max' or 'mse', "
                f"got {self.calibration_algorithm!r}."
            )
        if self.fp8_scale_sweep and self.calibration_algorithm != "mse":
            raise ValueError("fp8_scale_sweep requires calibration_algorithm='mse'.")
        if self.effective_bits < 4.5 or self.effective_bits > 16:
            raise ValueError(f"effective_bits must be in [4.5, 16], got {self.effective_bits}.")
        if self.calibration_batches <= 0:
            raise ValueError("calibration_batches must be greater than zero.")
        if self.quantization_mode == "auto" and self.score_batches <= 0:
            raise ValueError("score_batches must be greater than zero in auto mode.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if self.feature_alignment <= 0:
            raise ValueError("feature_alignment must be greater than zero.")
        if self.rht_block_size is not None and (
            self.rht_block_size <= 0
            or self.rht_block_size & (self.rht_block_size - 1)
        ):
            raise ValueError("rht_block_size must be a positive power of two or null.")
        if not self.device.startswith("cuda"):
            raise ValueError("SmolVLA NVFP4 PTQ requires a CUDA device.")


@dataclass(frozen=True)
class _CalibrationBatch:
    batch: dict[str, Any]
    noise: torch.Tensor
    time: torch.Tensor


_CALIBRATION_STAGES = ("early", "middle", "late")


def _random_calibration_selection(dataset, count: int, seed: int) -> tuple[list[int], dict[str, Any]]:
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    indices = indices[:count]
    return indices, {
        "strategy": "seeded_random",
        "seed": seed,
        "available_samples": len(dataset),
        "selected_samples": len(indices),
        "samples": [
            {"selection_order": order, "dataset_index": dataset_index}
            for order, dataset_index in enumerate(indices)
        ],
    }


def _select_calibration_indices(
    dataset, count: int, seed: int
) -> tuple[list[int], dict[str, Any]]:
    """Select transitions uniformly across task, episode, and trajectory stage."""

    count = min(count, len(dataset))
    hf_dataset = getattr(dataset, "hf_dataset", None)
    column_names = set(getattr(hf_dataset, "column_names", []))
    required_columns = {"task_index", "episode_index", "frame_index"}
    if hf_dataset is None or not required_columns.issubset(column_names):
        return _random_calibration_selection(dataset, count, seed)

    task_values = hf_dataset["task_index"]
    episode_values = hf_dataset["episode_index"]
    frame_values = hf_dataset["frame_index"]
    if not all(len(values) == len(dataset) for values in (task_values, episode_values, frame_values)):
        return _random_calibration_selection(dataset, count, seed)

    source_values = hf_dataset["index"] if "index" in column_names else range(len(dataset))
    rows_by_episode: dict[int, dict[int, list[tuple[int, int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for dataset_index, (task_id, episode_id, frame_index, source_index) in enumerate(
        zip(task_values, episode_values, frame_values, source_values, strict=True)
    ):
        rows_by_episode[int(task_id)][int(episode_id)].append(
            (int(frame_index), dataset_index, int(source_index))
        )

    rng = random.Random(seed)
    pools: dict[int, dict[int, dict[str, list[tuple[int, int, int, int, int]]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    for task_id in sorted(rows_by_episode):
        for episode_id in sorted(rows_by_episode[task_id]):
            rows = sorted(rows_by_episode[task_id][episode_id])
            episode_length = len(rows)
            for position, (frame_index, dataset_index, source_index) in enumerate(rows):
                stage_index = min(
                    len(_CALIBRATION_STAGES) - 1,
                    position * len(_CALIBRATION_STAGES) // episode_length,
                )
                stage = _CALIBRATION_STAGES[stage_index]
                pools[task_id][episode_id][stage].append(
                    (dataset_index, source_index, frame_index, position, episode_length)
                )

            for stage in _CALIBRATION_STAGES:
                rng.shuffle(pools[task_id][episode_id][stage])

    task_states: dict[int, dict[str, Any]] = {}
    for task_position, task_id in enumerate(sorted(pools)):
        episode_order = sorted(pools[task_id])
        rng.shuffle(episode_order)
        stage_episode_orders = {}
        episode_count = len(episode_order)
        for stage_index, stage in enumerate(_CALIBRATION_STAGES):
            shift = stage_index * max(1, episode_count // len(_CALIBRATION_STAGES))
            shift %= episode_count
            stage_episode_orders[stage] = episode_order[shift:] + episode_order[:shift]
        task_states[task_id] = {
            "stage_cursor": task_position % len(_CALIBRATION_STAGES),
            "episode_cursors": {stage: 0 for stage in _CALIBRATION_STAGES},
            "episode_orders": stage_episode_orders,
        }

    def take_next(task_id: int) -> dict[str, Any] | None:
        state = task_states[task_id]
        for _ in _CALIBRATION_STAGES:
            stage = _CALIBRATION_STAGES[state["stage_cursor"] % len(_CALIBRATION_STAGES)]
            state["stage_cursor"] += 1
            episode_order = state["episode_orders"][stage]
            start = state["episode_cursors"][stage]
            for offset in range(len(episode_order)):
                order_index = (start + offset) % len(episode_order)
                episode_id = episode_order[order_index]
                candidates = pools[task_id][episode_id][stage]
                if candidates:
                    state["episode_cursors"][stage] = (order_index + 1) % len(episode_order)
                    dataset_index, source_index, frame_index, position, episode_length = (
                        candidates.pop()
                    )
                    return {
                        "dataset_index": dataset_index,
                        "source_index": source_index,
                        "task_index": task_id,
                        "episode_index": episode_id,
                        "frame_index": frame_index,
                        "episode_position": position,
                        "episode_length": episode_length,
                        "stage": stage,
                    }
        return None

    selected_records: list[dict[str, Any]] = []
    task_order = sorted(task_states)
    while len(selected_records) < count:
        made_progress = False
        for task_id in task_order:
            record = take_next(task_id)
            if record is None:
                continue
            record["selection_order"] = len(selected_records)
            selected_records.append(record)
            made_progress = True
            if len(selected_records) == count:
                break
        if not made_progress:
            break

    stage_counts = Counter(record["stage"] for record in selected_records)
    task_summaries: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"samples": 0, "episodes": set(), "stages": Counter()}
    )
    for record in selected_records:
        task_summary = task_summaries[record["task_index"]]
        task_summary["samples"] += 1
        task_summary["episodes"].add(record["episode_index"])
        task_summary["stages"][record["stage"]] += 1

    selection_report = {
        "strategy": "task_episode_stage_stratified",
        "seed": seed,
        "available_samples": len(dataset),
        "available_tasks": len(rows_by_episode),
        "available_episodes": sum(len(episodes) for episodes in rows_by_episode.values()),
        "selected_samples": len(selected_records),
        "selected_tasks": len(task_summaries),
        "selected_episodes": len(
            {(record["task_index"], record["episode_index"]) for record in selected_records}
        ),
        "stage_counts": {
            stage: stage_counts.get(stage, 0) for stage in _CALIBRATION_STAGES
        },
        "task_counts": {
            str(task_id): {
                "samples": summary["samples"],
                "episodes": len(summary["episodes"]),
                "stages": {
                    stage: summary["stages"].get(stage, 0)
                    for stage in _CALIBRATION_STAGES
                },
            }
            for task_id, summary in sorted(task_summaries.items())
        },
        "samples": selected_records,
    }
    return [record["dataset_index"] for record in selected_records], selection_report


class _CalibrationData:
    """Re-iterable preprocessed batches with deterministic flow noise and time."""

    def __init__(self, policy, dataloader, preprocessor, camera_keys: list[str], seed: int):
        self.policy = policy
        self.dataloader = dataloader
        self.preprocessor = preprocessor
        self.camera_keys = camera_keys
        self.seed = seed

    def __iter__(self) -> Iterator[_CalibrationBatch]:
        for batch_index, batch in enumerate(self.dataloader):
            for camera_key in self.camera_keys:
                if camera_key in batch and batch[camera_key].dtype == torch.uint8:
                    batch[camera_key] = batch[camera_key].to(dtype=torch.float32) / 255.0
            batch = self.preprocessor(batch)

            prepared_actions = self.policy.prepare_action(batch)
            cuda_device = prepared_actions.device
            cuda_index = cuda_device.index
            if cuda_index is None:
                cuda_index = torch.cuda.current_device()
            with torch.random.fork_rng(devices=[cuda_index]):
                torch.manual_seed(self.seed + batch_index)
                noise = self.policy.model.sample_noise(prepared_actions.shape, cuda_device)
                time = self.policy.model.sample_time(prepared_actions.shape[0], cuda_device)
            yield _CalibrationBatch(batch=batch, noise=noise, time=time)


@contextmanager
def _autoquant_tensor_backward_hooks():
    """Attach ModelOpt score callbacks without creating full-backward-hook output views."""

    original_register = torch.nn.Module.register_full_backward_hook

    def register_output_hook(module, hook, prepend=False):
        def forward_hook(hooked_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                tensor.register_hook(lambda gradient: hook(hooked_module, (), (gradient,)))

        return module.register_forward_hook(forward_hook, prepend=prepend)

    torch.nn.Module.register_full_backward_hook = register_output_hook
    try:
        yield
    finally:
        torch.nn.Module.register_full_backward_hook = original_register


def _forward_loss(model, item: _CalibrationBatch) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(item.batch, noise=item.noise, time=item.time)
    loss = output[0] if isinstance(output, tuple) else output
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise TypeError("SmolVLA PTQ forward must return a scalar loss tensor.")
    return loss


def _make_forward_loop(calibration_data: _CalibrationData, max_batches: int):
    def forward_loop(model) -> None:
        with torch.no_grad():
            for batch_index, item in enumerate(calibration_data):
                if batch_index >= max_batches:
                    break
                _forward_loss(model, item)

    return forward_loop


def _load_bf16_policy(cfg: SmolVLAModelOptPTQConfig, dataset):
    policy_config = SmolVLAConfig.from_pretrained(cfg.policy_path)
    enabled_source_modes = [
        name
        for name in ("use_fp8", "use_nvfp4", "fp8_primary_weights")
        if bool(getattr(policy_config, name, False))
    ]
    if enabled_source_modes:
        raise ValueError(
            "ModelOpt PTQ requires an original BF16 SmolVLA checkpoint. The source enables "
            f"{enabled_source_modes}; use its corresponding BF16/master checkpoint instead."
        )

    # These assignments guarantee model construction cannot enter either Transformer Engine path.
    policy_config.use_fp8 = False
    policy_config.use_nvfp4 = False
    policy_config.fp8_primary_weights = False
    policy_config.compile_model = False
    policy_config.use_amp = False
    policy_config.device = cfg.device
    policy_config.pretrained_path = Path(cfg.policy_path)
    policy_config.push_to_hub = False

    policy = make_policy(
        cfg=policy_config,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )
    if getattr(policy.model, "_fp8", None) is not None or getattr(policy.model, "_nvfp4", None) is not None:
        raise RuntimeError("BF16 PTQ policy unexpectedly initialized a Transformer Engine runtime.")
    policy.to(device=torch.device(cfg.device), dtype=torch.bfloat16)
    policy.eval()
    return policy


def _make_preprocessor(cfg: SmolVLAModelOptPTQConfig, policy, dataset):
    preprocessor_overrides = {
        "device_processor": {"device": cfg.device},
        "normalizer_processor": {
            "features": {**policy.config.input_features, **policy.config.output_features},
            "norm_map": policy.config.normalization_mapping,
            "stats": dataset.meta.stats,
        },
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    postprocessor_overrides = {
        "unnormalizer_processor": {
            "features": policy.config.output_features,
            "norm_map": policy.config.normalization_mapping,
            "stats": dataset.meta.stats,
        },
    }
    processor_kwargs = ProcessorConfigKwargs(
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )
    return make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=cfg.policy_path,
        pretrained_revision=policy.config.pretrained_revision,
        **processor_kwargs,
    )


def _write_json(payload: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    temporary_path.replace(path)


@parser.wrap()
def ptq_smolvla(cfg: SmolVLAModelOptPTQConfig) -> Path:
    """Run fixed NVFP4 PTQ or hardware-aware NVFP4/FP8 AutoQuant on BF16 SmolVLA."""

    init_logging()
    set_seed(cfg.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SmolVLA NVFP4 PTQ.")
    if cfg.enable_rht and importlib.util.find_spec("fast_hadamard_transform") is None:
        raise ModuleNotFoundError(
            "ModelOpt RHT requires fast_hadamard_transform. Install it with "
            "`uv pip install --python /opt/lerobot/bin/python --no-build-isolation "
            "git+https://github.com/Dao-AILab/fast-hadamard-transform.git`."
        )
    if cfg.output_dir.exists():
        raise FileExistsError(
            f"Output directory {cfg.output_dir} already exists; choose a new directory to avoid overwriting."
        )
    cfg.output_dir.mkdir(parents=True)

    try:
        import modelopt
        import modelopt.torch.opt as mto
        import modelopt.torch.quantization as mtq
    except ImportError as exc:
        raise ImportError("SmolVLA PTQ requires NVIDIA ModelOpt with torch quantization support.") from exc

    if cfg.precompile_extensions:
        import modelopt.torch.quantization.extensions as extensions

        extensions.precompile()

    pipeline_cfg = TrainPipelineConfig(
        dataset=cfg.dataset,
        policy=SmolVLAConfig.from_pretrained(cfg.policy_path),
        output_dir=cfg.output_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        rename_map=cfg.rename_map,
    )
    dataset, _ = make_train_eval_datasets(pipeline_cfg)
    policy = _load_bf16_policy(cfg, dataset)
    preprocessor, postprocessor = _make_preprocessor(cfg, policy, dataset)

    decisions = analyze_linear_eligibility(
        policy,
        feature_alignment=cfg.feature_alignment,
        trainable_only=cfg.trainable_only,
        keep_vision_high_precision=cfg.keep_vision_high_precision,
        keep_output_heads_high_precision=cfg.keep_output_heads_high_precision,
        excluded_module_patterns=cfg.excluded_module_patterns,
    )
    eligible_count = sum(decision.eligible for decision in decisions)
    if eligible_count == 0:
        raise RuntimeError("No SmolVLA Linear modules are eligible for PTQ.")
    if cfg.enable_rht and cfg.rht_block_size is not None:
        incompatible = [
            decision
            for decision in decisions
            if decision.eligible and decision.in_features % cfg.rht_block_size
        ]
        if incompatible:
            examples = ", ".join(
                f"{decision.name} (K={decision.in_features})" for decision in incompatible[:5]
            )
            raise ValueError(
                f"rht_block_size={cfg.rht_block_size} must divide every eligible Linear K; "
                f"incompatible layers include: {examples}."
            )

    cost_excluded_names = validate_exclusions(decisions, cfg.cost_excluded_layers)
    budget_decisions = [d for d in decisions if d.name not in cost_excluded_names]
    if not budget_decisions:
        raise ValueError("cost_excluded_layers must leave some Linear weights in the budget.")
    total_linear_weights = sum(
        decision.in_features * decision.out_features for decision in budget_decisions
    )
    eligible_linear_weights = sum(
        decision.in_features * decision.out_features
        for decision in budget_decisions
        if decision.eligible
    )
    minimum_effective_bits = (
        eligible_linear_weights * 4 + (total_linear_weights - eligible_linear_weights) * 16
    ) / total_linear_weights
    if cfg.quantization_mode == "auto" and cfg.effective_bits < minimum_effective_bits:
        raise ValueError(
            f"effective_bits={cfg.effective_bits:.3f} is unreachable with the selected GB10/"
            f"high-precision constraints; minimum is {minimum_effective_bits:.3f}. "
            "Increase effective_bits or explicitly relax the exclusion settings."
        )

    max_batches = max(
        cfg.calibration_batches,
        cfg.score_batches if cfg.quantization_mode == "auto" else 0,
    )
    selected_indices, selection_report = _select_calibration_indices(
        dataset,
        count=max_batches * cfg.batch_size,
        seed=cfg.seed,
    )
    calibration_sample_count = min(
        len(selected_indices), cfg.calibration_batches * cfg.batch_size
    )
    score_sample_count = (
        min(len(selected_indices), cfg.score_batches * cfg.batch_size)
        if cfg.quantization_mode == "auto"
        else 0
    )
    for record in selection_report["samples"]:
        selection_order = record["selection_order"]
        record["used_for_calibration"] = selection_order < calibration_sample_count
        record["used_for_scoring"] = selection_order < score_sample_count

    subset = torch.utils.data.Subset(dataset, selected_indices)
    dataloader = torch.utils.data.DataLoader(
        subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=lerobot_collate_fn if dataset.meta.has_language_columns else None,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        multiprocessing_context="spawn" if cfg.num_workers > 0 else None,
    )
    calibration_data = _CalibrationData(
        policy,
        dataloader,
        preprocessor,
        list(dataset.meta.camera_keys),
        cfg.seed,
    )

    logging.info(
        "Calibration selection: strategy=%s, samples=%d, tasks=%s, episodes=%s, stages=%s.",
        selection_report["strategy"],
        len(selected_indices),
        selection_report.get("selected_tasks", "unavailable"),
        selection_report.get("selected_episodes", "unavailable"),
        selection_report.get("stage_counts", "unavailable"),
    )
    logging.info(
        "Running SmolVLA ModelOpt PTQ: mode=%s, BF16 source=%s, eligible=%d/%d, "
        "calibration_batches=%d, score_batches=%d.",
        cfg.quantization_mode,
        cfg.policy_path,
        eligible_count,
        len(decisions),
        cfg.calibration_batches,
        cfg.score_batches if cfg.quantization_mode == "auto" else 0,
    )

    if cfg.fp8_scale_sweep:
        logging.info(
            "NVFP4 calibration: MSE + FP8 scale sweep, static weight blocks, dynamic input blocks. "
            "AutoQuant FP8 candidate retains its default calibration."
        )

    search_state = None
    if cfg.quantization_mode == "nvfp4":
        quantization_config = build_fixed_nvfp4_config(
            mtq,
            decisions,
            calibration_algorithm=cfg.calibration_algorithm,
            fp8_scale_sweep=cfg.fp8_scale_sweep,
            enable_rht=cfg.enable_rht,
            rht_block_size=cfg.rht_block_size,
            rht_rotate_fp32=cfg.rht_rotate_fp32,
        )
        policy = mtq.quantize(
            policy,
            quantization_config,
            forward_loop=_make_forward_loop(calibration_data, cfg.calibration_batches),
        )
    else:
        with _autoquant_tensor_backward_hooks():
            policy, search_state = auto_quantize_with_budget(
                mtq,
                policy,
                cost_excluded_layers=cfg.cost_excluded_layers,
                constraints={"effective_bits": cfg.effective_bits},
                quantization_formats=build_autoquant_formats(
                    mtq,
                    calibration_algorithm=cfg.calibration_algorithm,
                    fp8_scale_sweep=cfg.fp8_scale_sweep,
                    enable_rht=cfg.enable_rht,
                    rht_block_size=cfg.rht_block_size,
                    rht_rotate_fp32=cfg.rht_rotate_fp32,
                ),
                data_loader=calibration_data,
                forward_step=_forward_loss,
                loss_func=lambda output, _batch: output,
                # Keep every budget-excluded module fixed at high precision, including vision Conv.
                disabled_layers=disabled_layer_names(decisions) + cfg.cost_excluded_layers,
                num_calib_steps=cfg.calibration_batches,
                num_score_steps=cfg.score_batches,
                verbose=True,
                method="gradient",
                checkpoint=str(cfg.output_dir / MODELOPT_AUTOQUANT_CHECKPOINT_NAME),
            )
            torch.save(search_state, cfg.output_dir / MODELOPT_AUTOQUANT_STATE_NAME)

    policy.zero_grad(set_to_none=True)
    policy.eval()
    mtq.print_quant_summary(policy, output_dir=str(cfg.output_dir))
    quantized_report = build_quantized_linear_report(policy)

    # This is intentionally mto.save(), not mtq.compress(): BF16 master weights remain trainable.
    mto.save(policy, cfg.output_dir / MODELOPT_PTQ_CHECKPOINT_NAME)
    torch.save(mto.modelopt_state(policy), cfg.output_dir / MODELOPT_STATE_NAME)

    policy.config.pretrained_path = None
    policy.config.push_to_hub = False
    policy.config._save_pretrained(cfg.output_dir)
    preprocessor.save_pretrained(cfg.output_dir)
    postprocessor.save_pretrained(cfg.output_dir)

    manifest = {
        "format": MODELOPT_PTQ_FORMAT,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_policy": cfg.policy_path,
        "source_precision": "bf16",
        "fake_quant": True,
        "compressed": False,
        "qad_ready": True,
        "transformer_engine_used": False,
        "quantization_mode": cfg.quantization_mode,
        "calibration_algorithm": cfg.calibration_algorithm,
        "rht": {
            "enabled": cfg.enable_rht,
            "block_size": cfg.rht_block_size,
            "rotate_fp32": cfg.rht_rotate_fp32,
            "applied_to": ["input_quantizer", "weight_quantizer"],
        },
        "effective_bits_target": cfg.effective_bits if cfg.quantization_mode == "auto" else None,
        "minimum_effective_bits": minimum_effective_bits,
        "feature_alignment": cfg.feature_alignment,
        "keep_vision_high_precision": cfg.keep_vision_high_precision,
        "keep_output_heads_high_precision": cfg.keep_output_heads_high_precision,
        "trainable_only": cfg.trainable_only,
        "calibration": {
            "dataset_repo_id": cfg.dataset.repo_id,
            "dataset_revision": cfg.dataset.revision,
            "selected_samples": len(selected_indices),
            "calibration_samples": calibration_sample_count,
            "score_samples": score_sample_count,
            "selection": selection_report,
            "batch_size": cfg.batch_size,
            "calibration_batches": cfg.calibration_batches,
            "score_batches": cfg.score_batches if cfg.quantization_mode == "auto" else 0,
            "seed": cfg.seed,
        },
        "software": {
            "modelopt": modelopt.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(torch.device(cfg.device)),
            "compute_capability": list(torch.cuda.get_device_capability(torch.device(cfg.device))),
        },
        "files": {
            "checkpoint": MODELOPT_PTQ_CHECKPOINT_NAME,
            "modelopt_state": MODELOPT_STATE_NAME,
            "autoquant_state": MODELOPT_AUTOQUANT_STATE_NAME if search_state is not None else None,
        },
        "eligibility": serialize_decisions(decisions),
        "quantized_linears": quantized_report,
    }
    if cfg.fp8_scale_sweep:
        manifest["fp8_scale_sweep"] = True
    if cfg.cost_excluded_layers:
        manifest["autoquant_budget"] = search_state["smolvla_budget"]
    _write_json(manifest, cfg.output_dir / MODELOPT_PTQ_MANIFEST_NAME)
    logging.info(
        "Saved QAD-ready fake-quant checkpoint to %s with precision counts %s.",
        cfg.output_dir,
        quantized_report["precision_counts"],
    )
    return cfg.output_dir


def main() -> None:
    register_third_party_plugins()
    ptq_smolvla()


if __name__ == "__main__":
    main()
