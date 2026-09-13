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

"""Evaluate a ModelOpt PTQ/QAD student using the existing environment rollout functions."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modelopt_qad import load_qad_student
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging


@dataclass
class SmolVLAQADEvalConfig(EvalPipelineConfig):
    save_video: bool = True


def load_eval_student(config):
    if not isinstance(config, SmolVLAConfig) or not config.pretrained_path:
        raise ValueError("Specify --policy.path pointing to a ModelOpt SmolVLA PTQ/QAD checkpoint.")
    if not config.device.startswith("cuda") or not config.use_amp:
        raise ValueError("QAD evaluation requires --policy.device=cuda --policy.use_amp=true (BF16).")
    saved = SmolVLAConfig.from_pretrained(config.pretrained_path)
    allowed = {"device", "use_amp", "pretrained_path", "n_action_steps", "num_steps"}
    for key, value in asdict(config).items():
        if key not in allowed and value != asdict(saved)[key]:
            raise ValueError(f"Unsupported QAD evaluation config override: {key}")
    if not 1 <= config.n_action_steps <= saved.chunk_size or config.num_steps <= 0:
        raise ValueError("Require 1 <= n_action_steps <= chunk_size and num_steps > 0.")
    policy = load_qad_student(config.pretrained_path, config.device).requires_grad_(False).eval()
    # policy and its flow model share the same config; reset action queues after overriding.
    policy.config.n_action_steps = config.n_action_steps
    policy.config.num_steps = config.num_steps
    policy.config.use_amp = True
    policy.config.pretrained_path = config.pretrained_path
    policy.reset()
    return policy


def make_eval_processors(policy, path, rename_map):
    overrides = {"device_processor": {"device": str(policy.config.device)}}
    if rename_map:
        overrides["rename_observations_processor"] = {"rename_map": rename_map}
    return make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=path, preprocessor_overrides=overrides
    )


def run_eval(cfg: SmolVLAQADEvalConfig):
    if cfg.env.max_parallel_tasks != 1:
        raise ValueError("QAD evaluation currently requires --env.max_parallel_tasks=1.")
    output = Path(cfg.output_dir)
    if output.exists():
        raise FileExistsError(f"Choose a new evaluation output directory: {output}")
    set_seed(cfg.seed)
    policy = load_eval_student(cfg.policy)
    preprocessor, postprocessor = make_eval_processors(policy, cfg.policy.pretrained_path, cfg.rename_map)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=cfg.env, policy_cfg=policy.config
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "eval_config.json").write_text(json.dumps(asdict(cfg), default=str, indent=2) + "\n")
    logging.info("Evaluating %s with NVFP4 fake quantization and BF16 autocast", cfg.policy.pretrained_path)
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )
    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            info = eval_policy_all(
                envs=envs,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=cfg.eval.n_episodes,
                max_episodes_rendered=10 if cfg.save_video and not cfg.eval.recording else 0,
                videos_dir=output / "videos" if cfg.save_video and not cfg.eval.recording else None,
                return_episode_data=False,
                start_seed=cfg.seed,
                max_parallel_tasks=1,
                recording_dir=output / "recordings" if cfg.eval.recording else None,
                env_features=cfg.env.features if cfg.eval.recording else None,
                recording_repo_id=cfg.eval.recording_repo_id,
                recording_private=cfg.eval.recording_private,
            )
    finally:
        close_envs(envs)
    (output / "eval_info.json").write_text(json.dumps(info, indent=2) + "\n")
    logging.info("Overall Aggregated Metrics: %s", info["overall"])
    return info


@parser.wrap()
def eval_main(cfg: SmolVLAQADEvalConfig):
    run_eval(cfg)


if __name__ == "__main__":
    init_logging()
    register_third_party_plugins()
    eval_main()
