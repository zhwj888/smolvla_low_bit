# SmolVLA Low Bit

A SmolVLA low precision training and quantization research repository built on [Hugging Face LeRobot](https://github.com/huggingface/lerobot), with support for **FP8, NVFP4, post-training quantization (PTQ), mixed precision search (AutoQuant), and quantization-aware distillation (QAD)**.

This repository retains LeRobot's dataset, training, evaluation, and robot interfaces. Most additions live in `src/lerobot/policies/smolvla/` and `src/lerobot/scripts/`. This guide describes the current implementation. Replace the example model, dataset, and output paths with those for your task.

## Features and implementation status

| Approach | Supported features | Checkpoints and loading |
| --- | --- | --- |
| TE FP8 | Trainable Linear replacement; optional Norm-to-Linear fusion; training and inference | Training retains high precision master weights; export to `model_fp8.pt` for loading with this repository's `lerobot-eval` |
| TE NVFP4 | `NVFP4BlockScaling`; Norm-to-Linear fusion; shape alignment and high precision fallback | NVFP4 execution from standard training checkpoints; `model_nvfp4.pt` export is available but not integrated into the standard loader |
| ModelOpt PTQ | Fixed NVFP4 assignment; calibration; optional RHT rotation | `modelopt_ptq.pt` retains high precision master weights and fake quantization state |
| ModelOpt AutoQuant | Search over NVFP4 / FP8 / no quantization for eligible Linear layers; bit budget constraints | PTQ artifacts plus search state and per-layer quantization reports |
| ModelOpt QAD | Preserve PTQ assignments; distill action velocity predictions from a high precision teacher | Single GPU training, resume, dedicated evaluation, and safetensors + JSON export |

**Low precision computation, quantized weight storage, and fake quantization are separate capabilities.** TE training retains master parameters, while frozen modules and some operations remain in high precision. ModelOpt PTQ/QAD currently uses fake quantization without weight compression or deployment engine generation. Lower bit widths alone do not establish proportional reductions in checkpoint size, memory use, or latency.

The repository also includes changes for VLA-JEPA / Qwen3-VL; see the [VLA-JEPA documentation](docs/source/policy_vla_jepa_README.md). The low bit commands below target SmolVLA.

## Environment and dependencies

Python **3.12+** is required. Transformer Engine (TE) execution targets Linux and NVIDIA CUDA GPUs, with hardware and runtime support for the selected recipe. The project constrains TE to **2.16.x**. The ModelOpt requirements pin **0.45.0**, matching the current AutoQuant budget adapter.

| Requirements file | Purpose |
| --- | --- |
| [requirements.txt](requirements.txt) | Base SmolVLA with an editable installation of this repository |
| [requirements-fp8.txt](requirements-fp8.txt) | SmolVLA + TE for both FP8 and NVFP4 |
| [requirements-modelopt.txt](requirements-modelopt.txt) | Base SmolVLA + ModelOpt for PTQ / AutoQuant / QAD |
| [requirements-rht.txt](requirements-rht.txt) | CUDA extension for ModelOpt's default RHT configuration, pinned to a source revision |

Base dependencies and TE version constraints are defined in [pyproject.toml](pyproject.toml); the requirements files reuse project extras. These files are not complete platform lockfiles. [uv.lock](uv.lock) locks the existing project dependencies; the separately installed ModelOpt and RHT packages are outside that lockfile.

### Installation

Use an existing Python 3.12+ environment, such as your current Conda environment, a virtual environment, or another Python installation. Creating a new virtual environment is optional. Activate or select the environment you want to use before installing:

```bash
git clone https://github.com/zhwj888/smolvla_low_bit.git
cd smolvla_low_bit
python --version
python -m pip --version
```

Install the dependencies for your chosen approach. Install both groups in sequence if you need all features:

```bash
# TE FP8 / NVFP4
python -m pip install -r requirements-fp8.txt

# ModelOpt PTQ / AutoQuant / QAD
python -m pip install -r requirements-modelopt.txt

# ModelOpt's default RHT: install the dependencies above before building the extension
python -m pip install setuptools wheel ninja
python -m pip install --no-build-isolation -r requirements-rht.txt
```

For standard SmolVLA only, install `requirements.txt`. Building RHT requires a working CUDA Toolkit / `nvcc`, a C++ toolchain, and a compatible PyTorch installation. TE installation also depends on available wheels and CUDA libraries for your platform. The requirements files do not install GPU drivers or system toolchains.

To run PTQ without RHT, skip the extension and explicitly pass `--enable_rht=false`. This changes the quantization configuration. Loading a PTQ/QAD model that uses RHT still requires the extension.

The examples use `python -m pip` to install into the environment selected by `python`. Run the training and evaluation commands from that same environment. If you prefer `uv`, use `uv pip install --python /path/to/your/python -r requirements-fp8.txt` (or another requirements file) to select the interpreter explicitly.

### Check GPU and TE support

```bash
python - <<'PY'
import torch
import transformer_engine.pytorch as te

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name())
    print("Compute capability:", torch.cuda.get_device_capability())
    print("FP8:", te.is_fp8_available(return_reason=True))
    print("NVFP4:", te.is_nvfp4_available(return_reason=True))
PY
```

FP8 support varies by recipe; the runtime performs an additional check for the selected recipe. For TE 2.16 on GB10 / SM12x, the NVFP4 implementation disables stochastic rounding by default, requires Linear feature dimensions aligned to 32, and retains ineligible modules in high precision. NVFP4 therefore covers only part of the model.

## FP8 training and export

Prepare a SmolVLA checkpoint and a dataset in LeRobot format that match your task. The local checkpoint directory below is a placeholder and should contain `config.json`, `model.safetensors`, and the corresponding preprocessing assets.

```bash
lerobot-train \
  --policy.path=/path/to/smolvla/pretrained_model \
  --dataset.repo_id=your-account/your-dataset \
  --output_dir=outputs/train/smolvla_fp8 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --accelerator.mixed_precision=bf16 \
  --policy.use_fp8=true \
  --policy.use_nvfp4=false \
  --policy.fp8_recipe=delayed \
  --policy.fp8_format=hybrid \
  --policy.fp8_fuse_layernorm=true \
  --batch_size=8 \
  --steps=10000 \
  --save_freq=1000
```

The `10000` training steps are illustrative and do not guarantee convergence. Camera keys, state/action dimensions, and normalization settings must match the policy.

| Parameter | Default | Description |
| --- | --- | --- |
| `policy.use_fp8` | `false` | Enable TE FP8; mutually exclusive with `use_nvfp4` |
| `policy.fp8_recipe` | `delayed` | `delayed` / `current` / `blockwise` / `mxfp8` |
| `policy.fp8_format` | `hybrid` | `hybrid` / `e4m3` |
| `policy.fp8_amax_history_len` | `16` | Length of the amax history for delayed scaling |
| `policy.fp8_fuse_layernorm` | `false` | Fuse supported trainable Norm-to-Linear paths |
| `policy.fp8_primary_weights` | `false` | Inference only; normally set by the exporter |

By default, only eligible trainable Linear layers are converted; frozen modules remain in high precision. Training handles batch alignment, while inference pads individual GEMMs and restores the original output shape. Coverage depends on model dimensions and freezing settings.

Export the final training checkpoint with FP8 primary weights:

```bash
lerobot-export-smolvla-fp8 \
  --checkpoint_dir=outputs/train/smolvla_fp8/checkpoints/last/pretrained_model \
  --output_dir=outputs/export/smolvla_fp8 \
  --device=cuda
```

The output includes `model_fp8.pt`, `fp8_manifest.json`, `config.json`, and copied preprocessing assets. The manifest lists quantized parameters and remaining high precision Linear layers. The source checkpoint is preserved. Standard high precision checkpoints can also be exported, but their quantized accuracy must be evaluated.

FP8 primary weight checkpoints are inference artifacts and cannot directly resume training. They currently prohibit `torch.compile`. Loading also checks compatibility of the PyTorch and TE major/minor versions.

## NVFP4 training and export

NVFP4 uses its own `NVFP4BlockScaling` recipe and enables supported Norm-to-Linear fusion:

```bash
lerobot-train \
  --policy.path=/path/to/smolvla/pretrained_model \
  --dataset.repo_id=your-account/your-dataset \
  --output_dir=outputs/train/smolvla_nvfp4 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --accelerator.mixed_precision=bf16 \
  --policy.use_fp8=false \
  --policy.use_nvfp4=true \
  --policy.nvfp4_disable_stochastic_rounding=true \
  --batch_size=32 \
  --steps=10000 \
  --save_freq=1000
```

The defaults are `nvfp4_disable_rht=false` and `nvfp4_disable_2d_quantization=false`. Training retains high precision master weights. Linear layers that do not meet runtime alignment requirements are skipped.

```bash
python -m lerobot.scripts.lerobot_export_smolvla_nvfp4 \
  --checkpoint_dir=outputs/train/smolvla_nvfp4/checkpoints/last/pretrained_model \
  --output_dir=outputs/export/smolvla_nvfp4 \
  --device=cuda
```

The exporter produces `model_nvfp4.pt` and `nvfp4_manifest.json`, listing actual NVFP4 parameters and high precision fallback modules. **`SmolVLAPolicy` currently has no loading path for `model_nvfp4.pt`, so this export directory cannot be passed directly to `lerobot-eval`.** For NVFP4 evaluation, use a standard training checkpoint with `use_nvfp4=true` and its `model.safetensors` file.

## ModelOpt PTQ and AutoQuant

Start from a standard high precision SmolVLA checkpoint with `use_fp8=false`, `use_nvfp4=false`, and `fp8_primary_weights=false`. TE exports with quantized primary weights are not accepted. The PTQ output can initialize a QAD student directly.

### Fixed NVFP4 quantization

```bash
python -m lerobot.scripts.lerobot_ptq_smolvla \
  --policy_path=/path/to/bf16/pretrained_model \
  --dataset.repo_id=your-account/your-dataset \
  --output_dir=outputs/ptq/smolvla_nvfp4 \
  --quantization_mode=nvfp4 \
  --calibration_algorithm=max \
  --calibration_batches=32 \
  --batch_size=8 \
  --device=cuda
```

### NVFP4 / FP8 mixed precision search

```bash
python -m lerobot.scripts.lerobot_ptq_smolvla \
  --policy_path=/path/to/bf16/pretrained_model \
  --dataset.repo_id=your-account/your-dataset \
  --output_dir=outputs/ptq/smolvla_auto \
  --quantization_mode=auto \
  --effective_bits=11.25 \
  --calibration_batches=32 \
  --score_batches=16 \
  --batch_size=8 \
  --device=cuda
```

The defaults are `feature_alignment=32` and `trainable_only=true`, with vision modules and output heads kept in high precision. `effective_bits` constrains search cost; it is not the actual average bit width of the checkpoint file, and a requested budget may be infeasible. `cost_excluded_layers` can remove modules already fixed at high precision from budget accounting. It cannot directly exclude modules that are still eligible for quantization search.

Calibration supports `max` and `mse`. Setting `fp8_scale_sweep=true` requires `calibration_algorithm=mse`. RHT is enabled by default and is controlled by `enable_rht`, `rht_block_size`, and `rht_rotate_fp32`.

Main artifacts include `modelopt_ptq.pt`, `modelopt_state.pth`, `modelopt_ptq_manifest.json`, and model/preprocessing configurations. AutoQuant additionally saves search state. The manifest records module eligibility, per-layer quantization results, and precision statistics.

## QAD: quantization-aware distillation

QAD restores PTQ quantizers and calibration state without repeating the bit assignment search. The student uses FP32 master parameters, BF16 autocast, and fake quantization. A frozen high precision teacher predicts action velocities using the same inputs, noise, and timesteps. The distillation loss excludes padded action dimensions and padded frames at episode boundaries.

```bash
python -m lerobot.scripts.lerobot_qad_smolvla \
  --ptq_path=outputs/ptq/smolvla_nvfp4 \
  --teacher_path=/path/to/bf16/pretrained_model \
  --dataset.repo_id=your-account/your-dataset \
  --output_dir=outputs/qad/smolvla_nvfp4 \
  --sample_fraction=0.1 \
  --balanced_sampling=true \
  --steps=1000 \
  --batch_size=4 \
  --learning_rate=1e-5 \
  --device=cuda
```

- Currently supports one process on one GPU. Streaming datasets and random image augmentation are unsupported.
- `steps=0` (the default) means one pass over the selected subset. The example explicitly runs 1000 optimizer steps.
- The defaults are `distillation_weight=1.0` and `task_weight=0.0`. Increase the task loss weight to include the task objective.
- Checkpoints are stored directly under `outputs/qad/smolvla_nvfp4/checkpoints/001000/`, without an additional `pretrained_model/` directory.
- To resume, add `--resume_from=/path/to/checkpoints/001000`, choose a new `output_dir`, and set `steps` to the desired total step count. Keep teacher, dataset, and sampling settings consistent.

Export to safetensors + JSON for distribution:

```bash
python -m lerobot.scripts.lerobot_export_smolvla_qad \
  --checkpoint=outputs/qad/smolvla_nvfp4/checkpoints/001000 \
  --output_dir=outputs/export/smolvla_qad
```

The output includes `model.safetensors`, `modelopt_state.json`, `modelopt_qad_safetensors_manifest.json`, and preprocessing assets. Export preserves tensor dtypes and fake quantization state; it does not produce a compressed NVFP4 deployment model. The exporter also accepts PTQ `.pt` checkpoints.

## Evaluation

The examples below use LIBERO. First install its dependencies and prepare a policy and observation mapping that match the task:

```bash
python -m pip install -e '.[libero]'
```

Use the standard evaluation entry point for **TE FP8 exports**. For TE NVFP4, replace `policy.path` with a standard NVFP4 training checkpoint directory:

```bash
lerobot-eval \
  --policy.path=outputs/export/smolvla_fp8 \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10 \
  --eval.batch_size=1 \
  --output_dir=outputs/eval/smolvla_fp8
```

Use the dedicated entry point for **ModelOpt PTQ, QAD, and QAD safetensors exports**:

```bash
python -m lerobot.scripts.lerobot_eval_smolvla_qad \
  --policy.path=outputs/export/smolvla_qad \
  --policy.device=cuda \
  --policy.use_amp=true \
  --env.type=libero \
  --env.task=libero_object \
  --env.max_parallel_tasks=1 \
  --eval.n_episodes=10 \
  --eval.batch_size=1 \
  --output_dir=outputs/eval/smolvla_qad
```

QAD evaluation requires CUDA, BF16 autocast, and `max_parallel_tasks=1`. It writes `eval_info.json` and renders videos for up to 10 episodes by default. Use new output directories for PTQ/QAD training, export, and dedicated evaluation.

This README does not claim speedups or success rates without a controlled evaluation. When comparing approaches, keep tasks, source checkpoints, seeds, action steps, batch sizes, and runtime environments consistent. Measure success rate, latency, memory use, and artifact size separately.

## Code and tests

| Path | Contents |
| --- | --- |
| [configuration_smolvla.py](src/lerobot/policies/smolvla/configuration_smolvla.py) | FP8 / NVFP4 options and configuration validation |
| [fp8.py](src/lerobot/policies/smolvla/fp8.py) | TE conversion, fusion, and FP8 primary weight saving/loading |
| [nvfp4.py](src/lerobot/policies/smolvla/nvfp4.py) | NVFP4 recipe, module selection, and shape alignment |
| [modelopt_ptq.py](src/lerobot/policies/smolvla/modelopt_ptq.py) | PTQ module selection, quantization configuration, and reporting |
| [autoquant_budget.py](src/lerobot/policies/smolvla/autoquant_budget.py) | Exclude fixed high precision modules from AutoQuant budgets |
| [modelopt_qad.py](src/lerobot/policies/smolvla/modelopt_qad.py) | Student/teacher loading and action velocity distillation |
| [qad_sampling.py](src/lerobot/policies/smolvla/qad_sampling.py) | Subset selection and balanced sampling |
| [tests/policies/smolvla](tests/policies/smolvla) | Tests for the features above |

```bash
python -m pip install -e '.[test]'
pytest tests/policies/smolvla -q
```

The tests cover configuration, module conversion, export, sampling, and distillation logic. Passing them does not establish successful training on real hardware or task success rates.

## Acknowledgments and license

This project builds on **LeRobot** and **SmolVLA**, using NVIDIA **Transformer Engine** and **ModelOpt** for low precision execution and quantization, and **fast-hadamard-transform** for the ModelOpt RHT extension. Upstream commit history and copyright notices are retained.

This repository uses the [Apache-2.0 license](LICENSE). Dependencies, models, and datasets remain subject to their respective licenses.

- [Upstream LeRobot repository](https://github.com/huggingface/lerobot)
- [Original SmolVLA documentation](docs/source/policy_smolvla_README.md)
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
- [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)
- [fast-hadamard-transform](https://github.com/Dao-AILab/fast-hadamard-transform)

If you use LeRobot in research, please cite:

```bibtex
@misc{cadene2024lerobot,
  author = {Cadene, Remi and others},
  title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
  howpublished = {\url{https://github.com/huggingface/lerobot}},
  year = {2024}
}
```
