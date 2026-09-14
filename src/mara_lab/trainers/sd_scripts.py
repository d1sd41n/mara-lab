from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from mara_lab.config import ResolvedTrainingExperimentConfig, TrainerProfile
from mara_lab.errors import TrainingError


@dataclass(frozen=True, slots=True)
class PreparedTrainer:
    source_dir: Path
    runtime_python: Path
    source_revision: str
    runtime_lock: Path


def _run_checked(
    command: list[str],
    *,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=1800,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {detail[-3000:]}" if detail else ""
        raise TrainingError(f"command failed: {' '.join(command)}{suffix}") from exc
    return result.stdout.strip()


def _git_revision(source_dir: Path) -> str | None:
    if not (source_dir / ".git").exists():
        return None
    try:
        return _run_checked(["git", "-C", str(source_dir), "rev-parse", "HEAD"])
    except TrainingError:
        return None


def _verify_sd_scripts_checkout(source_dir: Path, profile: TrainerProfile) -> str:
    revision = _git_revision(source_dir)
    if revision is None:
        raise TrainingError(f"trainer source is not a Git checkout: {source_dir}")
    if revision != profile.revision:
        raise TrainingError(
            f"sd-scripts at {source_dir} is revision {revision}; expected {profile.revision}"
        )
    origin = _run_checked(["git", "-C", str(source_dir), "remote", "get-url", "origin"])
    if origin.rstrip("/") != profile.repository.rstrip("/"):
        raise TrainingError(f"sd-scripts origin is {origin}; expected {profile.repository}")
    dirty = _run_checked(
        ["git", "-C", str(source_dir), "status", "--porcelain", "--untracked-files=all"]
    )
    if dirty:
        raise TrainingError(f"sd-scripts checkout contains local modifications: {source_dir}")
    return revision


def ensure_sd_scripts_source(profile: TrainerProfile) -> tuple[Path, str]:
    source_dir = profile.source_dir.resolve()
    revision = _git_revision(source_dir)
    if revision is not None:
        return source_dir, _verify_sd_scripts_checkout(source_dir, profile)
    if source_dir.exists():
        raise TrainingError(f"trainer source path exists but is not a Git checkout: {source_dir}")

    source_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = source_dir.with_name(f".{source_dir.name}.cloning-{os.getpid()}")
    if staging.exists():
        raise TrainingError(f"stale trainer clone is present: {staging}")
    try:
        _run_checked(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                profile.repository,
                str(staging),
            ]
        )
        _run_checked(
            ["git", "-C", str(staging), "fetch", "--depth", "1", "origin", profile.revision]
        )
        _run_checked(["git", "-C", str(staging), "checkout", "--detach", profile.revision])
        try:
            os.replace(staging, source_dir)
        except OSError as exc:
            raise TrainingError(f"cannot publish trainer checkout at {source_dir}: {exc}") from exc
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return source_dir, _verify_sd_scripts_checkout(source_dir, profile)


def _find_uv(runtime_project: Path) -> Path:
    configured = os.environ.get("MARA_UV")
    if configured:
        candidate = Path(configured).resolve()
        if candidate.is_file():
            return candidate
        raise TrainingError(f"MARA_UV does not point to a file: {candidate}")
    discovered = shutil.which("uv")
    if discovered:
        return Path(discovered)
    project_root = runtime_project.resolve().parents[1]
    local = project_root / ".tools" / "uv312" / "Scripts" / "uv.exe"
    if local.is_file():
        return local
    raise TrainingError("uv is not available; run scripts/bootstrap.ps1 first")


def ensure_sd_scripts_runtime(profile: TrainerProfile, source_dir: Path) -> tuple[Path, Path]:
    runtime_project = profile.runtime_project.resolve()
    project_file = runtime_project / "pyproject.toml"
    lock_file = runtime_project / "uv.lock"
    if not project_file.is_file() or not lock_file.is_file():
        raise TrainingError(f"trainer runtime is incomplete: {runtime_project}")
    uv = _find_uv(runtime_project)
    _run_checked([str(uv), "sync", "--project", str(runtime_project), "--locked", "--no-dev"])
    runtime_python = runtime_project / ".venv" / "Scripts" / "python.exe"
    if not runtime_python.is_file():
        raise TrainingError(f"trainer runtime did not create {runtime_python}")
    required_options = [
        "--cache_latents",
        "--cache_latents_to_disk",
        "--cache_text_encoder_outputs",
        "--cache_text_encoder_outputs_to_disk",
        "--dataset_config",
        "--full_bf16",
        "--gradient_accumulation_steps",
        "--gradient_checkpointing",
        "--learning_rate",
        "--log_with",
        "--logging_dir",
        "--lr_scheduler",
        "--lr_warmup_steps",
        "--max_data_loader_n_workers",
        "--max_train_steps",
        "--mixed_precision",
        "--network_alpha",
        "--network_dim",
        "--network_module",
        "--network_train_unet_only",
        "--optimizer_type",
        "--output_dir",
        "--output_name",
        "--pretrained_model_name_or_path",
        "--save_every_n_steps",
        "--save_model_as",
        "--save_precision",
        "--sdpa",
        "--seed",
        "--train_batch_size",
        "--unet_lr",
    ]
    probe = (
        "import bitsandbytes,sys,torch; "
        f"sys.path.insert(0,{json.dumps(str(source_dir))}); "
        "import sdxl_train_network as module; "
        "options=module.setup_parser()._option_string_actions; "
        f"missing=[name for name in {required_options!r} if name not in options]; "
        "assert not missing, f'missing trainer options: {missing}'; "
        "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
        "assert torch.cuda.is_bf16_supported(), 'BF16 is unavailable'"
    )
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["HF_HOME"] = str(
        (runtime_project.parents[1] / ".cache" / "huggingface-home").resolve()
    )
    environment["HF_HUB_CACHE"] = str(
        (runtime_project.parents[1] / ".cache" / "huggingface").resolve()
    )
    _run_checked(
        [
            str(runtime_python),
            "-c",
            probe,
        ],
        cwd=source_dir,
        environment=environment,
    )
    return runtime_python, lock_file


def prepare_sd_scripts_trainer(profile: TrainerProfile) -> PreparedTrainer:
    source_dir, revision = ensure_sd_scripts_source(profile)
    runtime_python, lock_file = ensure_sd_scripts_runtime(profile, source_dir)
    return PreparedTrainer(source_dir, runtime_python, revision, lock_file)


def compile_sd_scripts_arguments(
    config: ResolvedTrainingExperimentConfig,
    *,
    checkpoint_path: Path,
    dataset_config_path: Path,
    output_dir: Path,
    logging_dir: Path,
    max_train_steps: int,
) -> list[str]:
    profile = config.trainer
    arguments = [
        "--pretrained_model_name_or_path",
        str(checkpoint_path.resolve()),
        "--dataset_config",
        str(dataset_config_path.resolve()),
        "--output_dir",
        str(output_dir.resolve()),
        "--output_name",
        config.adapter_id,
        "--save_model_as",
        "safetensors",
        "--network_module",
        profile.network_module,
        "--network_dim",
        str(profile.network_dim),
        "--network_alpha",
        str(profile.network_alpha),
        "--network_train_unet_only",
        "--learning_rate",
        str(profile.learning_rate),
        "--unet_lr",
        str(profile.learning_rate),
        "--optimizer_type",
        profile.optimizer_type,
        "--lr_scheduler",
        profile.lr_scheduler,
        "--lr_warmup_steps",
        str(min(profile.lr_warmup_steps, max(0, max_train_steps - 1))),
        "--max_train_steps",
        str(max_train_steps),
        "--train_batch_size",
        str(profile.train_batch_size),
        "--gradient_accumulation_steps",
        str(profile.gradient_accumulation_steps),
        "--mixed_precision",
        profile.mixed_precision,
        "--save_precision",
        profile.mixed_precision,
        "--seed",
        str(profile.seed),
        "--max_data_loader_n_workers",
        str(profile.max_data_loader_workers),
        "--logging_dir",
        str(logging_dir.resolve()),
        "--log_with",
        "tensorboard",
        "--cache_latents",
        "--cache_latents_to_disk",
        "--cache_text_encoder_outputs",
        "--cache_text_encoder_outputs_to_disk",
        "--gradient_checkpointing",
        "--sdpa",
    ]
    if profile.full_bf16:
        arguments.append("--full_bf16")
    save_interval = min(profile.save_every_n_steps, max_train_steps)
    if save_interval < max_train_steps:
        arguments.extend(["--save_every_n_steps", str(save_interval)])
    return arguments


def serialize_command(command: list[str]) -> dict[str, object]:
    return {"argv": command, "display": subprocess.list2cmdline(command)}


def inspect_sd_scripts_adapter(runtime_python: Path, adapter_path: Path) -> dict[str, int | float]:
    code = f"""
import json
import torch
from safetensors import safe_open

path = {json.dumps(str(adapter_path.resolve()))}
tensor_count = 0
up_tensor_count = 0
nonzero_up_tensor_count = 0
max_abs_weight = 0.0
with safe_open(path, framework="pt", device="cpu") as reader:
    for name in reader.keys():
        tensor = reader.get_tensor(name)
        if not bool(torch.isfinite(tensor).all().item()):
            raise RuntimeError(f"non-finite adapter tensor: {{name}}")
        tensor_count += 1
        if tensor.numel():
            max_abs_weight = max(max_abs_weight, float(tensor.abs().max().float().item()))
        if "lora_up" in name:
            up_tensor_count += 1
            nonzero_up_tensor_count += int(bool(torch.count_nonzero(tensor).item()))
if tensor_count == 0:
    raise RuntimeError("adapter has no tensors")
if up_tensor_count == 0:
    raise RuntimeError("adapter has no lora_up tensors")
if nonzero_up_tensor_count == 0:
    raise RuntimeError("all lora_up tensors are zero")
print(json.dumps({{
    "tensor_count": tensor_count,
    "up_tensor_count": up_tensor_count,
    "nonzero_up_tensor_count": nonzero_up_tensor_count,
    "max_abs_weight": max_abs_weight,
}}, sort_keys=True))
"""
    output = _run_checked([str(runtime_python), "-c", code])
    try:
        value = json.loads(output.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise TrainingError("cannot parse adapter tensor statistics") from exc
    if not isinstance(value, dict):
        raise TrainingError("adapter tensor statistics must be an object")
    return value


def runtime_versions(runtime_python: Path) -> dict[str, str]:
    code = (
        "import importlib.metadata as m,json; "
        "names=['accelerate','bitsandbytes','diffusers','safetensors','torch','transformers']; "
        "print(json.dumps({name:m.version(name) for name in names},sort_keys=True))"
    )
    output = _run_checked([str(runtime_python), "-c", code])
    try:
        value = json.loads(output.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise TrainingError("cannot parse trainer runtime versions") from exc
    return value
