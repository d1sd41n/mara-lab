from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli_w

from mara_lab.artifacts import (
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    atomic_write_yaml,
    default_run_id,
    environment_snapshot,
    read_jsonl,
    sha256_file,
    utc_now,
)
from mara_lab.config import ResolvedTrainingExperimentConfig
from mara_lab.errors import ConfigurationError, MemoryBudgetExceededError, TrainingError
from mara_lab.manifests import AdapterRecord, DatasetRecord, RunStatus
from mara_lab.model_materialization import read_safetensors_layout
from mara_lab.trainers.sd_scripts import (
    PreparedTrainer,
    compile_sd_scripts_arguments,
    inspect_sd_scripts_adapter,
    prepare_sd_scripts_trainer,
    runtime_versions,
    serialize_command,
)


@dataclass(frozen=True, slots=True)
class ValidatedDataset:
    train_dir: Path
    train_count: int
    validation_count: int
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class TrainingRunSummary:
    run_id: str
    run_dir: Path
    adapters: list[Path]
    adapter_manifest_path: Path
    max_reserved_vram_gib: float
    wall_seconds: float


def _resolve_dataset_file(root: Path, relative_path: str, label: str) -> Path:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ConfigurationError(f"{label} path escapes dataset root: {relative_path}") from exc
    if not path.is_file():
        raise ConfigurationError(f"{label} is missing: {path}")
    return path


def validate_training_dataset(config: ResolvedTrainingExperimentConfig) -> ValidatedDataset:
    root = config.dataset_root.resolve()
    manifest_path = config.dataset_manifest.resolve()
    if not root.is_dir():
        raise ConfigurationError(f"training dataset does not exist: {root}")
    try:
        manifest_path.relative_to(root)
    except ValueError as exc:
        raise ConfigurationError("dataset manifest must be inside the dataset root") from exc
    if not manifest_path.is_file():
        raise ConfigurationError(f"dataset manifest does not exist: {manifest_path}")

    records = [DatasetRecord.model_validate(row) for row in read_jsonl(manifest_path)]
    if not records:
        raise ConfigurationError("dataset manifest is empty")
    item_ids = [record.item_id for record in records]
    if len(item_ids) != len(set(item_ids)):
        raise ConfigurationError("dataset manifest contains duplicate item IDs")
    train_records = [record for record in records if record.split == "train"]
    validation_records = [record for record in records if record.split == "validation"]
    if len(train_records) != config.expected_train_images:
        raise ConfigurationError(
            f"dataset has {len(train_records)} training images; "
            f"expected {config.expected_train_images}"
        )
    if len(validation_records) != config.expected_validation_images:
        raise ConfigurationError(
            f"dataset has {len(validation_records)} validation images; "
            f"expected {config.expected_validation_images}"
        )

    recorded_split_files: dict[str, set[Path]] = {"train": set(), "validation": set()}
    for record in records:
        if record.character_id != config.character_id:
            raise ConfigurationError(
                f"dataset item {record.item_id} belongs to {record.character_id}, "
                f"not {config.character_id}"
            )
        image = _resolve_dataset_file(root, record.image_path, "dataset image")
        caption = _resolve_dataset_file(root, record.caption_path, "dataset caption")
        if sha256_file(image) != record.image_sha256:
            raise ConfigurationError(f"dataset image hash mismatch: {record.item_id}")
        if sha256_file(caption) != record.caption_sha256:
            raise ConfigurationError(f"dataset caption hash mismatch: {record.item_id}")
        caption_text = caption.read_text(encoding="utf-8")
        if config.dataset_token not in caption_text or "adult woman" not in caption_text:
            raise ConfigurationError(
                f"dataset caption lacks identity token or class phrase: {record.item_id}"
            )
        recorded_split_files[record.split].update({image, caption})

    train_dir = root / "train"
    for split, expected_files in recorded_split_files.items():
        split_dir = root / split
        if not split_dir.is_dir():
            raise ConfigurationError(f"dataset split directory does not exist: {split_dir}")
        actual_files = {path.resolve() for path in split_dir.iterdir() if path.is_file()}
        if actual_files != expected_files:
            raise ConfigurationError(
                f"{split} directory contents do not exactly match the dataset manifest"
            )
    return ValidatedDataset(
        train_dir=train_dir,
        train_count=len(train_records),
        validation_count=len(validation_records),
        manifest_sha256=sha256_file(manifest_path),
    )


def write_sd_scripts_dataset_config(
    path: Path,
    config: ResolvedTrainingExperimentConfig,
    dataset: ValidatedDataset,
) -> None:
    profile = config.trainer
    content = {
        "general": {
            "caption_extension": ".txt",
            "shuffle_caption": False,
        },
        "datasets": [
            {
                "resolution": [profile.resolution, profile.resolution],
                "batch_size": profile.train_batch_size,
                "enable_bucket": True,
                "bucket_no_upscale": False,
                "min_bucket_reso": profile.min_bucket_reso,
                "max_bucket_reso": profile.max_bucket_reso,
                "bucket_reso_steps": profile.bucket_reso_steps,
                "subsets": [
                    {
                        "image_dir": str(dataset.train_dir.resolve()),
                        "num_repeats": 1,
                    }
                ],
            }
        ],
    }
    atomic_write_text(path, tomli_w.dumps(content))


def stage_training_split(dataset: ValidatedDataset, destination: Path) -> ValidatedDataset:
    if destination.exists():
        raise ConfigurationError(f"staged training directory already exists: {destination}")
    destination.mkdir(parents=True)
    for source in sorted(dataset.train_dir.iterdir(), key=lambda path: path.name):
        if not source.is_file():
            raise ConfigurationError(f"training split contains a non-file entry: {source}")
        staged = destination / source.name
        shutil.copyfile(source, staged)
        if sha256_file(staged) != sha256_file(source):
            raise ConfigurationError(f"staged training file hash mismatch: {source.name}")
    return ValidatedDataset(
        train_dir=destination,
        train_count=dataset.train_count,
        validation_count=dataset.validation_count,
        manifest_sha256=dataset.manifest_sha256,
    )


def _download_base_checkpoint(config: ResolvedTrainingExperimentConfig) -> tuple[Path, str]:
    try:
        import truststore
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise TrainingError(
            "model download dependencies are missing; run `uv sync --extra reference`"
        ) from exc
    if not config.model.weight_file or not config.model.weight_sha256:
        raise TrainingError("training requires a pinned base checkpoint file and SHA-256")
    truststore.inject_into_ssl()
    revision = config.model.resolved_revision or config.model.revision
    arguments = {
        "repo_id": config.model.model_id,
        "filename": config.model.weight_file,
        "revision": revision,
        "cache_dir": config.model.cache_dir.resolve(),
    }
    try:
        try:
            checkpoint = Path(hf_hub_download(**arguments, local_files_only=True))
        except Exception:
            checkpoint = Path(hf_hub_download(**arguments))
    except Exception as exc:
        raise TrainingError(f"cannot load the pinned base checkpoint: {exc}") from exc
    actual_hash = sha256_file(checkpoint)
    if actual_hash != config.model.weight_sha256:
        raise TrainingError("base checkpoint hash does not match the model profile")
    return checkpoint, actual_hash


def _read_telemetry(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError(f"cannot read trainer telemetry {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrainingError("trainer telemetry must be a JSON object")
    for name in ("wall_seconds", "max_reserved_vram_gib", "max_allocated_vram_gib"):
        if not isinstance(value.get(name), int | float):
            raise TrainingError(f"trainer telemetry lacks {name}")
    return value


def _adapter_steps(path: Path, max_train_steps: int) -> int:
    match = re.search(r"-step(\d+)$", path.stem)
    return int(match.group(1)) if match else max_train_steps


def run_training(
    config: ResolvedTrainingExperimentConfig,
    *,
    max_train_steps: int | None = None,
    run_id: str | None = None,
) -> TrainingRunSummary:
    steps = max_train_steps or config.trainer.max_train_steps
    if steps < 1:
        raise ConfigurationError("max training steps must be positive")
    dataset = validate_training_dataset(config)
    selected_run_id = run_id or default_run_id(config.experiment_id)
    run_dir = config.output.root.resolve() / selected_run_id
    if run_dir.exists():
        raise ConfigurationError(f"run directory already exists: {run_dir}")
    adapters_dir = run_dir / "adapters"
    logs_dir = run_dir / "logs"
    run_dir.mkdir(parents=True)
    adapters_dir.mkdir()
    logs_dir.mkdir()
    status = RunStatus(
        state="running",
        experiment_id=config.experiment_id,
        run_id=selected_run_id,
        started_at=utc_now(),
    )
    atomic_write_json(run_dir / "status.json", status.model_dump(mode="json"))
    resolved = config.model_copy(update={"cli_overrides": {"max_train_steps": steps}})
    atomic_write_yaml(run_dir / "resolved-config.yaml", resolved.model_dump(mode="json"))

    try:
        staged_dataset = stage_training_split(dataset, run_dir / "training-data")
        checkpoint, checkpoint_hash = _download_base_checkpoint(config)
        trainer: PreparedTrainer = prepare_sd_scripts_trainer(config.trainer)
        dataset_config_path = run_dir / "dataset.toml"
        write_sd_scripts_dataset_config(dataset_config_path, config, staged_dataset)
        arguments = compile_sd_scripts_arguments(
            config,
            checkpoint_path=checkpoint,
            dataset_config_path=dataset_config_path,
            output_dir=adapters_dir,
            logging_dir=logs_dir,
            max_train_steps=steps,
        )
        telemetry_path = run_dir / "telemetry.json"
        entrypoint = Path(__file__).parents[1] / "trainer_entrypoint.py"
        trainer_script = trainer.source_dir / "sdxl_train_network.py"
        command = [
            str(trainer.runtime_python),
            str(entrypoint.resolve()),
            "--telemetry",
            str(telemetry_path.resolve()),
            "--script",
            str(trainer_script.resolve()),
            "--",
            *arguments,
        ]
        atomic_write_json(run_dir / "command.json", serialize_command(command))
        environment = environment_snapshot(["mara-lab", "tomli-w"], lockfile=Path("uv.lock"))
        environment["trainer"] = {
            "repository": config.trainer.repository,
            "revision": trainer.source_revision,
            "runtime_python": trainer.runtime_python.as_posix(),
            "runtime_lock_sha256": sha256_file(trainer.runtime_lock),
            "packages": runtime_versions(trainer.runtime_python),
        }
        environment["dataset_manifest_sha256"] = dataset.manifest_sha256
        environment["base_checkpoint_sha256"] = checkpoint_hash
        atomic_write_json(run_dir / "environment.json", environment)

        process_environment = os.environ.copy()
        python_path = process_environment.get("PYTHONPATH")
        process_environment["PYTHONPATH"] = str(trainer.source_dir) + (
            os.pathsep + python_path if python_path else ""
        )
        process_environment["PYTHONUTF8"] = "1"
        process_environment["PYTHONIOENCODING"] = "utf-8"
        process_environment["HF_HOME"] = str((Path(".cache") / "huggingface-home").resolve())
        process_environment["HF_HUB_CACHE"] = str(config.model.cache_dir.resolve())
        process_environment["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        process_environment.setdefault(
            "MPLCONFIGDIR", str((Path(".cache") / "matplotlib").resolve())
        )
        trainer_log = run_dir / "trainer.log"
        with trainer_log.open("w", encoding="utf-8", newline="") as log_handle:
            result = subprocess.run(
                command,
                cwd=trainer.source_dir,
                env=process_environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode:
            tail = trainer_log.read_text(encoding="utf-8", errors="replace")[-5000:]
            raise TrainingError(
                f"sd-scripts exited with code {result.returncode}; see {trainer_log}\n{tail}"
            )
        telemetry = _read_telemetry(telemetry_path)
        peak_reserved = float(telemetry["max_reserved_vram_gib"])
        if peak_reserved > config.compute.max_reserved_vram_gib:
            raise MemoryBudgetExceededError(
                f"training exceeded VRAM budget: {peak_reserved:.2f} GiB > "
                f"{config.compute.max_reserved_vram_gib:.2f} GiB"
            )
        log_text = trainer_log.read_text(encoding="utf-8", errors="replace")
        if re.search(r"\bloss\s*[=:]\s*(?:nan|inf)\b", log_text, flags=re.IGNORECASE):
            raise TrainingError("trainer reported a non-finite loss")

        adapter_paths = sorted(adapters_dir.glob("*.safetensors"))
        if not adapter_paths:
            raise TrainingError("trainer completed without producing a safetensors adapter")
        manifest_path = adapters_dir / "manifest.jsonl"
        for adapter_path in adapter_paths:
            read_safetensors_layout(adapter_path)
            tensor_statistics = inspect_sd_scripts_adapter(trainer.runtime_python, adapter_path)
            record = AdapterRecord(
                adapter_id=adapter_path.stem,
                character_id=config.character_id,
                adapter_path=adapter_path.relative_to(run_dir).as_posix(),
                sha256=sha256_file(adapter_path),
                base_model_id=config.model.model_id,
                base_revision=config.model.resolved_revision or config.model.revision,
                base_weight_sha256=config.model.weight_sha256,
                trainer_repository=config.trainer.repository,
                trainer_revision=config.trainer.revision,
                dataset_manifest_sha256=dataset.manifest_sha256,
                train_steps=_adapter_steps(adapter_path, steps),
                network_dim=config.trainer.network_dim,
                network_alpha=config.trainer.network_alpha,
                tensor_count=int(tensor_statistics["tensor_count"]),
                up_tensor_count=int(tensor_statistics["up_tensor_count"]),
                nonzero_up_tensor_count=int(tensor_statistics["nonzero_up_tensor_count"]),
                max_abs_weight=float(tensor_statistics["max_abs_weight"]),
                max_reserved_vram_gib=peak_reserved,
                created_at=utc_now(),
            )
            append_jsonl(manifest_path, record.model_dump(mode="json"))
        completed = status.model_copy(
            update={
                "state": "completed",
                "generated_count": len(adapter_paths),
                "finished_at": utc_now(),
            }
        )
        atomic_write_json(run_dir / "status.json", completed.model_dump(mode="json"))
        return TrainingRunSummary(
            run_id=selected_run_id,
            run_dir=run_dir,
            adapters=adapter_paths,
            adapter_manifest_path=manifest_path,
            max_reserved_vram_gib=peak_reserved,
            wall_seconds=float(telemetry["wall_seconds"]),
        )
    except Exception as exc:
        failed = status.model_copy(
            update={
                "state": "failed",
                "finished_at": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_write_json(run_dir / "status.json", failed.model_dump(mode="json"))
        raise
