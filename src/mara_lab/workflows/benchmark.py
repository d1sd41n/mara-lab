from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mara_lab.artifacts import (
    append_jsonl,
    atomic_write_json,
    atomic_write_yaml,
    default_run_id,
    environment_snapshot,
    read_jsonl,
    save_png_atomic,
    sha256_file,
    utc_now,
)
from mara_lab.backends.base import ImageBackend
from mara_lab.backends.realvisxl import RealVisXLBackend
from mara_lab.config import ResolvedBenchmarkExperimentConfig
from mara_lab.contact_sheet import create_contact_sheet
from mara_lab.domain import GenerationRequest
from mara_lab.errors import ConfigurationError, MemoryBudgetExceededError
from mara_lab.manifests import AdapterRecord, BenchmarkRecord, RunStatus
from mara_lab.model_materialization import read_safetensors_layout

BenchmarkBackendFactory = Callable[[ResolvedBenchmarkExperimentConfig, Path, float], ImageBackend]


@dataclass(frozen=True, slots=True)
class BenchmarkRunSummary:
    run_id: str
    run_dir: Path
    generated_count: int
    manifest_path: Path
    contact_sheet_path: Path
    max_reserved_vram_gib: float


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read JSON document {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"JSON document {path} must contain an object")
    return value


def validate_adapter_for_benchmark(
    adapter_path: Path, config: ResolvedBenchmarkExperimentConfig
) -> AdapterRecord:
    adapter_path = adapter_path.resolve()
    if not adapter_path.is_file():
        raise ConfigurationError(f"adapter does not exist: {adapter_path}")
    layout = read_safetensors_layout(adapter_path)
    if not layout.tensors:
        raise ConfigurationError(f"adapter contains no tensors: {adapter_path}")

    run_dir = adapter_path.parent.parent
    status = RunStatus.model_validate(_read_json_object(run_dir / "status.json"))
    if status.state != "completed":
        raise ConfigurationError(f"adapter run is not completed: {status.state}")
    manifest_path = adapter_path.parent / "manifest.jsonl"
    if not manifest_path.is_file():
        raise ConfigurationError(f"adapter manifest does not exist: {manifest_path}")
    records = [AdapterRecord.model_validate(row) for row in read_jsonl(manifest_path)]
    matches: list[AdapterRecord] = []
    for record in records:
        recorded_path = (run_dir / record.adapter_path).resolve()
        try:
            recorded_path.relative_to(run_dir)
        except ValueError as exc:
            raise ConfigurationError(
                f"adapter manifest path escapes its run: {record.adapter_path}"
            ) from exc
        if recorded_path == adapter_path:
            matches.append(record)
    if len(matches) != 1:
        raise ConfigurationError("adapter manifest must contain exactly one matching record")
    record = matches[0]
    if sha256_file(adapter_path) != record.sha256:
        raise ConfigurationError("adapter hash does not match its manifest")
    if record.character_id != config.character_id:
        raise ConfigurationError(
            f"adapter belongs to {record.character_id}, not {config.character_id}"
        )
    expected_revision = config.model.resolved_revision or config.model.revision
    if (
        record.base_model_id != config.model.model_id
        or record.base_revision != expected_revision
        or record.base_weight_sha256 != config.model.weight_sha256
    ):
        raise ConfigurationError("adapter was trained against a different base checkpoint")
    return record


def _default_backend_factory(
    config: ResolvedBenchmarkExperimentConfig,
    adapter_path: Path,
    adapter_weight: float,
) -> ImageBackend:
    return RealVisXLBackend(
        config.model,
        config.compute,
        adapter_path=adapter_path,
        adapter_weight=adapter_weight,
    )


def _make_logger(run_dir: Path) -> tuple[logging.Logger, logging.Handler]:
    logger = logging.getLogger(f"mara_lab.benchmark.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger, handler


def _sheet_label(record: BenchmarkRecord) -> str:
    return f"{record.case_id} w{record.adapter_weight:g} s{record.seed}"


def run_benchmark(
    config: ResolvedBenchmarkExperimentConfig,
    adapter_path: Path,
    *,
    adapter_weight: float = 1.0,
    case_limit: int | None = None,
    run_id: str | None = None,
    backend_factory: BenchmarkBackendFactory = _default_backend_factory,
) -> BenchmarkRunSummary:
    if not 0 < adapter_weight <= 2:
        raise ConfigurationError("adapter weight must be above 0 and at most 2")
    cases = config.cases
    if case_limit is not None:
        if case_limit < 1:
            raise ConfigurationError("case limit must be at least 1")
        cases = cases[:case_limit]
    adapter_path = adapter_path.resolve()
    adapter_record = validate_adapter_for_benchmark(adapter_path, config)

    selected_run_id = run_id or default_run_id(config.experiment_id)
    run_dir = config.output.root.resolve() / selected_run_id
    if run_dir.exists():
        raise ConfigurationError(f"run directory already exists: {run_dir}")
    outputs_dir = run_dir / "outputs"
    manifest_path = outputs_dir / "manifest.jsonl"
    contact_sheet_path = run_dir / "contact-sheets" / "benchmark.png"
    run_dir.mkdir(parents=True)
    outputs_dir.mkdir()
    logger, handler = _make_logger(run_dir)
    status = RunStatus(
        state="running",
        experiment_id=config.experiment_id,
        run_id=selected_run_id,
        started_at=utc_now(),
    )
    atomic_write_json(run_dir / "status.json", status.model_dump(mode="json"))
    resolved = config.model_copy(
        update={
            "cli_overrides": {
                "adapter_path": adapter_path.as_posix(),
                "adapter_weight": adapter_weight,
                "case_limit": case_limit,
            }
        }
    )
    atomic_write_yaml(run_dir / "resolved-config.yaml", resolved.model_dump(mode="json"))
    records: list[BenchmarkRecord] = []
    max_reserved = 0.0
    backend: ImageBackend | None = None

    try:
        backend = backend_factory(config, adapter_path, adapter_weight)
        backend_info = backend.prepare()
        environment = environment_snapshot(
            [
                "accelerate",
                "diffusers",
                "huggingface-hub",
                "mara-lab",
                "safetensors",
                "torch",
                "transformers",
            ]
        )
        environment["backend"] = backend_info.to_dict()
        environment["adapter_manifest_record"] = adapter_record.model_dump(mode="json")
        atomic_write_json(run_dir / "environment.json", environment)

        if backend_info.resident_reserved_vram_gib > config.compute.max_reserved_vram_gib:
            raise MemoryBudgetExceededError(
                "model residency exceeds VRAM budget: "
                f"{backend_info.resident_reserved_vram_gib:.2f} GiB > "
                f"{config.compute.max_reserved_vram_gib:.2f} GiB"
            )

        weight_tag = f"{adapter_weight:.2f}".replace(".", "p")
        for sequence, case in enumerate(cases, start=1):
            artifact_id = f"benchmark-{case.case_id}-seed-{case.seed}-w-{weight_tag}"
            logger.info("generating %s", artifact_id)
            request = GenerationRequest(
                prompt=case.prompt,
                negative_prompt=config.negative_prompt,
                seed=case.seed,
                width=config.generation.width,
                height=config.generation.height,
                steps=config.generation.steps,
                guidance_scale=config.generation.guidance_scale,
            )
            result = backend.generate(request)
            if result.image.size != (request.width, request.height):
                raise ConfigurationError(
                    f"backend returned {result.image.size}, expected "
                    f"{(request.width, request.height)}"
                )
            relative_path = Path("outputs") / f"{artifact_id}.png"
            output_path = run_dir / relative_path
            save_png_atomic(result.image, output_path)
            max_reserved = max(max_reserved, result.metrics.max_reserved_vram_gib)
            record = BenchmarkRecord(
                artifact_id=artifact_id,
                sequence=sequence,
                case_id=case.case_id,
                character_id=config.character_id,
                image_path=relative_path.as_posix(),
                sha256=sha256_file(output_path),
                seed=case.seed,
                width=request.width,
                height=request.height,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                adapter=adapter_record,
                adapter_weight=adapter_weight,
                model=backend_info.to_dict(),
                generation={
                    "steps": request.steps,
                    "guidance_scale": request.guidance_scale,
                    "scheduler": config.model.scheduler,
                },
                metrics=result.metrics.to_dict(),
                created_at=utc_now(),
            )
            append_jsonl(manifest_path, record.model_dump(mode="json"))
            records.append(record)
            if result.metrics.max_reserved_vram_gib > config.compute.max_reserved_vram_gib:
                raise MemoryBudgetExceededError(
                    f"{artifact_id} exceeded VRAM budget: "
                    f"{result.metrics.max_reserved_vram_gib:.2f} GiB > "
                    f"{config.compute.max_reserved_vram_gib:.2f} GiB"
                )

        create_contact_sheet(
            run_dir,
            records,
            contact_sheet_path,
            config.output.contact_sheet_columns,
            config.output.contact_sheet_thumbnail_width,
            labeler=_sheet_label,
        )
        completed = status.model_copy(
            update={
                "state": "completed",
                "generated_count": len(records),
                "finished_at": utc_now(),
            }
        )
        atomic_write_json(run_dir / "status.json", completed.model_dump(mode="json"))
    except Exception as exc:
        if records:
            create_contact_sheet(
                run_dir,
                records,
                contact_sheet_path,
                config.output.contact_sheet_columns,
                config.output.contact_sheet_thumbnail_width,
                labeler=_sheet_label,
            )
        failed = status.model_copy(
            update={
                "state": "failed",
                "generated_count": len(records),
                "finished_at": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_write_json(run_dir / "status.json", failed.model_dump(mode="json"))
        logger.exception("benchmark run failed")
        raise
    finally:
        if backend is not None:
            backend.close()
        logger.removeHandler(handler)
        handler.close()

    return BenchmarkRunSummary(
        run_id=selected_run_id,
        run_dir=run_dir,
        generated_count=len(records),
        manifest_path=manifest_path,
        contact_sheet_path=contact_sheet_path,
        max_reserved_vram_gib=max_reserved,
    )
