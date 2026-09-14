from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
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
from mara_lab.backends.reference import ReferenceImageBackend, create_reference_backend
from mara_lab.config import (
    ReferenceCell,
    ResolvedReferenceExperimentConfig,
    load_resolved_reference_config,
)
from mara_lab.contact_sheet import ContactSheetRecord, create_contact_sheet
from mara_lab.domain import ReferenceGenerationRequest, ReferenceShot
from mara_lab.errors import ConfigurationError, MemoryBudgetExceededError
from mara_lab.manifests import ReferenceRecord, RunStatus

ReferenceBackendFactory = Callable[[ResolvedReferenceExperimentConfig], ReferenceImageBackend]


@dataclass(frozen=True, slots=True)
class ReferenceRunSummary:
    run_id: str
    run_dir: Path
    generated_count: int
    manifest_path: Path
    contact_sheet_path: Path
    max_reserved_vram_gib: float


def _make_logger(run_dir: Path) -> tuple[logging.Logger, logging.Handler]:
    logger = logging.getLogger(f"mara_lab.references.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger, handler


def _sheet_label(record: ContactSheetRecord) -> str:
    parts = record.artifact_id.split("-")
    return f"{parts[1].upper()}-{parts[2]} s{record.seed}"


def _artifact_id(
    config: ResolvedReferenceExperimentConfig,
    cell_id: str,
    seed_index: int,
    seed: int,
) -> str:
    normalized_cell = f"{config.pass_id}{cell_id.removeprefix(config.pass_id)}"
    return f"reference-{normalized_cell}-{seed_index:02d}-seed-{seed}"


def _request_for_job(
    config: ResolvedReferenceExperimentConfig,
    cell: ReferenceCell,
    seed: int,
) -> ReferenceGenerationRequest:
    shot = ReferenceShot(**cell.model_dump())
    return ReferenceGenerationRequest(
        shot=shot,
        seed=seed,
        width=config.generation.width,
        height=config.generation.height,
        steps=config.generation.steps,
        guidance_scale=config.generation.guidance_scale,
        start_merge_step=config.generation.start_merge_step,
    )


def _read_status(path: Path) -> RunStatus:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read run status {path}: {exc}") from exc
    return RunStatus.model_validate(value)


def _resume_config_payload(config: ResolvedReferenceExperimentConfig) -> dict[str, Any]:
    return config.model_dump(
        mode="json",
        exclude={"cli_overrides", "maximum_reference_images"},
    )


def _load_resume_records(
    run_dir: Path,
    config: ResolvedReferenceExperimentConfig,
    jobs: list[tuple[ReferenceCell, int, int]],
    source_references: list[dict[str, str]],
) -> tuple[RunStatus, list[ReferenceRecord]]:
    status = _read_status(run_dir / "status.json")
    if status.experiment_id != config.experiment_id or status.run_id != run_dir.name:
        raise ConfigurationError("resume target status does not match the requested experiment")
    if status.state == "completed":
        raise ConfigurationError(f"reference run is already completed: {run_dir}")
    stored_config = load_resolved_reference_config(run_dir / "resolved-config.yaml")
    if _resume_config_payload(stored_config) != _resume_config_payload(config):
        raise ConfigurationError("resume configuration does not match the original run")
    manifest_path = run_dir / "outputs" / "manifest.jsonl"
    try:
        records = (
            [ReferenceRecord.model_validate(row) for row in read_jsonl(manifest_path)]
            if manifest_path.is_file()
            else []
        )
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"cannot read resume manifest {manifest_path}: {exc}") from exc
    if len(records) > len(jobs):
        raise ConfigurationError("resume manifest contains more records than planned jobs")
    for sequence, (record, job) in enumerate(zip(records, jobs, strict=False), start=1):
        cell, seed_index, seed = job
        expected_id = _artifact_id(config, cell.cell_id, seed_index, seed)
        expected_generation = {
            "steps": config.generation.steps,
            "guidance_scale": config.generation.guidance_scale,
            "start_merge_step": config.generation.start_merge_step,
        }
        if (
            record.sequence != sequence
            or record.artifact_id != expected_id
            or record.character_id != config.character_id
            or record.pass_id != config.pass_id
            or record.cell_id != cell.cell_id
            or record.seed != seed
            or record.width != config.generation.width
            or record.height != config.generation.height
            or record.shot != cell.model_dump()
            or record.negative_prompt != config.negative_prompt
            or record.source_references != source_references
            or record.generation != expected_generation
        ):
            raise ConfigurationError(
                f"resume manifest diverges from planned job {sequence}: {record.artifact_id}"
            )
        image_path = (run_dir / record.image_path).resolve()
        try:
            image_path.relative_to(run_dir.resolve())
        except ValueError as exc:
            raise ConfigurationError(
                f"resume artifact path escapes its run: {record.image_path}"
            ) from exc
        if not image_path.is_file() or sha256_file(image_path) != record.sha256:
            raise ConfigurationError(
                f"resume artifact is missing or modified: {record.artifact_id}"
            )
    return status, records


def run_references(
    config: ResolvedReferenceExperimentConfig,
    *,
    image_limit: int | None = None,
    run_id: str | None = None,
    resume: bool = False,
    backend_factory: ReferenceBackendFactory = create_reference_backend,
) -> ReferenceRunSummary:
    if len(config.reference_images) < config.minimum_reference_images:
        raise ConfigurationError(
            f"experiment requires at least {config.minimum_reference_images} reference images; "
            f"got {len(config.reference_images)}"
        )
    if len(config.reference_images) > config.maximum_reference_images:
        raise ConfigurationError(
            f"experiment accepts at most {config.maximum_reference_images} reference images; "
            f"got {len(config.reference_images)}"
        )
    jobs = [
        (cell, seed_index, seed)
        for cell in config.cells
        for seed_index, seed in enumerate(config.generation.seeds.values(), start=1)
    ]
    if image_limit is not None:
        if image_limit < 1:
            raise ConfigurationError("image limit must be at least 1")
        jobs = jobs[:image_limit]
    if resume and run_id is None:
        raise ConfigurationError("resume requires an explicit run ID")
    selected_run_id = run_id or default_run_id(config.experiment_id)
    run_dir = config.output.root / selected_run_id
    if run_dir.exists() and not resume:
        raise ConfigurationError(f"run directory already exists: {run_dir}")
    if resume and not run_dir.is_dir():
        raise ConfigurationError(f"resume run directory does not exist: {run_dir}")

    outputs_dir = run_dir / "outputs"
    manifest_path = outputs_dir / "manifest.jsonl"
    contact_sheet_path = run_dir / "contact-sheets" / "references.png"
    source_references = [
        {"path": path.as_posix(), "sha256": sha256_file(path)} for path in config.reference_images
    ]
    previous_status: RunStatus | None = None
    if resume:
        if not outputs_dir.is_dir():
            raise ConfigurationError(f"resume outputs directory does not exist: {outputs_dir}")
        previous_status, records = _load_resume_records(
            run_dir,
            config,
            jobs,
            source_references,
        )
        status = previous_status.model_copy(
            update={
                "state": "running",
                "generated_count": len(records),
                "finished_at": None,
                "error": None,
            }
        )
    else:
        run_dir.mkdir(parents=True)
        outputs_dir.mkdir()
        records = []
        status = RunStatus(
            state="running",
            experiment_id=config.experiment_id,
            run_id=selected_run_id,
            started_at=utc_now(),
        )
        atomic_write_yaml(run_dir / "resolved-config.yaml", config.model_dump(mode="json"))

    logger, handler = _make_logger(run_dir)
    atomic_write_json(run_dir / "status.json", status.model_dump(mode="json"))
    backend: ReferenceImageBackend | None = None
    max_reserved = max(
        (float(record.metrics.get("max_reserved_vram_gib", 0.0)) for record in records),
        default=0.0,
    )

    try:
        remaining_jobs = jobs[len(records) :]
        if remaining_jobs or records:
            backend = backend_factory(config)
            for record, (cell, _, seed) in zip(records, jobs, strict=False):
                prompt = backend.compile_prompt(_request_for_job(config, cell, seed))
                if record.prompt != prompt.positive or record.negative_prompt != prompt.negative:
                    raise ConfigurationError(
                        f"resume prompt diverges for existing artifact: {record.artifact_id}"
                    )

        if remaining_jobs:
            backend_info = backend.prepare()
            environment = environment_snapshot(
                [
                    "accelerate",
                    "diffusers",
                    "insightface",
                    "mara-lab",
                    "onnxruntime",
                    "peft",
                    "photomaker",
                    "torch",
                    "transformers",
                ]
            )
            environment["backend"] = backend_info.to_dict()
            if resume:
                append_jsonl(
                    run_dir / "resume-events.jsonl",
                    {
                        "event": "resume",
                        "resumed_at": utc_now(),
                        "previous_status": previous_status.model_dump(mode="json"),
                        "existing_count": len(records),
                        "environment": environment,
                    },
                )
            else:
                atomic_write_json(run_dir / "environment.json", environment)

        for sequence, (cell, seed_index, seed) in enumerate(
            remaining_jobs,
            start=len(records) + 1,
        ):
            request = _request_for_job(config, cell, seed)
            prompt = backend.compile_prompt(request)
            artifact_id = _artifact_id(config, cell.cell_id, seed_index, seed)
            logger.info("generating %s", artifact_id)
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
            record = ReferenceRecord(
                artifact_id=artifact_id,
                sequence=sequence,
                character_id=config.character_id,
                pass_id=config.pass_id,
                cell_id=cell.cell_id,
                image_path=relative_path.as_posix(),
                sha256=sha256_file(output_path),
                seed=seed,
                width=request.width,
                height=request.height,
                shot=asdict(request.shot),
                prompt=prompt.positive,
                negative_prompt=prompt.negative,
                source_references=source_references,
                model=backend_info.to_dict(),
                conditioner=config.conditioner.model_dump(mode="json"),
                generation={
                    "steps": request.steps,
                    "guidance_scale": request.guidance_scale,
                    "start_merge_step": request.start_merge_step,
                },
                metrics=result.metrics.to_dict(),
                created_at=utc_now(),
            )
            append_jsonl(manifest_path, record.model_dump(mode="json"))
            records.append(record)
            status = status.model_copy(update={"generated_count": len(records)})
            atomic_write_json(run_dir / "status.json", status.model_dump(mode="json"))
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
        status = status.model_copy(
            update={
                "state": "completed",
                "generated_count": len(records),
                "finished_at": utc_now(),
            }
        )
        atomic_write_json(run_dir / "status.json", status.model_dump(mode="json"))
    except Exception as exc:
        logger.exception("reference run failed")
        contact_sheet_error: Exception | None = None
        if records:
            try:
                create_contact_sheet(
                    run_dir,
                    records,
                    contact_sheet_path,
                    config.output.contact_sheet_columns,
                    config.output.contact_sheet_thumbnail_width,
                    labeler=_sheet_label,
                )
            except Exception as sheet_exc:
                contact_sheet_error = sheet_exc
                logger.exception("could not refresh the partial contact sheet")
        error = f"{type(exc).__name__}: {exc}"
        if contact_sheet_error is not None:
            error += (
                "; contact sheet error: "
                f"{type(contact_sheet_error).__name__}: {contact_sheet_error}"
            )
        failed = status.model_copy(
            update={
                "state": "failed",
                "generated_count": len(records),
                "finished_at": utc_now(),
                "error": error,
            }
        )
        try:
            atomic_write_json(run_dir / "status.json", failed.model_dump(mode="json"))
        except Exception:
            logger.exception("could not persist failed run status")
        raise
    finally:
        if backend is not None:
            backend.close()
        logger.removeHandler(handler)
        handler.close()

    return ReferenceRunSummary(
        run_id=selected_run_id,
        run_dir=run_dir,
        generated_count=len(records),
        manifest_path=manifest_path,
        contact_sheet_path=contact_sheet_path,
        max_reserved_vram_gib=max_reserved,
    )
