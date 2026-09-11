from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from mara_lab.artifacts import (
    append_jsonl,
    atomic_write_json,
    atomic_write_yaml,
    default_run_id,
    environment_snapshot,
    save_png_atomic,
    sha256_file,
    utc_now,
)
from mara_lab.backends import ImageBackend, create_backend
from mara_lab.config import ResolvedExperimentConfig
from mara_lab.contact_sheet import create_contact_sheet
from mara_lab.domain import GenerationRequest
from mara_lab.errors import ConfigurationError, MemoryBudgetExceededError
from mara_lab.manifests import CandidateRecord, RunStatus

BackendFactory = Callable[[ResolvedExperimentConfig], ImageBackend]


@dataclass(frozen=True, slots=True)
class CandidateRunSummary:
    run_id: str
    run_dir: Path
    generated_count: int
    manifest_path: Path
    contact_sheet_path: Path
    max_reserved_vram_gib: float


def _default_backend_factory(config: ResolvedExperimentConfig) -> ImageBackend:
    return create_backend(config.model, config.compute)


def _make_logger(run_dir: Path) -> tuple[logging.Logger, logging.Handler]:
    logger = logging.getLogger(f"mara_lab.run.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger, handler


def run_candidates(
    config: ResolvedExperimentConfig,
    *,
    seed_limit: int | None = None,
    run_id: str | None = None,
    backend_factory: BackendFactory = _default_backend_factory,
) -> CandidateRunSummary:
    seeds = config.generation.seeds.values(seed_limit)
    selected_run_id = run_id or default_run_id(config.experiment_id)
    run_dir = config.output.root / selected_run_id
    if run_dir.exists():
        raise ConfigurationError(f"run directory already exists: {run_dir}")

    outputs_dir = run_dir / "outputs"
    contact_sheet_path = run_dir / "contact-sheets" / "candidates.png"
    manifest_path = outputs_dir / "manifest.jsonl"
    run_dir.mkdir(parents=True)
    outputs_dir.mkdir()
    logger, handler = _make_logger(run_dir)
    started_at = utc_now()
    records: list[CandidateRecord] = []
    max_reserved = 0.0
    backend: ImageBackend | None = None

    status = RunStatus(
        state="running",
        experiment_id=config.experiment_id,
        run_id=selected_run_id,
        started_at=started_at,
    )
    atomic_write_json(run_dir / "status.json", status.model_dump(mode="json"))
    atomic_write_yaml(run_dir / "resolved-config.yaml", config.model_dump(mode="json"))
    logger.info("starting candidate run %s with %d seed(s)", selected_run_id, len(seeds))

    try:
        backend = backend_factory(config)
        backend_info = backend.prepare()
        config = config.model_copy(
            update={
                "model": config.model.model_copy(
                    update={"resolved_revision": backend_info.resolved_revision}
                )
            }
        )
        atomic_write_yaml(run_dir / "resolved-config.yaml", config.model_dump(mode="json"))
        environment = environment_snapshot(
            [
                "accelerate",
                "diffusers",
                "huggingface-hub",
                "mara-lab",
                "pillow",
                "pydantic",
                "pyyaml",
                "torch",
                "transformers",
            ]
        )
        environment["backend"] = backend_info.to_dict()
        atomic_write_json(run_dir / "environment.json", environment)

        if backend_info.resident_reserved_vram_gib > config.compute.max_reserved_vram_gib:
            raise MemoryBudgetExceededError(
                "model residency exceeds VRAM budget: "
                f"{backend_info.resident_reserved_vram_gib:.2f} GiB > "
                f"{config.compute.max_reserved_vram_gib:.2f} GiB"
            )

        for sequence, seed in enumerate(seeds, start=1):
            artifact_id = f"candidate-{sequence:04d}-seed-{seed}"
            logger.info("generating %s", artifact_id)
            positive_prompt, prompt_variant = config.prompts.render_positive(sequence)
            request = GenerationRequest(
                prompt=positive_prompt,
                negative_prompt=config.prompts.negative,
                seed=seed,
                width=config.generation.width,
                height=config.generation.height,
                steps=config.generation.steps,
                guidance_scale=config.generation.guidance_scale,
            )
            result = backend.generate(request)
            if result.image.size != (request.width, request.height):
                expected_size = (request.width, request.height)
                raise ConfigurationError(
                    f"backend returned {result.image.size}, expected {expected_size}"
                )

            image_relative_path = Path("outputs") / f"{artifact_id}.png"
            image_path = run_dir / image_relative_path
            save_png_atomic(result.image, image_path)
            max_reserved = max(max_reserved, result.metrics.max_reserved_vram_gib)
            record = CandidateRecord(
                artifact_id=artifact_id,
                sequence=sequence,
                character_id=config.character_id,
                image_path=image_relative_path.as_posix(),
                sha256=sha256_file(image_path),
                seed=seed,
                width=request.width,
                height=request.height,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                model=backend_info.to_dict(),
                generation={
                    "steps": request.steps,
                    "guidance_scale": request.guidance_scale,
                    "scheduler": config.model.scheduler,
                    "identity_variant": prompt_variant,
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
        )
        status = status.model_copy(
            update={
                "state": "completed",
                "generated_count": len(records),
                "finished_at": utc_now(),
            }
        )
        atomic_write_json(run_dir / "status.json", status.model_dump(mode="json"))
        logger.info("completed candidate run with %d image(s)", len(records))
    except Exception as exc:
        if records:
            create_contact_sheet(
                run_dir,
                records,
                contact_sheet_path,
                config.output.contact_sheet_columns,
                config.output.contact_sheet_thumbnail_width,
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
        logger.exception("candidate run failed")
        raise
    finally:
        if backend is not None:
            backend.close()
        logger.removeHandler(handler)
        handler.close()

    return CandidateRunSummary(
        run_id=selected_run_id,
        run_dir=run_dir,
        generated_count=len(records),
        manifest_path=manifest_path,
        contact_sheet_path=contact_sheet_path,
        max_reserved_vram_gib=max_reserved,
    )
