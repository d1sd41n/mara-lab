from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mara_lab.artifacts import atomic_write_json, read_jsonl, save_png_atomic, sha256_file, utc_now
from mara_lab.backends import create_backend
from mara_lab.config import load_resolved_config
from mara_lab.domain import GenerationRequest
from mara_lab.errors import ConfigurationError, MemoryBudgetExceededError, ReproductionMismatchError
from mara_lab.manifests import CandidateRecord


@dataclass(frozen=True, slots=True)
class ReproductionSummary:
    output_path: Path
    expected_sha256: str
    actual_sha256: str
    exact_match: bool


def reproduce_candidate(
    run_dir: Path, artifact_id: str, *, strict: bool = True
) -> ReproductionSummary:
    config = load_resolved_config(run_dir / "resolved-config.yaml")
    records = [
        CandidateRecord.model_validate(row)
        for row in read_jsonl(run_dir / "outputs" / "manifest.jsonl")
    ]
    record = next((item for item in records if item.artifact_id == artifact_id), None)
    if record is None:
        raise ConfigurationError(f"artifact not found in manifest: {artifact_id}")

    backend = create_backend(config.model, config.compute)
    try:
        info = backend.prepare()
        if info.resolved_revision != record.model["resolved_revision"]:
            raise ConfigurationError(
                "resolved model revision differs from the original artifact manifest"
            )
        original_weight_sha256 = record.model.get("details", {}).get("selected_weight_sha256")
        current_weight_sha256 = info.details.get("selected_weight_sha256")
        if original_weight_sha256 and current_weight_sha256 != original_weight_sha256:
            raise ConfigurationError(
                "model weight hash differs from the original artifact manifest"
            )
        result = backend.generate(
            GenerationRequest(
                prompt=record.prompt,
                negative_prompt=record.negative_prompt,
                seed=record.seed,
                width=record.width,
                height=record.height,
                steps=int(record.generation["steps"]),
                guidance_scale=float(record.generation["guidance_scale"]),
            )
        )
    finally:
        backend.close()

    if result.metrics.max_reserved_vram_gib > config.compute.max_reserved_vram_gib:
        raise MemoryBudgetExceededError(
            f"reproduction used {result.metrics.max_reserved_vram_gib:.2f} GiB, above budget"
        )

    output_path = run_dir / "reproduced" / f"{artifact_id}.png"
    save_png_atomic(result.image, output_path)
    actual_sha256 = sha256_file(output_path)
    exact_match = actual_sha256 == record.sha256
    report = {
        "schema_version": 1,
        "artifact_id": artifact_id,
        "created_at": utc_now(),
        "expected_sha256": record.sha256,
        "actual_sha256": actual_sha256,
        "exact_match": exact_match,
        "metrics": result.metrics.to_dict(),
    }
    atomic_write_json(run_dir / "reproduced" / f"{artifact_id}.json", report)
    if strict and not exact_match:
        raise ReproductionMismatchError(
            f"reproduction hash mismatch for {artifact_id}; see {output_path.parent}"
        )
    return ReproductionSummary(
        output_path=output_path,
        expected_sha256=record.sha256,
        actual_sha256=actual_sha256,
        exact_match=exact_match,
    )
