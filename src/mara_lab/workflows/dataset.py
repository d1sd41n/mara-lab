from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from mara_lab.artifacts import (
    append_jsonl,
    atomic_write_text,
    atomic_write_yaml,
    read_jsonl,
    sha256_file,
    utc_now,
)
from mara_lab.errors import ConfigurationError
from mara_lab.manifests import DatasetRecord, ReferenceRecord, RunStatus


@dataclass(frozen=True, slots=True)
class DatasetBuildSummary:
    dataset_root: Path
    manifest_path: Path
    train_count: int
    validation_count: int


def compile_training_caption(record: ReferenceRecord, token: str) -> str:
    shot = record.shot
    required = (
        "capture_style",
        "framing",
        "view",
        "expression",
        "wardrobe",
        "setting",
        "lighting",
        "camera_behavior",
    )
    missing = [name for name in required if not isinstance(shot.get(name), str)]
    if missing:
        raise ConfigurationError(
            f"source artifact {record.artifact_id} lacks shot fields: {', '.join(missing)}"
        )
    return (
        f"a {shot['capture_style']} {shot['framing']} photo of {token}, an adult woman, "
        f"{shot['view']}, {shot['expression']}, wearing {shot['wardrobe']}, in "
        f"{shot['setting']}, lit by {shot['lighting']}, {shot['camera_behavior']}"
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read JSON document {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"JSON document {path} must contain an object")
    return value


def _source_image(run_dir: Path, record: ReferenceRecord) -> Path:
    image = (run_dir / record.image_path).resolve()
    try:
        image.relative_to(run_dir)
    except ValueError as exc:
        raise ConfigurationError(f"artifact path escapes source run: {record.image_path}") from exc
    if not image.is_file():
        raise ConfigurationError(f"source image is missing: {image}")
    actual_hash = sha256_file(image)
    if actual_hash != record.sha256:
        raise ConfigurationError(
            f"source hash mismatch for {record.artifact_id}: {actual_hash} != {record.sha256}"
        )
    return image


def build_character_dataset(
    run_dir: Path,
    train_artifact_ids: list[str],
    validation_artifact_ids: list[str],
    *,
    dataset_root: Path,
    dataset_id: str,
    token: str,
    expected_train_count: int = 28,
    expected_validation_count: int = 6,
    source_pass_id: Literal["a", "b", "c"] = "c",
) -> DatasetBuildSummary:
    if len(train_artifact_ids) != expected_train_count:
        raise ConfigurationError(
            f"training selection has {len(train_artifact_ids)} images; "
            f"expected {expected_train_count}"
        )
    if len(validation_artifact_ids) != expected_validation_count:
        raise ConfigurationError(
            f"validation selection has {len(validation_artifact_ids)} images; "
            f"expected {expected_validation_count}"
        )
    all_ids = [*train_artifact_ids, *validation_artifact_ids]
    if len(all_ids) != len(set(all_ids)):
        raise ConfigurationError("training and validation selections must be unique")

    run_dir = run_dir.resolve()
    status = RunStatus.model_validate(_read_json_object(run_dir / "status.json"))
    if status.state != "completed":
        raise ConfigurationError(f"source run is not completed: {status.state}")
    source_records = [
        ReferenceRecord.model_validate(row)
        for row in read_jsonl(run_dir / "outputs" / "manifest.jsonl")
    ]
    records_by_id = {record.artifact_id: record for record in source_records}
    if len(records_by_id) != len(source_records):
        raise ConfigurationError("source manifest contains duplicate artifact IDs")
    missing = [artifact_id for artifact_id in all_ids if artifact_id not in records_by_id]
    if missing:
        raise ConfigurationError("selected artifacts are missing: " + ", ".join(missing))

    selected = [records_by_id[artifact_id] for artifact_id in all_ids]
    if any(record.pass_id != source_pass_id for record in selected):
        raise ConfigurationError(
            f"dataset selections must come from reference pass {source_pass_id}"
        )
    character_ids = {record.character_id for record in selected}
    if len(character_ids) != 1:
        raise ConfigurationError("selected artifacts do not share one character ID")
    character_id = next(iter(character_ids))

    dataset_root = dataset_root.resolve()
    if dataset_root.exists():
        raise ConfigurationError(f"dataset already exists: {dataset_root}")
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    staging = dataset_root.parent / f".{dataset_root.name}-{uuid4().hex}.tmp"
    staging.mkdir()
    manifest_path = staging / "manifest.jsonl"
    created_at = utc_now()

    try:
        selections: tuple[tuple[Literal["train", "validation"], list[str]], ...] = (
            ("train", train_artifact_ids),
            ("validation", validation_artifact_ids),
        )
        for split, artifact_ids in selections:
            split_dir = staging / split
            split_dir.mkdir()
            for artifact_id in artifact_ids:
                source_record = records_by_id[artifact_id]
                source = _source_image(run_dir, source_record)
                image_relative = Path(split) / f"{artifact_id}.png"
                caption_relative = image_relative.with_suffix(".txt")
                image_destination = staging / image_relative
                caption_destination = staging / caption_relative
                shutil.copyfile(source, image_destination)
                caption = compile_training_caption(source_record, token)
                atomic_write_text(caption_destination, caption + "\n")
                item = DatasetRecord(
                    dataset_id=dataset_id,
                    item_id=f"{dataset_id}-{split}-{artifact_id}",
                    character_id=character_id,
                    split=split,
                    image_path=image_relative.as_posix(),
                    image_sha256=sha256_file(image_destination),
                    caption_path=caption_relative.as_posix(),
                    caption_sha256=sha256_file(caption_destination),
                    source_run_id=status.run_id,
                    source_artifact=source_record,
                    created_at=created_at,
                )
                append_jsonl(manifest_path, item.model_dump(mode="json"))
        atomic_write_yaml(
            staging / "dataset.yaml",
            {
                "schema_version": 1,
                "dataset_id": dataset_id,
                "character_id": character_id,
                "token": token,
                "train_count": len(train_artifact_ids),
                "validation_count": len(validation_artifact_ids),
                "source_run_id": status.run_id,
                "source_pass_id": source_pass_id,
                "manifest": "manifest.jsonl",
                "created_at": created_at,
            },
        )
        os.replace(staging, dataset_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return DatasetBuildSummary(
        dataset_root=dataset_root,
        manifest_path=dataset_root / "manifest.jsonl",
        train_count=len(train_artifact_ids),
        validation_count=len(validation_artifact_ids),
    )
