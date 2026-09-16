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


def _verified_dataset_file(
    dataset_root: Path,
    relative_path: str,
    expected_sha256: str,
    label: str,
) -> Path:
    path = (dataset_root / relative_path).resolve()
    try:
        path.relative_to(dataset_root)
    except ValueError as exc:
        raise ConfigurationError(f"{label} path escapes source dataset: {relative_path}") from exc
    if not path.is_file():
        raise ConfigurationError(f"{label} is missing: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ConfigurationError(
            f"{label} hash mismatch: {actual_sha256} != {expected_sha256}"
        )
    return path


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


def revise_character_dataset(
    base_dataset_root: Path,
    run_dir: Path,
    *,
    drop_train_artifact_ids: list[str],
    drop_validation_artifact_ids: list[str],
    add_train_artifact_ids: list[str],
    add_validation_artifact_ids: list[str],
    dataset_root: Path,
    dataset_id: str,
    token: str,
    expected_train_count: int = 28,
    expected_validation_count: int = 6,
    source_pass_id: Literal["a", "b", "c"] = "c",
) -> DatasetBuildSummary:
    """Create an immutable dataset revision from a base plus selected replacements."""
    selections = {
        "drop train": drop_train_artifact_ids,
        "drop validation": drop_validation_artifact_ids,
        "add train": add_train_artifact_ids,
        "add validation": add_validation_artifact_ids,
    }
    for label, artifact_ids in selections.items():
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ConfigurationError(f"{label} selection contains duplicate artifact IDs")
    dropped_ids = [*drop_train_artifact_ids, *drop_validation_artifact_ids]
    added_ids = [*add_train_artifact_ids, *add_validation_artifact_ids]
    if len(dropped_ids) != len(set(dropped_ids)):
        raise ConfigurationError("train and validation drop selections overlap")
    if len(added_ids) != len(set(added_ids)):
        raise ConfigurationError("train and validation addition selections overlap")

    base_dataset_root = base_dataset_root.resolve()
    base_manifest_path = base_dataset_root / "manifest.jsonl"
    if not base_manifest_path.is_file():
        raise ConfigurationError(f"base dataset manifest is missing: {base_manifest_path}")
    base_records = [
        DatasetRecord.model_validate(row) for row in read_jsonl(base_manifest_path)
    ]
    if not base_records:
        raise ConfigurationError("base dataset manifest is empty")
    base_character_ids = {record.character_id for record in base_records}
    if len(base_character_ids) != 1:
        raise ConfigurationError("base dataset contains multiple character IDs")
    character_id = next(iter(base_character_ids))

    base_by_split: dict[str, dict[str, DatasetRecord]] = {
        "train": {},
        "validation": {},
    }
    for record in base_records:
        artifact_id = record.source_artifact.artifact_id
        split_records = base_by_split[record.split]
        if artifact_id in split_records:
            raise ConfigurationError(
                f"base dataset contains duplicate source artifact ID: {artifact_id}"
            )
        split_records[artifact_id] = record
        _verified_dataset_file(
            base_dataset_root,
            record.image_path,
            record.image_sha256,
            "base dataset image",
        )
        _verified_dataset_file(
            base_dataset_root,
            record.caption_path,
            record.caption_sha256,
            "base dataset caption",
        )

    missing_drops = [
        artifact_id
        for split, artifact_ids in (
            ("train", drop_train_artifact_ids),
            ("validation", drop_validation_artifact_ids),
        )
        for artifact_id in artifact_ids
        if artifact_id not in base_by_split[split]
    ]
    if missing_drops:
        raise ConfigurationError(
            "drop selections are missing from the requested base split: "
            + ", ".join(missing_drops)
        )

    run_dir = run_dir.resolve()
    status = RunStatus.model_validate(_read_json_object(run_dir / "status.json"))
    if status.state != "completed":
        raise ConfigurationError(f"addition source run is not completed: {status.state}")
    source_records = [
        ReferenceRecord.model_validate(row)
        for row in read_jsonl(run_dir / "outputs" / "manifest.jsonl")
    ]
    additions_by_id = {record.artifact_id: record for record in source_records}
    if len(additions_by_id) != len(source_records):
        raise ConfigurationError("addition source manifest contains duplicate artifact IDs")
    missing_additions = [
        artifact_id for artifact_id in added_ids if artifact_id not in additions_by_id
    ]
    if missing_additions:
        raise ConfigurationError(
            "addition selections are missing: " + ", ".join(missing_additions)
        )
    selected_additions = [additions_by_id[artifact_id] for artifact_id in added_ids]
    if any(record.pass_id != source_pass_id for record in selected_additions):
        raise ConfigurationError(
            f"dataset additions must come from reference pass {source_pass_id}"
        )
    if any(record.character_id != character_id for record in selected_additions):
        raise ConfigurationError("dataset additions belong to a different character")

    drop_sets = {
        "train": set(drop_train_artifact_ids),
        "validation": set(drop_validation_artifact_ids),
    }
    kept_by_split = {
        split: [
            record
            for record in base_records
            if record.split == split
            and record.source_artifact.artifact_id not in drop_sets[split]
        ]
        for split in ("train", "validation")
    }
    additions_by_split = {
        "train": [additions_by_id[artifact_id] for artifact_id in add_train_artifact_ids],
        "validation": [
            additions_by_id[artifact_id] for artifact_id in add_validation_artifact_ids
        ],
    }
    final_counts = {
        split: len(kept_by_split[split]) + len(additions_by_split[split])
        for split in ("train", "validation")
    }
    if final_counts["train"] != expected_train_count:
        raise ConfigurationError(
            f"revised training selection has {final_counts['train']} images; "
            f"expected {expected_train_count}"
        )
    if final_counts["validation"] != expected_validation_count:
        raise ConfigurationError(
            f"revised validation selection has {final_counts['validation']} images; "
            f"expected {expected_validation_count}"
        )

    final_artifact_ids = [
        *(
            record.source_artifact.artifact_id
            for split in ("train", "validation")
            for record in kept_by_split[split]
        ),
        *added_ids,
    ]
    if len(final_artifact_ids) != len(set(final_artifact_ids)):
        raise ConfigurationError("revised dataset would contain duplicate artifact IDs")

    dataset_root = dataset_root.resolve()
    if dataset_root.exists():
        raise ConfigurationError(f"dataset already exists: {dataset_root}")
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    staging = dataset_root.parent / f".{dataset_root.name}-{uuid4().hex}.tmp"
    staging.mkdir()
    manifest_path = staging / "manifest.jsonl"
    created_at = utc_now()

    try:
        for split in ("train", "validation"):
            split_dir = staging / split
            split_dir.mkdir()
            for base_record in kept_by_split[split]:
                source_artifact = base_record.source_artifact
                artifact_id = source_artifact.artifact_id
                source = _verified_dataset_file(
                    base_dataset_root,
                    base_record.image_path,
                    base_record.image_sha256,
                    "base dataset image",
                )
                image_relative = Path(split) / f"{artifact_id}.png"
                caption_relative = image_relative.with_suffix(".txt")
                image_destination = staging / image_relative
                caption_destination = staging / caption_relative
                shutil.copyfile(source, image_destination)
                caption = compile_training_caption(source_artifact, token)
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
                    source_run_id=base_record.source_run_id,
                    source_artifact=source_artifact,
                    created_at=created_at,
                )
                append_jsonl(manifest_path, item.model_dump(mode="json"))

            for source_artifact in additions_by_split[split]:
                artifact_id = source_artifact.artifact_id
                source = _source_image(run_dir, source_artifact)
                image_relative = Path(split) / f"{artifact_id}.png"
                caption_relative = image_relative.with_suffix(".txt")
                image_destination = staging / image_relative
                caption_destination = staging / caption_relative
                shutil.copyfile(source, image_destination)
                caption = compile_training_caption(source_artifact, token)
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
                    source_artifact=source_artifact,
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
                "train_count": final_counts["train"],
                "validation_count": final_counts["validation"],
                "base_dataset": base_dataset_root.as_posix(),
                "base_manifest_sha256": sha256_file(base_manifest_path),
                "addition_source_run_id": status.run_id,
                "source_pass_id": source_pass_id,
                "revision": {
                    "drop_train": drop_train_artifact_ids,
                    "drop_validation": drop_validation_artifact_ids,
                    "add_train": add_train_artifact_ids,
                    "add_validation": add_validation_artifact_ids,
                },
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
        train_count=final_counts["train"],
        validation_count=final_counts["validation"],
    )
