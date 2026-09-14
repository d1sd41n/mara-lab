from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from mara_lab.artifacts import (
    append_jsonl,
    atomic_write_yaml,
    read_jsonl,
    sha256_file,
    utc_now,
)
from mara_lab.errors import ConfigurationError
from mara_lab.manifests import (
    ApprovedReferenceRecord,
    ReferenceRecord,
    ReferenceSetMemberRecord,
    RunStatus,
)


@dataclass(frozen=True, slots=True)
class ReferenceSetSummary:
    reference_set_dir: Path
    descriptor_path: Path
    manifest_path: Path
    image_paths: list[Path]


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read JSON document {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"JSON document {path} must contain an object")
    return value


def _read_reference_set_document(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot read reference set {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ConfigurationError(f"invalid reference set descriptor: {path}")
    return data


def _resolve_source(run_dir: Path, record: ReferenceRecord) -> Path:
    source = (run_dir / record.image_path).resolve()
    try:
        source.relative_to(run_dir)
    except ValueError as exc:
        raise ConfigurationError(f"artifact path escapes source run: {record.image_path}") from exc
    if not source.is_file() or sha256_file(source) != record.sha256:
        raise ConfigurationError(f"source image is missing or modified: {record.artifact_id}")
    return source


def approve_reference_set(
    run_dir: Path,
    artifact_ids: list[str],
    *,
    characters_root: Path = Path("characters"),
    reference_set_id: str = "pass-a-v001",
    minimum: int = 2,
    maximum: int = 4,
) -> ReferenceSetSummary:
    if not minimum <= len(artifact_ids) <= maximum:
        raise ConfigurationError(
            f"select between {minimum} and {maximum} reference artifacts; got {len(artifact_ids)}"
        )
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ConfigurationError("approved reference artifacts must be unique")
    run_dir = run_dir.resolve()
    status = RunStatus.model_validate(_read_json_object(run_dir / "status.json"))
    if status.state != "completed":
        raise ConfigurationError(f"source run is not completed: {status.state}")
    source_records = [
        ReferenceRecord.model_validate(row)
        for row in read_jsonl(run_dir / "outputs" / "manifest.jsonl")
    ]
    records = {record.artifact_id: record for record in source_records}
    missing = [artifact_id for artifact_id in artifact_ids if artifact_id not in records]
    if missing:
        raise ConfigurationError("approved artifacts are missing: " + ", ".join(missing))
    selected = [records[artifact_id] for artifact_id in artifact_ids]
    if any(record.pass_id != "a" for record in selected):
        raise ConfigurationError("only Pass A artifacts may seed a Pass B reference set")
    character_ids = {record.character_id for record in selected}
    if len(character_ids) != 1:
        raise ConfigurationError("approved artifacts do not share one character ID")
    character_id = next(iter(character_ids))

    master_records = selected[0].source_references
    if len(master_records) != 1:
        raise ConfigurationError("Pass A must have exactly one canonical source reference")
    master_record = master_records[0]
    master_source = Path(master_record["path"]).resolve()
    if not master_source.is_file() or sha256_file(master_source) != master_record["sha256"]:
        raise ConfigurationError("canonical master image is missing or modified")
    if any(record.source_references != master_records for record in selected[1:]):
        raise ConfigurationError("approved artifacts do not share the same canonical master")

    reference_set_dir = (
        characters_root.resolve() / character_id / "reference-sets" / reference_set_id
    )
    if reference_set_dir.exists():
        raise ConfigurationError(f"reference set already exists: {reference_set_dir}")
    reference_set_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = reference_set_dir.parent / f".{reference_set_id}-{uuid4().hex}.tmp"
    staging.mkdir()
    manifest_path = staging / "manifest.jsonl"
    approved_at = utc_now()
    try:
        master_destination = staging / "master.png"
        shutil.copyfile(master_source, master_destination)
        image_records = [
            {
                "role": "master",
                "path": master_destination.name,
                "sha256": sha256_file(master_destination),
                "source_path": master_source.as_posix(),
            }
        ]
        for sequence, record in enumerate(selected, start=1):
            source = _resolve_source(run_dir, record)
            destination = staging / f"approved-{sequence:02d}-{record.artifact_id}.png"
            shutil.copyfile(source, destination)
            approved = ApprovedReferenceRecord(
                reference_set_id=reference_set_id,
                sequence=sequence,
                character_id=character_id,
                image_path=destination.name,
                sha256=sha256_file(destination),
                source_run_id=status.run_id,
                source_artifact=record,
                approved_at=approved_at,
            )
            append_jsonl(manifest_path, approved.model_dump(mode="json"))
            image_records.append(
                {
                    "role": "approved",
                    "path": destination.name,
                    "sha256": approved.sha256,
                    "source_artifact_id": record.artifact_id,
                }
            )
        atomic_write_yaml(
            staging / "reference-set.yaml",
            {
                "schema_version": 1,
                "reference_set_id": reference_set_id,
                "character_id": character_id,
                "source_run_id": status.run_id,
                "selection_method": "human",
                "approved_at": approved_at,
                "images": image_records,
                "manifest": manifest_path.name,
            },
        )
        os.replace(staging, reference_set_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    descriptor = reference_set_dir / "reference-set.yaml"
    return ReferenceSetSummary(
        reference_set_dir=reference_set_dir,
        descriptor_path=descriptor,
        manifest_path=reference_set_dir / manifest_path.name,
        image_paths=load_reference_set_images(descriptor, character_id=character_id),
    )


def _source_reference_signature(records: list[dict[str, str]]) -> list[tuple[Path, str]]:
    signature: list[tuple[Path, str]] = []
    for record in records:
        path = record.get("path")
        digest = record.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ConfigurationError("source reference record is incomplete")
        signature.append((Path(path).resolve(), digest))
    return signature


def finalize_reference_set(
    base_reference_set: Path,
    run_dir: Path,
    artifact_ids: list[str],
    *,
    characters_root: Path = Path("characters"),
    reference_set_id: str = "canonical-v001",
    expected_image_count: int = 7,
) -> ReferenceSetSummary:
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ConfigurationError("final reference artifacts must be unique")
    if not artifact_ids:
        raise ConfigurationError("select at least one Pass B reference artifact")

    run_dir = run_dir.resolve()
    status = RunStatus.model_validate(_read_json_object(run_dir / "status.json"))
    if status.state != "completed":
        raise ConfigurationError(f"source run is not completed: {status.state}")
    source_records = [
        ReferenceRecord.model_validate(row)
        for row in read_jsonl(run_dir / "outputs" / "manifest.jsonl")
    ]
    records = {record.artifact_id: record for record in source_records}
    if len(records) != len(source_records):
        raise ConfigurationError("source manifest contains duplicate artifact IDs")
    missing = [artifact_id for artifact_id in artifact_ids if artifact_id not in records]
    if missing:
        raise ConfigurationError("approved artifacts are missing: " + ", ".join(missing))
    selected = [records[artifact_id] for artifact_id in artifact_ids]
    if any(record.pass_id != "b" for record in selected):
        raise ConfigurationError("only Pass B artifacts may complete the canonical set")
    character_ids = {record.character_id for record in selected}
    if len(character_ids) != 1:
        raise ConfigurationError("approved artifacts do not share one character ID")
    character_id = next(iter(character_ids))

    base_reference_set = base_reference_set.resolve()
    base_document = _read_reference_set_document(base_reference_set)
    if base_document.get("character_id") != character_id:
        raise ConfigurationError("base reference set belongs to another character")
    base_images = load_reference_set_images(base_reference_set, character_id=character_id)
    if len(base_images) + len(selected) != expected_image_count:
        raise ConfigurationError(
            f"final reference set needs {expected_image_count} images; "
            f"base has {len(base_images)} and selection has {len(selected)}"
        )
    expected_sources = [(image.resolve(), sha256_file(image)) for image in base_images]
    for record in selected:
        if _source_reference_signature(record.source_references) != expected_sources:
            raise ConfigurationError(
                f"{record.artifact_id} was not generated from the supplied base reference set"
            )

    reference_set_dir = (
        characters_root.resolve() / character_id / "reference-sets" / reference_set_id
    )
    if reference_set_dir.exists():
        raise ConfigurationError(f"reference set already exists: {reference_set_dir}")
    reference_set_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = reference_set_dir.parent / f".{reference_set_id}-{uuid4().hex}.tmp"
    staging.mkdir()
    manifest_path = staging / "manifest.jsonl"
    approved_at = utc_now()
    image_records: list[dict[str, Any]] = []
    try:
        base_descriptor_hash = sha256_file(base_reference_set)
        for sequence, source in enumerate(base_images, start=1):
            role = "master" if sequence == 1 else "approved"
            stage = "canonical" if sequence == 1 else "pass-a"
            filename = "master.png" if sequence == 1 else f"approved-pass-a-{sequence - 1:02d}.png"
            destination = staging / filename
            shutil.copyfile(source, destination)
            member = ReferenceSetMemberRecord(
                reference_set_id=reference_set_id,
                sequence=sequence,
                character_id=character_id,
                role=role,
                stage=stage,
                image_path=filename,
                sha256=sha256_file(destination),
                source={
                    "kind": "reference-set",
                    "descriptor_path": base_reference_set.as_posix(),
                    "descriptor_sha256": base_descriptor_hash,
                    "member_path": source.as_posix(),
                    "member_sha256": sha256_file(source),
                },
                approved_at=approved_at,
            )
            append_jsonl(manifest_path, member.model_dump(mode="json"))
            image_records.append(
                {
                    "role": role,
                    "stage": stage,
                    "path": filename,
                    "sha256": member.sha256,
                }
            )

        for offset, record in enumerate(selected, start=1):
            sequence = len(base_images) + offset
            source = _resolve_source(run_dir, record)
            filename = f"approved-pass-b-{offset:02d}-{record.artifact_id}.png"
            destination = staging / filename
            shutil.copyfile(source, destination)
            member = ReferenceSetMemberRecord(
                reference_set_id=reference_set_id,
                sequence=sequence,
                character_id=character_id,
                role="approved",
                stage="pass-b",
                image_path=filename,
                sha256=sha256_file(destination),
                source={
                    "kind": "reference-candidate",
                    "run_id": status.run_id,
                    "artifact": record.model_dump(mode="json"),
                },
                approved_at=approved_at,
            )
            append_jsonl(manifest_path, member.model_dump(mode="json"))
            image_records.append(
                {
                    "role": "approved",
                    "stage": "pass-b",
                    "path": filename,
                    "sha256": member.sha256,
                    "source_artifact_id": record.artifact_id,
                }
            )

        atomic_write_yaml(
            staging / "reference-set.yaml",
            {
                "schema_version": 1,
                "reference_set_id": reference_set_id,
                "character_id": character_id,
                "source_reference_set": {
                    "path": base_reference_set.as_posix(),
                    "sha256": base_descriptor_hash,
                },
                "source_run_id": status.run_id,
                "selection_method": "human",
                "approved_at": approved_at,
                "images": image_records,
                "manifest": manifest_path.name,
            },
        )
        os.replace(staging, reference_set_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    descriptor = reference_set_dir / "reference-set.yaml"
    return ReferenceSetSummary(
        reference_set_dir=reference_set_dir,
        descriptor_path=descriptor,
        manifest_path=reference_set_dir / manifest_path.name,
        image_paths=load_reference_set_images(descriptor, character_id=character_id),
    )


def load_reference_set_images(path: Path, *, character_id: str) -> list[Path]:
    path = path.resolve()
    data = _read_reference_set_document(path)
    if data.get("character_id") != character_id:
        raise ConfigurationError(
            f"reference set belongs to {data.get('character_id')}, not {character_id}"
        )
    images = data.get("images")
    if not isinstance(images, list) or len(images) < 3:
        raise ConfigurationError("reference set must contain a master and at least two approvals")
    if not all(isinstance(record, dict) for record in images):
        raise ConfigurationError("reference set image record must be an object")
    if images[0].get("role") != "master" or any(
        record.get("role") == "master" for record in images[1:]
    ):
        raise ConfigurationError("reference set must contain exactly one leading master")
    resolved: list[Path] = []
    for record in images:
        relative = record.get("path")
        expected_hash = record.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ConfigurationError("reference set image record is incomplete")
        image = (path.parent / relative).resolve()
        try:
            image.relative_to(path.parent)
        except ValueError as exc:
            raise ConfigurationError(
                f"reference set path escapes its directory: {relative}"
            ) from exc
        if not image.is_file() or sha256_file(image) != expected_hash:
            raise ConfigurationError(f"reference set image is missing or modified: {image}")
        resolved.append(image)
    return resolved
