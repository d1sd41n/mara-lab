from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from mara_lab.artifacts import (
    append_jsonl,
    atomic_write_json,
    atomic_write_yaml,
    read_jsonl,
    sha256_file,
    utc_now,
)
from mara_lab.config import load_resolved_config
from mara_lab.errors import ConfigurationError
from mara_lab.manifests import (
    CandidateRecord,
    CanonicalPointers,
    CanonicalRecord,
    CanonicalSourceSnapshot,
    CharacterDefinition,
    RunStatus,
)


@dataclass(frozen=True, slots=True)
class CanonSelectionSummary:
    character_dir: Path
    manifest_path: Path
    master_path: Path
    backup_path: Path
    master_artifact_id: str
    backup_artifact_id: str


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read JSON document {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"JSON document {path} must contain an object")
    return value


def _resolve_run_artifact(run_dir: Path, image_path: str) -> Path:
    resolved_run = run_dir.resolve()
    resolved_image = (resolved_run / image_path).resolve()
    try:
        resolved_image.relative_to(resolved_run)
    except ValueError as exc:
        raise ConfigurationError(f"artifact path escapes the source run: {image_path}") from exc
    if not resolved_image.is_file():
        raise ConfigurationError(f"artifact image is missing: {resolved_image}")
    return resolved_image


def _select_record(
    records: dict[str, CandidateRecord], artifact_id: str, role: str
) -> CandidateRecord:
    record = records.get(artifact_id)
    if record is None:
        raise ConfigurationError(f"{role} artifact not found in manifest: {artifact_id}")
    return record


def select_canon(
    run_dir: Path,
    master_artifact_id: str,
    backup_artifact_id: str,
    *,
    canon_root: Path = Path("characters"),
    canonical_version: str = "v001",
) -> CanonSelectionSummary:
    if master_artifact_id == backup_artifact_id:
        raise ConfigurationError("master and backup artifacts must be different")
    if re.fullmatch(r"v[0-9]{3}", canonical_version) is None:
        raise ConfigurationError("canonical version must look like v001")

    resolved_run = run_dir.resolve()
    status = RunStatus.model_validate(_read_json_object(resolved_run / "status.json"))
    if status.state != "completed":
        raise ConfigurationError(f"source run is not completed: {status.state}")

    config_path = resolved_run / "resolved-config.yaml"
    environment_path = resolved_run / "environment.json"
    config = load_resolved_config(config_path)
    environment = _read_json_object(environment_path)
    environment.pop("python_executable", None)

    candidate_records = [
        CandidateRecord.model_validate(row)
        for row in read_jsonl(resolved_run / "outputs" / "manifest.jsonl")
    ]
    records_by_id = {record.artifact_id: record for record in candidate_records}
    if len(records_by_id) != len(candidate_records):
        raise ConfigurationError("source manifest contains duplicate artifact IDs")

    master = _select_record(records_by_id, master_artifact_id, "master")
    backup = _select_record(records_by_id, backup_artifact_id, "backup")
    if master.character_id != config.character_id or backup.character_id != config.character_id:
        raise ConfigurationError("selected artifacts do not belong to the configured character")

    master_source = _resolve_run_artifact(resolved_run, master.image_path)
    backup_source = _resolve_run_artifact(resolved_run, backup.image_path)
    for record, source in ((master, master_source), (backup, backup_source)):
        actual_sha256 = sha256_file(source)
        if actual_sha256 != record.sha256:
            raise ConfigurationError(
                f"source hash mismatch for {record.artifact_id}: {actual_sha256} != {record.sha256}"
            )

    character_dir = canon_root.resolve() / config.character_id
    if character_dir.exists():
        raise ConfigurationError(f"character canon already exists: {character_dir}")
    character_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = character_dir.parent / f".{config.character_id}-{uuid4().hex}.tmp"
    staging.mkdir()

    version_relative = Path("canonical") / canonical_version
    version_staging = staging / version_relative
    version_staging.mkdir(parents=True)
    master_staging = version_staging / "master.png"
    backup_staging = version_staging / "backup.png"
    manifest_staging = version_staging / "manifest.jsonl"
    config_staging = version_staging / "source-config.yaml"
    environment_staging = version_staging / "source-environment.json"

    try:
        shutil.copyfile(master_source, master_staging)
        shutil.copyfile(backup_source, backup_staging)
        shutil.copyfile(config_path, config_staging)
        atomic_write_json(environment_staging, environment)

        source_snapshot = CanonicalSourceSnapshot(
            config_path=(version_relative / config_staging.name).as_posix(),
            config_sha256=sha256_file(config_staging),
            environment_path=(version_relative / environment_staging.name).as_posix(),
            environment_sha256=sha256_file(environment_staging),
        )
        selected_at = utc_now()
        selections = (
            ("master", master, master_staging),
            ("backup", backup, backup_staging),
        )
        for role, source_record, destination in selections:
            canonical_record = CanonicalRecord(
                canonical_id=f"{config.character_id}-{canonical_version}-{role}",
                canonical_version=canonical_version,
                character_id=config.character_id,
                role=role,
                image_path=(version_relative / destination.name).as_posix(),
                sha256=sha256_file(destination),
                selected_at=selected_at,
                source_run_id=status.run_id,
                source_snapshot=source_snapshot,
                source_artifact=source_record,
            )
            append_jsonl(manifest_staging, canonical_record.model_dump(mode="json"))

        character = CharacterDefinition(
            character_id=config.character_id,
            display_name=config.character_id.capitalize(),
            canonical=CanonicalPointers(
                version=canonical_version,
                master=(version_relative / master_staging.name).as_posix(),
                backup=(version_relative / backup_staging.name).as_posix(),
                manifest=(version_relative / manifest_staging.name).as_posix(),
            ),
        )
        atomic_write_yaml(staging / "character.yaml", character.model_dump(mode="json"))
        os.replace(staging, character_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    final_version_dir = character_dir / version_relative
    return CanonSelectionSummary(
        character_dir=character_dir,
        manifest_path=final_version_dir / manifest_staging.name,
        master_path=final_version_dir / master_staging.name,
        backup_path=final_version_dir / backup_staging.name,
        master_artifact_id=master_artifact_id,
        backup_artifact_id=backup_artifact_id,
    )
