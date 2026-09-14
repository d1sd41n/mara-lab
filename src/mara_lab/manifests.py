from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from mara_lab.config import StrictModel


class CandidateRecord(StrictModel):
    schema_version: int = 1
    artifact_id: str
    kind: Literal["candidate"] = "candidate"
    sequence: int = Field(ge=1)
    character_id: str
    image_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    width: int
    height: int
    prompt: str
    negative_prompt: str
    model: dict[str, Any]
    generation: dict[str, Any]
    metrics: dict[str, Any]
    review_status: Literal["pending", "accepted", "rejected"] = "pending"
    created_at: str


class RunStatus(StrictModel):
    schema_version: int = 1
    state: Literal["running", "completed", "failed"]
    experiment_id: str
    run_id: str
    generated_count: int = 0
    started_at: str
    finished_at: str | None = None
    error: str | None = None


class CanonicalSourceSnapshot(StrictModel):
    config_path: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_path: str
    environment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CanonicalRecord(StrictModel):
    schema_version: int = 1
    canonical_id: str
    canonical_version: str = Field(pattern=r"^v[0-9]{3}$")
    character_id: str
    role: Literal["master", "backup"]
    image_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_at: str
    selection_method: Literal["human"] = "human"
    source_run_id: str
    source_snapshot: CanonicalSourceSnapshot
    source_artifact: CandidateRecord


class CharacterPolicy(StrictModel):
    fictional: bool = True
    adult: bool = True
    policy_version: str = "no_nudity_v001"


class CanonicalPointers(StrictModel):
    version: str = Field(pattern=r"^v[0-9]{3}$")
    master: str
    backup: str
    manifest: str


class CharacterDefinition(StrictModel):
    schema_version: int = 1
    character_id: str
    display_name: str
    policy: CharacterPolicy = Field(default_factory=CharacterPolicy)
    canonical: CanonicalPointers
