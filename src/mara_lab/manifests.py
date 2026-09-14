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


class ReferenceRecord(StrictModel):
    schema_version: int = 1
    artifact_id: str
    kind: Literal["reference-candidate"] = "reference-candidate"
    sequence: int = Field(ge=1)
    character_id: str
    pass_id: Literal["a", "b", "c"]
    cell_id: str
    image_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    width: int
    height: int
    shot: dict[str, Any]
    prompt: str
    negative_prompt: str
    source_references: list[dict[str, str]]
    model: dict[str, Any]
    conditioner: dict[str, Any]
    generation: dict[str, Any]
    metrics: dict[str, Any]
    review_status: Literal["pending", "accepted", "rejected"] = "pending"
    created_at: str


class DatasetRecord(StrictModel):
    schema_version: int = 1
    dataset_id: str
    item_id: str
    character_id: str
    split: Literal["train", "validation"]
    image_path: str
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    caption_path: str
    caption_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_run_id: str
    source_artifact: ReferenceRecord
    created_at: str


class AdapterRecord(StrictModel):
    schema_version: int = 1
    adapter_id: str
    character_id: str
    adapter_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_model_id: str
    base_revision: str
    base_weight_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trainer_repository: str
    trainer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    train_steps: int = Field(ge=1)
    network_dim: int = Field(ge=1)
    network_alpha: int = Field(ge=1)
    tensor_count: int = Field(ge=1)
    up_tensor_count: int = Field(ge=1)
    nonzero_up_tensor_count: int = Field(ge=1)
    max_abs_weight: float = Field(gt=0)
    max_reserved_vram_gib: float = Field(ge=0)
    created_at: str


class BenchmarkRecord(StrictModel):
    schema_version: int = 1
    artifact_id: str
    kind: Literal["benchmark"] = "benchmark"
    sequence: int = Field(ge=1)
    case_id: str
    character_id: str
    image_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    width: int
    height: int
    prompt: str
    negative_prompt: str
    adapter: AdapterRecord
    adapter_weight: float = Field(gt=0, le=2)
    model: dict[str, Any]
    generation: dict[str, Any]
    metrics: dict[str, Any]
    created_at: str


class ApprovedReferenceRecord(StrictModel):
    schema_version: int = 1
    reference_set_id: str
    sequence: int = Field(ge=1)
    character_id: str
    image_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_run_id: str
    source_artifact: ReferenceRecord
    selection_method: Literal["human"] = "human"
    approved_at: str


class ReferenceSetMemberRecord(StrictModel):
    schema_version: int = 1
    kind: Literal["reference-set-member"] = "reference-set-member"
    reference_set_id: str
    sequence: int = Field(ge=1)
    character_id: str
    role: Literal["master", "approved"]
    stage: Literal["canonical", "pass-a", "pass-b"]
    image_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: dict[str, Any]
    selection_method: Literal["human"] = "human"
    approved_at: str


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
