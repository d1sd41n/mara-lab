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
