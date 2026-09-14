from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from PIL import Image


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    prompt: str
    negative_prompt: str
    seed: int
    width: int
    height: int
    steps: int
    guidance_scale: float


@dataclass(frozen=True, slots=True)
class RuntimeMetrics:
    wall_seconds: float
    max_allocated_vram_gib: float
    max_reserved_vram_gib: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BackendInfo:
    backend: str
    model_id: str
    requested_revision: str
    resolved_revision: str
    device: str
    dtype: str
    resident_allocated_vram_gib: float
    resident_reserved_vram_gib: float
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GenerationResult:
    image: Image.Image
    metrics: RuntimeMetrics


@dataclass(frozen=True, slots=True)
class ReferenceShot:
    cell_id: str
    view: str
    expression: str
    lighting: str
    setting: str
    wardrobe: str
    framing: str
    capture_style: str
    camera_behavior: str


@dataclass(frozen=True, slots=True)
class ReferenceGenerationRequest:
    shot: ReferenceShot
    seed: int
    width: int
    height: int
    steps: int
    guidance_scale: float
    start_merge_step: int


@dataclass(frozen=True, slots=True)
class CompiledPrompt:
    positive: str
    negative: str
