from __future__ import annotations

import random
import time

from PIL import Image, ImageDraw

from mara_lab.backends.base import ImageBackend
from mara_lab.config import ComputeProfile, ModelProfile
from mara_lab.domain import BackendInfo, GenerationRequest, GenerationResult, RuntimeMetrics


class FakeImageBackend(ImageBackend):
    """Deterministic backend used to test artifact plumbing without CUDA."""

    def __init__(self, model: ModelProfile, compute: ComputeProfile) -> None:
        self._model = model
        self._compute = compute

    def prepare(self) -> BackendInfo:
        return BackendInfo(
            backend="fake",
            model_id=self._model.model_id,
            requested_revision=self._model.revision,
            resolved_revision=self._model.resolved_revision or "fake-v1",
            device="cpu",
            dtype="uint8",
            resident_allocated_vram_gib=0.0,
            resident_reserved_vram_gib=0.0,
            details={"purpose": "workflow-test"},
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        started = time.perf_counter()
        randomizer = random.Random(request.seed)
        background = tuple(randomizer.randint(35, 220) for _ in range(3))
        image = Image.new("RGB", (request.width, request.height), background)
        draw = ImageDraw.Draw(image)

        for _ in range(12):
            x0 = randomizer.randint(0, max(0, request.width - 1))
            y0 = randomizer.randint(0, max(0, request.height - 1))
            x1 = min(request.width, x0 + randomizer.randint(16, max(17, request.width // 3)))
            y1 = min(request.height, y0 + randomizer.randint(16, max(17, request.height // 3)))
            color = tuple(randomizer.randint(0, 255) for _ in range(3))
            draw.rectangle((x0, y0, x1, y1), fill=color)

        draw.text((16, 16), f"seed {request.seed}", fill=(255, 255, 255))
        wall_seconds = time.perf_counter() - started
        return GenerationResult(
            image=image,
            metrics=RuntimeMetrics(
                wall_seconds=wall_seconds,
                max_allocated_vram_gib=0.0,
                max_reserved_vram_gib=0.0,
            ),
        )

    def close(self) -> None:
        return None
