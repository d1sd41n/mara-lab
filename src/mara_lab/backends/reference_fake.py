from __future__ import annotations

from mara_lab.backends.fake import FakeImageBackend
from mara_lab.backends.reference import ReferenceImageBackend
from mara_lab.config import ResolvedReferenceExperimentConfig
from mara_lab.domain import (
    BackendInfo,
    CompiledPrompt,
    GenerationRequest,
    GenerationResult,
    ReferenceGenerationRequest,
)


class FakeReferenceBackend(ReferenceImageBackend):
    def __init__(self, config: ResolvedReferenceExperimentConfig) -> None:
        self._config = config
        self._backend = FakeImageBackend(config.model, config.compute)

    def prepare(self) -> BackendInfo:
        info = self._backend.prepare()
        return BackendInfo(
            backend="fake-reference",
            model_id=info.model_id,
            requested_revision=info.requested_revision,
            resolved_revision=info.resolved_revision,
            device=info.device,
            dtype=info.dtype,
            resident_allocated_vram_gib=0,
            resident_reserved_vram_gib=0,
            details={"purpose": "reference-workflow-test"},
        )

    def compile_prompt(self, request: ReferenceGenerationRequest) -> CompiledPrompt:
        shot = request.shot
        return CompiledPrompt(
            positive=(
                f"fake photo of an adult woman, {shot.view}, {shot.expression}, "
                f"{shot.lighting}, {shot.setting}"
            ),
            negative=self._config.negative_prompt,
        )

    def generate(self, request: ReferenceGenerationRequest) -> GenerationResult:
        prompt = self.compile_prompt(request)
        return self._backend.generate(
            GenerationRequest(
                prompt=prompt.positive,
                negative_prompt=prompt.negative,
                seed=request.seed,
                width=request.width,
                height=request.height,
                steps=request.steps,
                guidance_scale=request.guidance_scale,
            )
        )

    def close(self) -> None:
        self._backend.close()
