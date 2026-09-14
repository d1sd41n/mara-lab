from __future__ import annotations

from abc import ABC, abstractmethod

from mara_lab.config import ResolvedReferenceExperimentConfig
from mara_lab.domain import (
    BackendInfo,
    CompiledPrompt,
    GenerationResult,
    ReferenceGenerationRequest,
)
from mara_lab.errors import BackendUnavailableError


class ReferenceImageBackend(ABC):
    @abstractmethod
    def prepare(self) -> BackendInfo:
        """Load a reference-conditioned backend."""

    @abstractmethod
    def compile_prompt(self, request: ReferenceGenerationRequest) -> CompiledPrompt:
        """Compile typed shot data into backend-specific prompt text."""

    @abstractmethod
    def generate(self, request: ReferenceGenerationRequest) -> GenerationResult:
        """Generate one conditioned image."""

    @abstractmethod
    def close(self) -> None:
        """Release backend resources."""


def create_reference_backend(
    config: ResolvedReferenceExperimentConfig,
) -> ReferenceImageBackend:
    if config.conditioner.backend.value == "photomaker_v2":
        from mara_lab.backends.photomaker_v2 import PhotoMakerV2Backend

        return PhotoMakerV2Backend(config)
    if config.conditioner.backend.value == "fake":
        from mara_lab.backends.reference_fake import FakeReferenceBackend

        return FakeReferenceBackend(config)
    raise BackendUnavailableError(f"unsupported reference backend: {config.conditioner.backend}")
