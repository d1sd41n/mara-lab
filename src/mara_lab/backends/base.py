from __future__ import annotations

from abc import ABC, abstractmethod

from mara_lab.config import BackendName, ComputeProfile, ModelProfile
from mara_lab.domain import BackendInfo, GenerationRequest, GenerationResult
from mara_lab.errors import BackendUnavailableError


class ImageBackend(ABC):
    @abstractmethod
    def prepare(self) -> BackendInfo:
        """Load the backend and return its pinned runtime identity."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate one image and return measured runtime data."""

    @abstractmethod
    def close(self) -> None:
        """Release backend resources."""


def create_backend(model: ModelProfile, compute: ComputeProfile) -> ImageBackend:
    if model.backend is BackendName.FAKE:
        from mara_lab.backends.fake import FakeImageBackend

        return FakeImageBackend(model, compute)
    if model.backend is BackendName.REALVISXL:
        from mara_lab.backends.realvisxl import RealVisXLBackend

        return RealVisXLBackend(model, compute)
    raise BackendUnavailableError(f"unsupported backend: {model.backend}")
