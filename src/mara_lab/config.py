from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from mara_lab.errors import ConfigurationError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BackendName(StrEnum):
    REALVISXL = "realvisxl"
    FAKE = "fake"


class ModelProfile(StrictModel):
    backend: BackendName
    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    resolved_revision: str | None = None
    weight_file: str | None = None
    weight_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    dtype: str = "bfloat16"
    scheduler: str = "dpmpp_sde_karras"
    cache_dir: Path = Path(".cache/huggingface")
    use_safetensors: bool = True
    vae_tiling: bool = True
    trust_remote_code: bool = False

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, value: str) -> str:
        if value not in {"bfloat16", "float16"}:
            raise ValueError("dtype must be bfloat16 or float16")
        return value

    @field_validator("scheduler")
    @classmethod
    def validate_scheduler(cls, value: str) -> str:
        if value != "dpmpp_sde_karras":
            raise ValueError("v0.1 only supports dpmpp_sde_karras")
        return value

    @field_validator("use_safetensors")
    @classmethod
    def validate_safetensors(cls, value: bool) -> bool:
        if not value:
            raise ValueError("v0.1 requires safetensors model weights")
        return value

    @field_validator("trust_remote_code")
    @classmethod
    def validate_remote_code(cls, value: bool) -> bool:
        if value:
            raise ValueError("v0.1 does not execute remote model code")
        return value


class ComputeProfile(StrictModel):
    device: str = "cuda"
    require_cuda: bool = True
    max_reserved_vram_gib: float = Field(default=10.5, gt=0, le=12)
    allow_cpu_offload: bool = False
    stage_text_encoders: bool = True
    attention_backend: str = "sdpa"
    sequential_jobs: int = Field(default=1, ge=1, le=1)

    @field_validator("attention_backend")
    @classmethod
    def validate_attention_backend(cls, value: str) -> str:
        if value != "sdpa":
            raise ValueError("v0.1 only supports the PyTorch SDPA attention backend")
        return value


class SeedRange(StrictModel):
    start: int = Field(ge=0, le=2**63 - 1)
    count: int = Field(gt=0, le=10_000)

    def values(self, limit: int | None = None) -> list[int]:
        count = self.count if limit is None else min(limit, self.count)
        if count < 1:
            raise ConfigurationError("seed limit must be at least 1")
        return list(range(self.start, self.start + count))


class GenerationSettings(StrictModel):
    width: int = Field(gt=0, le=2048)
    height: int = Field(gt=0, le=2048)
    steps: int = Field(ge=1, le=200)
    guidance_scale: float = Field(ge=0, le=30)
    seeds: SeedRange

    @field_validator("width", "height")
    @classmethod
    def validate_image_dimension(cls, value: int) -> int:
        if value % 8:
            raise ValueError("image dimensions must be divisible by 8")
        return value


class PromptSettings(StrictModel):
    positive_template: str = Field(min_length=20)
    identity_variants: list[str] = Field(min_length=1)
    negative: str = Field(min_length=20)

    @field_validator("positive_template")
    @classmethod
    def validate_positive_template(cls, value: str) -> str:
        if value.count("{identity}") != 1:
            raise ValueError("positive_template must contain {identity} exactly once")
        return value

    def render_positive(self, sequence: int) -> tuple[str, int]:
        variant_index = (sequence - 1) % len(self.identity_variants)
        identity = self.identity_variants[variant_index]
        return self.positive_template.format(identity=identity), variant_index + 1


class OutputSettings(StrictModel):
    root: Path = Path("experiments")
    image_format: str = "png"
    contact_sheet_columns: int = Field(default=8, ge=1, le=12)
    contact_sheet_thumbnail_width: int = Field(default=180, ge=96, le=512)

    @field_validator("image_format")
    @classmethod
    def validate_image_format(cls, value: str) -> str:
        if value.lower() != "png":
            raise ValueError("v0.1 stores lossless PNG outputs only")
        return value.lower()


class ExperimentFile(StrictModel):
    schema_version: int = Field(default=1, ge=1, le=1)
    experiment_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    character_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    model_profile: Path
    compute_profile: Path
    generation: GenerationSettings
    prompts: PromptSettings
    output: OutputSettings = OutputSettings()


class ConfigSources(StrictModel):
    experiment: str
    model: str
    compute: str


class ResolvedExperimentConfig(StrictModel):
    schema_version: int = 1
    experiment_id: str
    character_id: str
    model: ModelProfile
    compute: ComputeProfile
    generation: GenerationSettings
    prompts: PromptSettings
    output: OutputSettings
    sources: ConfigSources
    cli_overrides: dict[str, Any] = Field(default_factory=dict)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"cannot read configuration {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"configuration {path} must contain a mapping")
    return data


def load_experiment_config(path: Path) -> ResolvedExperimentConfig:
    experiment_path = path.resolve()
    try:
        experiment = ExperimentFile.model_validate(_read_yaml(experiment_path))
        model_path = (experiment_path.parent / experiment.model_profile).resolve()
        compute_path = (experiment_path.parent / experiment.compute_profile).resolve()
        model = ModelProfile.model_validate(_read_yaml(model_path))
        compute = ComputeProfile.model_validate(_read_yaml(compute_path))
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"invalid experiment configuration: {exc}") from exc

    return ResolvedExperimentConfig(
        schema_version=experiment.schema_version,
        experiment_id=experiment.experiment_id,
        character_id=experiment.character_id,
        model=model,
        compute=compute,
        generation=experiment.generation,
        prompts=experiment.prompts,
        output=experiment.output,
        sources=ConfigSources(
            experiment=path.as_posix(),
            model=experiment.model_profile.as_posix(),
            compute=experiment.compute_profile.as_posix(),
        ),
    )


def load_resolved_config(path: Path) -> ResolvedExperimentConfig:
    try:
        return ResolvedExperimentConfig.model_validate(_read_yaml(path))
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"invalid resolved configuration {path}: {exc}") from exc
