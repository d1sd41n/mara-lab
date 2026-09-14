from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mara_lab.errors import ConfigurationError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BackendName(StrEnum):
    REALVISXL = "realvisxl"
    FAKE = "fake"


class ReferenceBackendName(StrEnum):
    PHOTOMAKER_V2 = "photomaker_v2"
    FAKE = "fake"


class TrainerBackendName(StrEnum):
    SD_SCRIPTS_SDXL = "sd_scripts_sdxl"


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
    materialized_cache_dir: Path = Path(".cache/materialized")
    materialized_max_shard_mib: int = Field(default=512, ge=64, le=2048)
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


class ConditionerProfile(StrictModel):
    backend: ReferenceBackendName
    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    weight_file: str = Field(min_length=1)
    weight_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_repository: str = Field(min_length=1)
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    trigger_word: str = Field(default="img", pattern=r"^[a-z][a-z0-9_]*$")
    version: Literal["v2"] = "v2"
    insightface_model: str = "buffalo_l"
    insightface_provider: Literal["CPUExecutionProvider"] = "CPUExecutionProvider"
    cache_dir: Path = Path(".cache/huggingface")
    insightface_cache_dir: Path = Path(".cache/insightface")
    offload_cache_dir: Path = Path(".cache/offload")
    execution_mode: Literal["auto", "direct", "disk"] = "auto"


class ReferenceCell(StrictModel):
    cell_id: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    view: str = Field(min_length=3)
    expression: str = Field(min_length=3)
    lighting: str = Field(min_length=3)
    setting: str = Field(min_length=3)
    wardrobe: str = Field(min_length=3)
    framing: str = Field(min_length=3)
    capture_style: str = Field(min_length=3)
    camera_behavior: str = Field(min_length=3)


class ReferenceGenerationSettings(StrictModel):
    width: int = Field(gt=0, le=2048)
    height: int = Field(gt=0, le=2048)
    steps: int = Field(ge=1, le=200)
    guidance_scale: float = Field(ge=0, le=30)
    start_merge_step: int = Field(ge=0)
    seeds: SeedRange

    @field_validator("width", "height")
    @classmethod
    def validate_image_dimension(cls, value: int) -> int:
        if value % 8:
            raise ValueError("image dimensions must be divisible by 8")
        return value

    @model_validator(mode="after")
    def validate_merge_step(self) -> ReferenceGenerationSettings:
        if self.start_merge_step >= self.steps:
            raise ValueError("start_merge_step must be smaller than steps")
        return self


class ReferenceExperimentFile(StrictModel):
    schema_version: int = Field(default=1, ge=1, le=1)
    experiment_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    character_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    pass_id: Literal["a", "b", "c"]
    model_profile: Path
    compute_profile: Path
    conditioner_profile: Path
    reference_images: list[Path] = Field(min_length=1, max_length=8)
    minimum_reference_images: int = Field(default=1, ge=1, le=8)
    maximum_reference_images: int = Field(default=8, ge=1, le=8)
    prompt_template_version: Literal["v1", "v2"] = "v1"
    generation: ReferenceGenerationSettings
    cells: list[ReferenceCell] = Field(min_length=1, max_length=32)
    negative_prompt: str = Field(min_length=20)
    output: OutputSettings = OutputSettings()

    @model_validator(mode="after")
    def validate_unique_cells(self) -> ReferenceExperimentFile:
        if self.minimum_reference_images > self.maximum_reference_images:
            raise ValueError("minimum_reference_images cannot exceed maximum_reference_images")
        cell_ids = [cell.cell_id for cell in self.cells]
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("reference cell IDs must be unique")
        return self


class TrainerProfile(StrictModel):
    backend: TrainerBackendName
    repository: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_dir: Path = Path(".cache/upstream/sd-scripts")
    runtime_project: Path = Path("runtimes/sd-scripts")
    network_module: Literal["networks.lora"] = "networks.lora"
    network_dim: int = Field(default=16, ge=1, le=256)
    network_alpha: int = Field(default=16, ge=1, le=256)
    optimizer_type: Literal["AdamW8bit"] = "AdamW8bit"
    learning_rate: float = Field(default=1e-4, gt=0, le=1)
    lr_scheduler: Literal["constant_with_warmup"] = "constant_with_warmup"
    lr_warmup_steps: int = Field(default=100, ge=0)
    mixed_precision: Literal["bf16", "fp16"] = "bf16"
    full_bf16: bool = True
    resolution: int = Field(default=768, ge=256, le=2048)
    min_bucket_reso: int = Field(default=512, ge=256, le=2048)
    max_bucket_reso: int = Field(default=1024, ge=256, le=2048)
    bucket_reso_steps: int = Field(default=64, ge=8, le=256)
    train_batch_size: int = Field(default=1, ge=1, le=8)
    gradient_accumulation_steps: int = Field(default=1, ge=1, le=32)
    max_train_steps: int = Field(default=200, ge=1, le=100_000)
    save_every_n_steps: int = Field(default=200, ge=1, le=100_000)
    seed: int = Field(default=240311, ge=0, le=2**63 - 1)
    max_data_loader_workers: int = Field(default=0, ge=0, le=16)
    cache_latents_to_disk: bool = True
    cache_text_encoder_outputs_to_disk: bool = True
    gradient_checkpointing: bool = True
    sdpa: bool = True
    torch_compile: bool = False
    cpu_offload: bool = False

    @model_validator(mode="after")
    def validate_training_profile(self) -> TrainerProfile:
        if self.network_alpha > self.network_dim:
            raise ValueError("network_alpha cannot exceed network_dim")
        if self.min_bucket_reso > self.resolution:
            raise ValueError("min_bucket_reso cannot exceed resolution")
        if self.max_bucket_reso < self.resolution:
            raise ValueError("max_bucket_reso cannot be smaller than resolution")
        for value in (self.resolution, self.min_bucket_reso, self.max_bucket_reso):
            if value % self.bucket_reso_steps:
                raise ValueError("training resolutions must be divisible by bucket_reso_steps")
        if self.full_bf16 and self.mixed_precision != "bf16":
            raise ValueError("full_bf16 requires mixed_precision=bf16")
        if not self.cache_latents_to_disk or not self.cache_text_encoder_outputs_to_disk:
            raise ValueError("the 12 GiB SDXL profile requires disk-backed feature caches")
        if not self.gradient_checkpointing or not self.sdpa:
            raise ValueError("the 12 GiB SDXL profile requires checkpointing and SDPA")
        if self.torch_compile or self.cpu_offload:
            raise ValueError("v0.1 training disallows torch.compile and CPU/NVMe model offload")
        return self


class TrainingExperimentFile(StrictModel):
    schema_version: int = Field(default=1, ge=1, le=1)
    experiment_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    character_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    adapter_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    model_profile: Path
    compute_profile: Path
    trainer_profile: Path
    dataset_root: Path
    dataset_manifest: Path
    dataset_token: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    expected_train_images: int = Field(default=28, ge=1, le=10_000)
    expected_validation_images: int = Field(default=6, ge=0, le=10_000)
    output: OutputSettings = OutputSettings()


class BenchmarkGenerationSettings(StrictModel):
    width: int = Field(gt=0, le=2048)
    height: int = Field(gt=0, le=2048)
    steps: int = Field(ge=1, le=200)
    guidance_scale: float = Field(ge=0, le=30)

    @field_validator("width", "height")
    @classmethod
    def validate_image_dimension(cls, value: int) -> int:
        if value % 8:
            raise ValueError("image dimensions must be divisible by 8")
        return value


class BenchmarkCase(StrictModel):
    case_id: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    prompt: str = Field(min_length=20)
    seed: int = Field(ge=0, le=2**63 - 1)


class BenchmarkExperimentFile(StrictModel):
    schema_version: int = Field(default=1, ge=1, le=1)
    experiment_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    character_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    identity_token: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    model_profile: Path
    compute_profile: Path
    generation: BenchmarkGenerationSettings
    cases: list[BenchmarkCase] = Field(min_length=1, max_length=64)
    negative_prompt: str = Field(min_length=20)
    output: OutputSettings = OutputSettings()

    @model_validator(mode="after")
    def validate_cases(self) -> BenchmarkExperimentFile:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("benchmark case IDs must be unique")
        for case in self.cases:
            if case.prompt.count(self.identity_token) != 1:
                raise ValueError(
                    f"benchmark case {case.case_id} must contain identity token "
                    f"{self.identity_token} exactly once"
                )
            if "adult woman" not in case.prompt:
                raise ValueError(
                    f"benchmark case {case.case_id} must contain the adult class phrase"
                )
        return self


class ConfigSources(StrictModel):
    experiment: str
    model: str
    compute: str


class ReferenceConfigSources(StrictModel):
    experiment: str
    model: str
    compute: str
    conditioner: str


class TrainingConfigSources(StrictModel):
    experiment: str
    model: str
    compute: str
    trainer: str


class BenchmarkConfigSources(StrictModel):
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


class ResolvedReferenceExperimentConfig(StrictModel):
    schema_version: int = 1
    experiment_id: str
    character_id: str
    pass_id: Literal["a", "b", "c"]
    model: ModelProfile
    compute: ComputeProfile
    conditioner: ConditionerProfile
    reference_images: list[Path]
    minimum_reference_images: int = 1
    maximum_reference_images: int = 8
    prompt_template_version: Literal["v1", "v2"] = "v1"
    generation: ReferenceGenerationSettings
    cells: list[ReferenceCell]
    negative_prompt: str
    output: OutputSettings
    sources: ReferenceConfigSources
    cli_overrides: dict[str, Any] = Field(default_factory=dict)


class ResolvedTrainingExperimentConfig(StrictModel):
    schema_version: int = 1
    experiment_id: str
    character_id: str
    adapter_id: str
    model: ModelProfile
    compute: ComputeProfile
    trainer: TrainerProfile
    dataset_root: Path
    dataset_manifest: Path
    dataset_token: str
    expected_train_images: int
    expected_validation_images: int
    output: OutputSettings
    sources: TrainingConfigSources
    cli_overrides: dict[str, Any] = Field(default_factory=dict)


class ResolvedBenchmarkExperimentConfig(StrictModel):
    schema_version: int = 1
    experiment_id: str
    character_id: str
    identity_token: str
    model: ModelProfile
    compute: ComputeProfile
    generation: BenchmarkGenerationSettings
    cases: list[BenchmarkCase]
    negative_prompt: str
    output: OutputSettings
    sources: BenchmarkConfigSources
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


def load_model_profile(path: Path) -> ModelProfile:
    try:
        return ModelProfile.model_validate(_read_yaml(path.resolve()))
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"invalid model profile {path}: {exc}") from exc


def load_reference_experiment_config(path: Path) -> ResolvedReferenceExperimentConfig:
    experiment_path = path.resolve()
    try:
        experiment = ReferenceExperimentFile.model_validate(_read_yaml(experiment_path))
        model_path = (experiment_path.parent / experiment.model_profile).resolve()
        compute_path = (experiment_path.parent / experiment.compute_profile).resolve()
        conditioner_path = (experiment_path.parent / experiment.conditioner_profile).resolve()
        model = ModelProfile.model_validate(_read_yaml(model_path))
        compute = ComputeProfile.model_validate(_read_yaml(compute_path))
        conditioner = ConditionerProfile.model_validate(_read_yaml(conditioner_path))
        reference_images = [
            (experiment_path.parent / reference).resolve()
            for reference in experiment.reference_images
        ]
        missing = [str(reference) for reference in reference_images if not reference.is_file()]
        if missing:
            raise ConfigurationError("missing reference image(s): " + ", ".join(missing))
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"invalid reference experiment configuration: {exc}") from exc

    return ResolvedReferenceExperimentConfig(
        schema_version=experiment.schema_version,
        experiment_id=experiment.experiment_id,
        character_id=experiment.character_id,
        pass_id=experiment.pass_id,
        model=model,
        compute=compute,
        conditioner=conditioner,
        reference_images=reference_images,
        minimum_reference_images=experiment.minimum_reference_images,
        maximum_reference_images=experiment.maximum_reference_images,
        prompt_template_version=experiment.prompt_template_version,
        generation=experiment.generation,
        cells=experiment.cells,
        negative_prompt=experiment.negative_prompt,
        output=experiment.output,
        sources=ReferenceConfigSources(
            experiment=path.as_posix(),
            model=experiment.model_profile.as_posix(),
            compute=experiment.compute_profile.as_posix(),
            conditioner=experiment.conditioner_profile.as_posix(),
        ),
    )


def load_training_experiment_config(path: Path) -> ResolvedTrainingExperimentConfig:
    experiment_path = path.resolve()
    try:
        experiment = TrainingExperimentFile.model_validate(_read_yaml(experiment_path))
        model_path = (experiment_path.parent / experiment.model_profile).resolve()
        compute_path = (experiment_path.parent / experiment.compute_profile).resolve()
        trainer_path = (experiment_path.parent / experiment.trainer_profile).resolve()
        model = ModelProfile.model_validate(_read_yaml(model_path))
        compute = ComputeProfile.model_validate(_read_yaml(compute_path))
        trainer = TrainerProfile.model_validate(_read_yaml(trainer_path))
        trainer = trainer.model_copy(
            update={
                "source_dir": (trainer_path.parent / trainer.source_dir).resolve(),
                "runtime_project": (trainer_path.parent / trainer.runtime_project).resolve(),
            }
        )
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"invalid training experiment configuration: {exc}") from exc

    return ResolvedTrainingExperimentConfig(
        schema_version=experiment.schema_version,
        experiment_id=experiment.experiment_id,
        character_id=experiment.character_id,
        adapter_id=experiment.adapter_id,
        model=model,
        compute=compute,
        trainer=trainer,
        dataset_root=(experiment_path.parent / experiment.dataset_root).resolve(),
        dataset_manifest=(experiment_path.parent / experiment.dataset_manifest).resolve(),
        dataset_token=experiment.dataset_token,
        expected_train_images=experiment.expected_train_images,
        expected_validation_images=experiment.expected_validation_images,
        output=experiment.output,
        sources=TrainingConfigSources(
            experiment=path.as_posix(),
            model=experiment.model_profile.as_posix(),
            compute=experiment.compute_profile.as_posix(),
            trainer=experiment.trainer_profile.as_posix(),
        ),
    )


def load_benchmark_experiment_config(path: Path) -> ResolvedBenchmarkExperimentConfig:
    experiment_path = path.resolve()
    try:
        experiment = BenchmarkExperimentFile.model_validate(_read_yaml(experiment_path))
        model_path = (experiment_path.parent / experiment.model_profile).resolve()
        compute_path = (experiment_path.parent / experiment.compute_profile).resolve()
        model = ModelProfile.model_validate(_read_yaml(model_path))
        compute = ComputeProfile.model_validate(_read_yaml(compute_path))
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"invalid benchmark configuration: {exc}") from exc

    return ResolvedBenchmarkExperimentConfig(
        schema_version=experiment.schema_version,
        experiment_id=experiment.experiment_id,
        character_id=experiment.character_id,
        identity_token=experiment.identity_token,
        model=model,
        compute=compute,
        generation=experiment.generation,
        cases=experiment.cases,
        negative_prompt=experiment.negative_prompt,
        output=experiment.output,
        sources=BenchmarkConfigSources(
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


def load_resolved_reference_config(path: Path) -> ResolvedReferenceExperimentConfig:
    try:
        return ResolvedReferenceExperimentConfig.model_validate(_read_yaml(path))
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"invalid resolved reference configuration {path}: {exc}") from exc
