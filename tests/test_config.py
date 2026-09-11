from pathlib import Path

import pytest
from pydantic import ValidationError

from mara_lab.config import (
    ComputeProfile,
    GenerationSettings,
    ModelProfile,
    SeedRange,
    load_experiment_config,
)

ROOT = Path(__file__).parents[1]


def test_loads_versioned_candidate_experiment() -> None:
    config = load_experiment_config(ROOT / "configs" / "experiments" / "mara-candidates-v001.yaml")

    assert config.character_id == "mara"
    assert config.model.model_id == "SG161222/RealVisXL_V5.0"
    assert config.model.revision == "ac93e0dda1f6d448cae19bbfab8c5e720a5e48bc"
    assert config.model.weight_file == "RealVisXL_V5.0_fp16.safetensors"
    assert (
        config.model.weight_sha256
        == "6a35a7855770ae9820a3c931d4964c3817b6d9e3c6f9c4dabb5b3a94e5643b80"
    )
    assert config.model.dtype == "float16"
    assert config.compute.max_reserved_vram_gib == 10.5
    assert config.compute.stage_text_encoders is True
    assert (config.generation.width, config.generation.height) == (640, 832)
    assert config.generation.seeds.values(3) == [11000, 11001, 11002]
    first_prompt, first_variant = config.prompts.render_positive(1)
    ninth_prompt, ninth_variant = config.prompts.render_positive(9)
    assert first_variant == ninth_variant == 1
    assert first_prompt == ninth_prompt
    assert "{identity}" not in first_prompt


def test_rejects_dimensions_not_divisible_by_eight() -> None:
    with pytest.raises(ValidationError, match="divisible by 8"):
        GenerationSettings(
            width=767,
            height=1024,
            steps=30,
            guidance_scale=6,
            seeds=SeedRange(start=1, count=1),
        )


def test_rejects_unimplemented_or_unsafe_model_options() -> None:
    with pytest.raises(ValidationError, match="only supports the PyTorch SDPA"):
        ComputeProfile(attention_backend="xformers")

    with pytest.raises(ValidationError, match="requires safetensors"):
        ModelProfile(
            backend="realvisxl",
            model_id="example/model",
            revision="pinned-revision",
            use_safetensors=False,
        )

    with pytest.raises(ValidationError, match="does not execute remote model code"):
        ModelProfile(
            backend="realvisxl",
            model_id="example/model",
            revision="pinned-revision",
            trust_remote_code=True,
        )
