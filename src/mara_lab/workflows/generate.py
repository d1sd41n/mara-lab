from __future__ import annotations

from mara_lab.config import BenchmarkCase, ResolvedBenchmarkExperimentConfig
from mara_lab.errors import ConfigurationError


def build_generation_config(
    config: ResolvedBenchmarkExperimentConfig,
    scene: str,
    *,
    seed: int,
) -> ResolvedBenchmarkExperimentConfig:
    """Turn a plain scene description into one traced character generation case."""
    normalized_scene = " ".join(scene.split()).strip(" ,")
    if len(normalized_scene) < 3:
        raise ConfigurationError("scene description must contain at least three characters")
    if config.identity_token in normalized_scene:
        raise ConfigurationError(
            "describe only the scene; the character identity token is added automatically"
        )

    prompt = (
        f"unretouched ordinary photograph of {config.identity_token}, an adult woman, "
        f"{normalized_scene}"
    )
    case = BenchmarkCase(case_id="custom-scene", prompt=prompt, seed=seed)
    release_version = config.experiment_id.rsplit("-", maxsplit=1)[-1]
    return config.model_copy(
        update={
            "experiment_id": f"{config.character_id}-generate-{release_version}",
            "cases": [case],
        }
    )
