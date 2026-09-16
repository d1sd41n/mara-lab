from pathlib import Path

import pytest

from mara_lab.config import load_benchmark_experiment_config
from mara_lab.errors import ConfigurationError
from mara_lab.workflows.generate import build_generation_config

ROOT = Path(__file__).parents[1]


def test_builds_single_traced_generation_case() -> None:
    config = load_benchmark_experiment_config(
        ROOT / "configs" / "benchmarks" / "mara-lora-final-v001.yaml"
    )

    generated = build_generation_config(
        config,
        "  reading beside a laundromat window in overcast daylight  ",
        seed=43001,
    )

    assert generated.experiment_id == "mara-generate-v001"
    assert len(generated.cases) == 1
    assert generated.cases[0].seed == 43001
    assert generated.cases[0].prompt.count("mara_v01") == 1
    assert "an adult woman" in generated.cases[0].prompt
    assert "  " not in generated.cases[0].prompt


def test_generation_run_id_tracks_release_version() -> None:
    config = load_benchmark_experiment_config(
        ROOT / "configs" / "benchmarks" / "mara-lora-final-v002.yaml"
    )

    generated = build_generation_config(config, "reading beside a window", seed=43002)

    assert generated.experiment_id == "mara-generate-v002"


@pytest.mark.parametrize("scene", ["", "  ", "mara_v01 in a kitchen"])
def test_rejects_empty_or_pre_tokenized_scene(scene: str) -> None:
    config = load_benchmark_experiment_config(
        ROOT / "configs" / "benchmarks" / "mara-lora-final-v001.yaml"
    )

    with pytest.raises(ConfigurationError):
        build_generation_config(config, scene, seed=43001)
