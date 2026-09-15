import json
import os
import tomllib
from pathlib import Path

import pytest

from mara_lab.backends.reference_fake import FakeReferenceBackend
from mara_lab.config import (
    ReferenceBackendName,
    load_reference_experiment_config,
    load_training_experiment_config,
)
from mara_lab.errors import ConfigurationError
from mara_lab.trainers.sd_scripts import compile_sd_scripts_arguments
from mara_lab.workflows.dataset import build_character_dataset
from mara_lab.workflows.references import run_references
from mara_lab.workflows.train import (
    _training_process_environment,
    stage_training_split,
    validate_training_dataset,
    write_sd_scripts_dataset_config,
)

ROOT = Path(__file__).parents[1]


def test_training_process_uses_pinned_offline_model_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "existing-path")
    config = load_training_experiment_config(
        ROOT / "configs" / "training" / "mara-lora-smoke-v001.yaml"
    )

    environment = _training_process_environment(config, ROOT / "trainer-source")

    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["HF_HUB_CACHE"] == str(config.model.cache_dir.resolve())
    assert environment["PYTHONPATH"].split(os.pathsep) == [
        str(ROOT / "trainer-source"),
        "existing-path",
    ]


def _reference_run(tmp_path: Path):
    config = load_reference_experiment_config(
        ROOT / "configs" / "experiments" / "mara-references-pass-a-v001.yaml"
    )
    config = config.model_copy(
        update={
            "conditioner": config.conditioner.model_copy(
                update={"backend": ReferenceBackendName.FAKE}
            ),
            "output": config.output.model_copy(update={"root": tmp_path / "experiments"}),
        }
    )
    return run_references(
        config,
        image_limit=3,
        run_id="dataset-source",
        backend_factory=FakeReferenceBackend,
    )


def test_builds_immutable_captioned_dataset(tmp_path: Path) -> None:
    run = _reference_run(tmp_path)
    dataset_root = tmp_path / "characters" / "mara" / "datasets" / "v001"

    summary = build_character_dataset(
        run.run_dir,
        ["reference-a1-01-seed-31000", "reference-a1-02-seed-31001"],
        ["reference-a1-03-seed-31002"],
        dataset_root=dataset_root,
        dataset_id="mara-v001",
        token="mara_v01",
        expected_train_count=2,
        expected_validation_count=1,
        source_pass_id="a",
    )

    assert summary.train_count == 2
    assert summary.validation_count == 1
    records = [
        json.loads(line) for line in summary.manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["split"] for record in records] == ["train", "train", "validation"]
    caption = (dataset_root / records[0]["caption_path"]).read_text(encoding="utf-8")
    assert "photo of mara_v01, an adult woman" in caption
    assert "indirect window light" in caption


def test_final_dataset_rejects_intermediate_reference_pass(tmp_path: Path) -> None:
    run = _reference_run(tmp_path)

    with pytest.raises(ConfigurationError, match="reference pass c"):
        build_character_dataset(
            run.run_dir,
            ["reference-a1-01-seed-31000", "reference-a1-02-seed-31001"],
            ["reference-a1-03-seed-31002"],
            dataset_root=tmp_path / "dataset",
            dataset_id="mara-v001",
            token="mara_v01",
            expected_train_count=2,
            expected_validation_count=1,
        )


def test_validates_dataset_and_compiles_sd_scripts_contract(tmp_path: Path) -> None:
    run = _reference_run(tmp_path)
    dataset_root = tmp_path / "dataset"
    summary = build_character_dataset(
        run.run_dir,
        ["reference-a1-01-seed-31000", "reference-a1-02-seed-31001"],
        ["reference-a1-03-seed-31002"],
        dataset_root=dataset_root,
        dataset_id="mara-v001",
        token="mara_v01",
        expected_train_count=2,
        expected_validation_count=1,
        source_pass_id="a",
    )
    config = load_training_experiment_config(
        ROOT / "configs" / "training" / "mara-lora-smoke-v001.yaml"
    ).model_copy(
        update={
            "dataset_root": dataset_root,
            "dataset_manifest": summary.manifest_path,
            "expected_train_images": 2,
            "expected_validation_images": 1,
        }
    )

    validated = validate_training_dataset(config)
    staged = stage_training_split(validated, tmp_path / "staged-training")
    dataset_toml = tmp_path / "dataset.toml"
    write_sd_scripts_dataset_config(dataset_toml, config, staged)
    parsed = tomllib.loads(dataset_toml.read_text(encoding="utf-8"))
    assert parsed["datasets"][0]["resolution"] == [768, 768]
    assert parsed["datasets"][0]["min_bucket_reso"] == 512
    assert parsed["datasets"][0]["subsets"][0]["image_dir"] == str(staged.train_dir.resolve())
    assert {path.name for path in staged.train_dir.iterdir()} == {
        path.name for path in (dataset_root / "train").iterdir()
    }

    arguments = compile_sd_scripts_arguments(
        config,
        checkpoint_path=tmp_path / "base.safetensors",
        dataset_config_path=dataset_toml,
        output_dir=tmp_path / "adapters",
        logging_dir=tmp_path / "logs",
        max_train_steps=200,
    )
    assert "--network_train_unet_only" in arguments
    assert arguments[arguments.index("--network_dim") + 1] == "16"
    assert arguments[arguments.index("--optimizer_type") + 1] == "AdamW8bit"
    assert arguments[arguments.index("--mixed_precision") + 1] == "bf16"
    assert "--full_bf16" in arguments
    assert "--cache_text_encoder_outputs_to_disk" in arguments
    assert "--sdpa" in arguments
    assert "--torch_compile" not in arguments

    one_step_arguments = compile_sd_scripts_arguments(
        config,
        checkpoint_path=tmp_path / "base.safetensors",
        dataset_config_path=dataset_toml,
        output_dir=tmp_path / "one-step-adapters",
        logging_dir=tmp_path / "one-step-logs",
        max_train_steps=1,
    )
    assert one_step_arguments[one_step_arguments.index("--lr_warmup_steps") + 1] == "0"


def test_rejects_wrong_validation_split_size(tmp_path: Path) -> None:
    run = _reference_run(tmp_path)
    dataset_root = tmp_path / "dataset"
    summary = build_character_dataset(
        run.run_dir,
        ["reference-a1-01-seed-31000", "reference-a1-02-seed-31001"],
        ["reference-a1-03-seed-31002"],
        dataset_root=dataset_root,
        dataset_id="mara-v001",
        token="mara_v01",
        expected_train_count=2,
        expected_validation_count=1,
        source_pass_id="a",
    )
    config = load_training_experiment_config(
        ROOT / "configs" / "training" / "mara-lora-smoke-v001.yaml"
    ).model_copy(
        update={
            "dataset_root": dataset_root,
            "dataset_manifest": summary.manifest_path,
            "expected_train_images": 2,
            "expected_validation_images": 2,
        }
    )

    with pytest.raises(ConfigurationError, match="dataset has 1 validation images"):
        validate_training_dataset(config)


def test_rejects_untracked_training_file(tmp_path: Path) -> None:
    run = _reference_run(tmp_path)
    dataset_root = tmp_path / "dataset"
    summary = build_character_dataset(
        run.run_dir,
        ["reference-a1-01-seed-31000", "reference-a1-02-seed-31001"],
        ["reference-a1-03-seed-31002"],
        dataset_root=dataset_root,
        dataset_id="mara-v001",
        token="mara_v01",
        expected_train_count=2,
        expected_validation_count=1,
        source_pass_id="a",
    )
    (dataset_root / "train" / "untracked.txt").write_text("surprise\n", encoding="utf-8")
    config = load_training_experiment_config(
        ROOT / "configs" / "training" / "mara-lora-smoke-v001.yaml"
    ).model_copy(
        update={
            "dataset_root": dataset_root,
            "dataset_manifest": summary.manifest_path,
            "expected_train_images": 2,
            "expected_validation_images": 1,
        }
    )

    with pytest.raises(ConfigurationError, match="train directory contents"):
        validate_training_dataset(config)
