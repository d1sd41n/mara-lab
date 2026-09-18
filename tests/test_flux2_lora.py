from pathlib import Path

from mara_lab.flux2_lora import (
    flux2_training_command,
    load_flux2_lora_config,
    materialize_flux2_lora_dataset,
)

ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs" / "training" / "mara-flux2-klein-lora-smoke-v001.yaml"
CANDIDATE_CONFIG_PATH = (
    ROOT / "configs" / "training" / "mara-flux2-klein-lora-candidate-v001.yaml"
)


def test_flux2_dataset_selection_is_curated_and_reproducible(tmp_path: Path) -> None:
    config = load_flux2_lora_config(CONFIG_PATH)
    output_root = tmp_path / "mara-v001"
    config = config.model_copy(
        update={"dataset": config.dataset.model_copy(update={"output_root": output_root})}
    )

    first = materialize_flux2_lora_dataset(config, ROOT)
    second = materialize_flux2_lora_dataset(config, ROOT)

    assert first == second
    assert first.count == 20
    assert len(list(first.images_dir.glob("*.png"))) == 20
    captions = [path.read_text(encoding="utf-8") for path in first.captions_dir.glob("*.txt")]
    assert len(captions) == 20
    assert all("MARA_K1" in caption for caption in captions)
    assert all("mara_v01" not in caption for caption in captions)


def test_flux2_smoke_command_uses_consumer_gpu_memory_controls(tmp_path: Path) -> None:
    config = load_flux2_lora_config(CONFIG_PATH)
    dataset_root = tmp_path / "dataset"
    config = config.model_copy(
        update={"dataset": config.dataset.model_copy(update={"output_root": dataset_root})}
    )
    dataset = materialize_flux2_lora_dataset(config, ROOT)
    command = flux2_training_command(
        config,
        ROOT,
        ROOT / ".venv" / "Scripts" / "python.exe",
        tmp_path / "trainer.py",
        dataset,
        tmp_path / "output",
        max_train_steps=1,
    )

    assert "--precomputed_prompt_embeddings" in command
    assert "--bnb_quantization_config_path" in command
    assert "--gradient_checkpointing" in command
    assert "--cache_latents" in command
    assert "--use_8bit_adam" in command
    assert command[command.index("--max_train_steps") + 1] == "1"
    assert command[command.index("--aspect_ratio_buckets") + 1] == "512,384"


def test_flux2_candidate_uses_per_image_prompts_and_three_checkpoints(
    tmp_path: Path,
) -> None:
    config = load_flux2_lora_config(CANDIDATE_CONFIG_PATH)
    dataset_root = tmp_path / "dataset"
    config = config.model_copy(
        update={"dataset": config.dataset.model_copy(update={"output_root": dataset_root})}
    )
    dataset = materialize_flux2_lora_dataset(config, ROOT)
    command = flux2_training_command(
        config,
        ROOT,
        ROOT / ".venv" / "Scripts" / "python.exe",
        tmp_path / "trainer.py",
        dataset,
        tmp_path / "output",
    )

    assert config.prompt_cache.mode == "per_image"
    assert command[command.index("--max_train_steps") + 1] == "600"
    assert command[command.index("--checkpointing_steps") + 1] == "200"
    assert command[command.index("--checkpoints_total_limit") + 1] == "3"
