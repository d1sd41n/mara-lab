from __future__ import annotations

# ruff: noqa: E501 - exact upstream source fragments must retain their line layout.
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mara_lab.artifacts import atomic_write_json, atomic_write_text, read_jsonl, sha256_file
from mara_lab.model_materialization import shard_safetensors

TRANSFORMER_MAX_SHARD_BYTES = 512 * 1024**2


class CanonicalImage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    path: Path
    sha256: str
    caption: str


class DatasetSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_root: Path
    expected_images: int = Field(gt=0)
    source_manifest: Path
    source_manifest_sha256: str
    source_token: str
    trigger_token: str
    instance_prompt: str
    canonical_images: list[CanonicalImage]
    selected_artifact_ids: list[str]

    @model_validator(mode="after")
    def validate_selection(self) -> DatasetSettings:
        artifact_ids = [item.artifact_id for item in self.canonical_images]
        artifact_ids.extend(self.selected_artifact_ids)
        if len(artifact_ids) != self.expected_images:
            raise ValueError(
                f"expected {self.expected_images} images, configured {len(artifact_ids)}"
            )
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("dataset artifact IDs must be unique")
        if self.source_token == self.trigger_token:
            raise ValueError("source_token and trigger_token must differ")
        if self.trigger_token not in self.instance_prompt:
            raise ValueError("instance_prompt must contain trigger_token")
        return self


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    revision: str
    output_root: Path
    transformer_sha256: str
    shared_components_root: Path
    vae_sha256: str


class PromptCacheSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_path: Path
    text_encoder_root: Path
    max_sequence_length: int = Field(gt=0)
    text_encoder_out_layers: list[int]


class TrainerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repository_url: str
    repository_root: Path
    revision: str
    source_path: Path
    generated_path: Path
    quantization_config: Path
    mixed_precision: str
    aspect_ratio_buckets: str
    train_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    rank: int = Field(gt=0)
    lora_alpha: int = Field(gt=0)
    max_train_steps: int = Field(gt=0)
    seed: int


class Flux2LoraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    experiment_id: str
    character_id: str
    dataset: DatasetSettings
    model: ModelSettings
    prompt_cache: PromptCacheSettings
    trainer: TrainerSettings
    output_root: Path


@dataclass(frozen=True)
class MaterializedDataset:
    root: Path
    images_dir: Path
    captions_dir: Path
    manifest_path: Path
    count: int


def load_flux2_lora_config(path: Path) -> Flux2LoraConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Flux2LoraConfig.model_validate(raw)


def config_fingerprint(config: Flux2LoraConfig) -> str:
    payload = json.dumps(
        config.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _resolve(repository_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repository_root / path


def _write_once(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"existing cache file has different content: {path}")
        return
    atomic_write_text(path, content)


def _copy_once(source: Path, destination: Path, expected_sha256: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    found_sha256 = sha256_file(source)
    if found_sha256 != expected_sha256:
        raise ValueError(
            f"source hash mismatch for {source}: expected {expected_sha256}, found {found_sha256}"
        )
    if destination.exists():
        if sha256_file(destination) != expected_sha256:
            raise FileExistsError(f"existing cache image has different content: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def materialize_flux2_lora_dataset(
    config: Flux2LoraConfig, repository_root: Path
) -> MaterializedDataset:
    repository_root = repository_root.resolve()
    settings = config.dataset
    source_manifest = _resolve(repository_root, settings.source_manifest)
    if sha256_file(source_manifest) != settings.source_manifest_sha256:
        raise ValueError(f"source dataset manifest hash mismatch: {source_manifest}")

    source_records = read_jsonl(source_manifest)
    source_by_artifact_id: dict[str, dict[str, Any]] = {}
    for record in source_records:
        artifact_id = record["source_artifact"]["artifact_id"]
        if artifact_id in source_by_artifact_id:
            raise ValueError(f"duplicate source artifact ID: {artifact_id}")
        source_by_artifact_id[artifact_id] = record

    missing = sorted(set(settings.selected_artifact_ids) - source_by_artifact_id.keys())
    if missing:
        raise ValueError(f"selected source artifacts not found: {', '.join(missing)}")

    output_root = _resolve(repository_root, settings.output_root)
    images_dir = output_root / "images"
    captions_dir = output_root / "captions"
    images_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    materialized_records: list[dict[str, Any]] = []
    for canonical in settings.canonical_images:
        source_image = _resolve(repository_root, canonical.path)
        image_path = images_dir / f"{canonical.artifact_id}{source_image.suffix.lower()}"
        caption_path = captions_dir / f"{canonical.artifact_id}.txt"
        caption = canonical.caption.strip() + "\n"
        if settings.trigger_token not in caption:
            raise ValueError(f"canonical caption lacks trigger token: {canonical.artifact_id}")
        _copy_once(source_image, image_path, canonical.sha256)
        _write_once(caption_path, caption)
        materialized_records.append(
            {
                "artifact_id": canonical.artifact_id,
                "caption_path": caption_path.relative_to(output_root).as_posix(),
                "caption_sha256": sha256_file(caption_path),
                "image_path": image_path.relative_to(output_root).as_posix(),
                "image_sha256": sha256_file(image_path),
                "source_kind": "canonical",
                "source_path": canonical.path.as_posix(),
            }
        )

    source_dataset_root = source_manifest.parent
    for artifact_id in settings.selected_artifact_ids:
        source_record = source_by_artifact_id[artifact_id]
        source_image = source_dataset_root / source_record["image_path"]
        source_caption = source_dataset_root / source_record["caption_path"]
        _copy_once(source_image, images_dir / f"{artifact_id}.png", source_record["image_sha256"])
        if sha256_file(source_caption) != source_record["caption_sha256"]:
            raise ValueError(f"source caption hash mismatch: {source_caption}")
        caption = source_caption.read_text(encoding="utf-8")
        if settings.source_token not in caption:
            raise ValueError(f"source token missing from caption: {artifact_id}")
        caption = caption.replace(settings.source_token, settings.trigger_token)
        if settings.source_token in caption or settings.trigger_token not in caption:
            raise ValueError(f"caption token replacement failed: {artifact_id}")
        destination_caption = captions_dir / f"{artifact_id}.txt"
        _write_once(destination_caption, caption)
        destination_image = images_dir / f"{artifact_id}.png"
        materialized_records.append(
            {
                "artifact_id": artifact_id,
                "caption_path": destination_caption.relative_to(output_root).as_posix(),
                "caption_sha256": sha256_file(destination_caption),
                "image_path": destination_image.relative_to(output_root).as_posix(),
                "image_sha256": sha256_file(destination_image),
                "source_kind": "mara-v002",
                "source_item_id": source_record["item_id"],
                "source_split": source_record["split"],
            }
        )

    expected_image_names = {Path(record["image_path"]).name for record in materialized_records}
    expected_caption_names = {Path(record["caption_path"]).name for record in materialized_records}
    found_image_names = {path.name for path in images_dir.iterdir() if path.is_file()}
    found_caption_names = {path.name for path in captions_dir.iterdir() if path.is_file()}
    if found_image_names != expected_image_names:
        raise ValueError("materialized image directory contains unexpected files")
    if found_caption_names != expected_caption_names:
        raise ValueError("materialized caption directory contains unexpected files")

    manifest_content = "".join(
        json.dumps(record, sort_keys=True) + "\n" for record in materialized_records
    )
    manifest_path = output_root / "manifest.jsonl"
    _write_once(manifest_path, manifest_content)
    materialization_path = output_root / "materialization.json"
    metadata = {
        "schema_version": 1,
        "character_id": config.character_id,
        "config_sha256": config_fingerprint(config),
        "count": len(materialized_records),
        "manifest_sha256": sha256_file(manifest_path),
        "source_manifest": settings.source_manifest.as_posix(),
        "source_manifest_sha256": settings.source_manifest_sha256,
        "trigger_token": settings.trigger_token,
    }
    if materialization_path.exists():
        if json.loads(materialization_path.read_text(encoding="utf-8")) != metadata:
            raise FileExistsError(
                f"existing materialization metadata differs: {materialization_path}"
            )
    else:
        atomic_write_json(materialization_path, metadata)

    if len(materialized_records) != settings.expected_images:
        raise ValueError(
            f"materialized {len(materialized_records)} images, expected {settings.expected_images}"
        )
    return MaterializedDataset(
        root=output_root,
        images_dir=images_dir,
        captions_dir=captions_dir,
        manifest_path=manifest_path,
        count=len(materialized_records),
    )


def materialize_flux2_base_model(config: Flux2LoraConfig, repository_root: Path) -> Path:
    repository_root = repository_root.resolve()
    settings = config.model
    model_root = _resolve(repository_root, settings.output_root)
    transformer_path = model_root / "transformer" / "diffusion_pytorch_model.safetensors"
    vae_path = model_root / "vae" / "diffusion_pytorch_model.safetensors"

    required_small_files = (
        model_root / "model_index.json",
        model_root / "scheduler" / "scheduler_config.json",
        model_root / "transformer" / "config.json",
        model_root / "vae" / "config.json",
    )
    needs_download = not transformer_path.is_file() or any(
        not path.is_file() for path in required_small_files
    )
    if needs_download:
        try:
            import truststore

            truststore.inject_into_ssl()
        except ImportError:
            pass
        from huggingface_hub import snapshot_download

        model_root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=settings.model_id,
            revision=settings.revision,
            local_dir=model_root,
            allow_patterns=(
                "model_index.json",
                "scheduler/*",
                "transformer/*",
                "vae/config.json",
            ),
        )

    if sha256_file(transformer_path) != settings.transformer_sha256:
        raise ValueError(f"base transformer hash mismatch: {transformer_path}")

    transformer_index = transformer_path.with_name(f"{transformer_path.name}.index.json")
    if not transformer_index.is_file():
        print("Re-sharding the base transformer into 512 MiB files...", flush=True)
        shard_safetensors(
            transformer_path,
            transformer_path.parent,
            max_shard_bytes=TRANSFORMER_MAX_SHARD_BYTES,
        )
    try:
        transformer_index_data = json.loads(transformer_index.read_text(encoding="utf-8"))
        transformer_shards = sorted(set(transformer_index_data["weight_map"].values()))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"invalid transformer shard index: {transformer_index}") from exc
    if not transformer_shards or any(
        not (transformer_path.parent / shard).is_file() for shard in transformer_shards
    ):
        raise ValueError(f"transformer shard set is incomplete: {transformer_path.parent}")

    shared_vae = (
        _resolve(repository_root, settings.shared_components_root)
        / "vae"
        / "diffusion_pytorch_model.safetensors"
    )
    if sha256_file(shared_vae) != settings.vae_sha256:
        raise ValueError(f"shared VAE hash mismatch: {shared_vae}")
    if not vae_path.exists():
        vae_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(shared_vae, vae_path)
        except OSError:
            shutil.copy2(shared_vae, vae_path)
    if sha256_file(vae_path) != settings.vae_sha256:
        raise ValueError(f"materialized VAE hash mismatch: {vae_path}")

    atomic_write_json(
        model_root / "mara-materialization.json",
        {
            "schema_version": 1,
            "model_id": settings.model_id,
            "revision": settings.revision,
            "max_transformer_shard_bytes": TRANSFORMER_MAX_SHARD_BYTES,
            "transformer_index_sha256": sha256_file(transformer_index),
            "transformer_shard_count": len(transformer_shards),
            "transformer_sha256": settings.transformer_sha256,
            "vae_sha256": settings.vae_sha256,
        },
    )
    return model_root


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"trainer patch marker {label!r} occurred {count} times")
    return source.replace(old, new, 1)


def patch_flux2_trainer_source(source: str) -> str:
    source = _replace_once(
        source,
        "        log_with=args.report_to,\n",
        '        log_with=None if args.report_to == "none" else args.report_to,\n',
        "disabled logging",
    )
    source = _replace_once(
        source,
        """    parser.add_argument(
        "--do_fp8_training",
        action="store_true",
        help="if we are doing FP8 training.",
    )
""",
        """    parser.add_argument(
        "--precomputed_prompt_embeddings",
        type=str,
        default=None,
        help="Load one precomputed prompt embedding and skip the text encoder.",
    )
    parser.add_argument(
        "--do_fp8_training",
        action="store_true",
        help="if we are doing FP8 training.",
    )
""",
        "argument",
    )
    source = _replace_once(
        source,
        """    if args.do_fp8_training and args.bnb_quantization_config_path:
        raise ValueError("Both `do_fp8_training` and `bnb_quantization_config_path` cannot be passed.")
""",
        """    if args.do_fp8_training and args.bnb_quantization_config_path:
        raise ValueError("Both `do_fp8_training` and `bnb_quantization_config_path` cannot be passed.")
    if args.precomputed_prompt_embeddings is not None:
        if args.dataset_name is not None:
            raise ValueError("Precomputed prompt embeddings require --instance_data_dir.")
        if args.with_prior_preservation:
            raise ValueError("Precomputed prompt embeddings do not support prior preservation.")
        if args.validation_prompt is not None or args.final_validation_prompt is not None:
            raise ValueError("Precomputed prompt embeddings do not support trainer validation.")
        if args.fsdp_text_encoder:
            raise ValueError("Precomputed prompt embeddings cannot be combined with --fsdp_text_encoder.")
""",
        "validation",
    )
    source = _replace_once(
        source,
        """    # Load the tokenizers
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
""",
        """    # The tokenizer and text encoder are unnecessary when a verified prompt cache is supplied.
    tokenizer = None
    if args.precomputed_prompt_embeddings is None:
        tokenizer = Qwen2TokenizerFast.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
        )
""",
        "tokenizer",
    )
    source = _replace_once(
        source,
        """    text_encoder = Qwen3ForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder.requires_grad_(False)
""",
        """    text_encoder = None
    if args.precomputed_prompt_embeddings is None:
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.revision,
            variant=args.variant,
        )
        text_encoder.requires_grad_(False)
""",
        "text encoder",
    )
    source = _replace_once(
        source,
        """    text_encoder.to(**to_kwargs)
    # Initialize a text encoding pipeline and keep it to CPU for now.
    text_encoding_pipeline = Flux2KleinPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=None,
        revision=args.revision,
    )
""",
        """    text_encoding_pipeline = None
    if text_encoder is not None:
        text_encoder.to(**to_kwargs)
        # Initialize a text encoding pipeline and keep it to CPU for now.
        text_encoding_pipeline = Flux2KleinPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=None,
            transformer=None,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=None,
            revision=args.revision,
        )
""",
        "text pipeline",
    )
    source = _replace_once(
        source,
        """    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.
    if not train_dataset.custom_instance_prompts:
        with offload_models(text_encoding_pipeline, device=accelerator.device, offload=args.offload):
            instance_prompt_hidden_states, instance_text_ids = compute_text_embeddings(
                args.instance_prompt, text_encoding_pipeline
            )
""",
        """    if args.precomputed_prompt_embeddings is not None:
        cached_prompt = torch.load(
            args.precomputed_prompt_embeddings, map_location="cpu", weights_only=True
        )
        required_cache_keys = {
            "prompt",
            "max_sequence_length",
            "text_encoder_out_layers",
            "prompt_embeds",
            "text_ids",
        }
        missing_cache_keys = required_cache_keys - cached_prompt.keys()
        if missing_cache_keys:
            raise ValueError(f"Prompt cache is missing keys: {sorted(missing_cache_keys)}")
        if cached_prompt["prompt"] != args.instance_prompt:
            raise ValueError("Prompt cache does not match --instance_prompt.")
        if cached_prompt["max_sequence_length"] != args.max_sequence_length:
            raise ValueError("Prompt cache max sequence length does not match the trainer.")
        if cached_prompt["text_encoder_out_layers"] != args.text_encoder_out_layers:
            raise ValueError("Prompt cache text encoder layers do not match the trainer.")
        instance_prompt_hidden_states = cached_prompt["prompt_embeds"].to(
            device=accelerator.device, dtype=weight_dtype
        )
        instance_text_ids = cached_prompt["text_ids"].to(device=accelerator.device)
        del cached_prompt

    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.
    if not train_dataset.custom_instance_prompts and args.precomputed_prompt_embeddings is None:
        with offload_models(text_encoding_pipeline, device=accelerator.device, offload=args.offload):
            instance_prompt_hidden_states, instance_text_ids = compute_text_embeddings(
                args.instance_prompt, text_encoding_pipeline
            )
""",
        "prompt cache",
    )
    source = _replace_once(
        source,
        """    # move back to cpu before deleting to ensure memory is freed see: https://github.com/huggingface/diffusers/issues/11376#issue-3008144624
    text_encoding_pipeline = text_encoding_pipeline.to("cpu")
    del text_encoder, tokenizer
    free_memory()
""",
        """    # move back to cpu before deleting to ensure memory is freed see: https://github.com/huggingface/diffusers/issues/11376#issue-3008144624
    if text_encoding_pipeline is not None:
        text_encoding_pipeline = text_encoding_pipeline.to("cpu")
    del text_encoding_pipeline, text_encoder, tokenizer
    free_memory()
""",
        "cleanup",
    )
    return source


def prepare_flux2_trainer(config: Flux2LoraConfig, repository_root: Path) -> Path:
    repository_root = repository_root.resolve()
    settings = config.trainer
    trainer_repository = _resolve(repository_root, settings.repository_root)
    if not (trainer_repository / ".git").is_dir():
        trainer_repository.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                settings.repository_url,
                str(trainer_repository),
            ],
            check=True,
        )
    status = subprocess.run(
        ["git", "-C", str(trainer_repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError(f"Diffusers trainer checkout is dirty: {trainer_repository}")
    head = subprocess.run(
        ["git", "-C", str(trainer_repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != settings.revision:
        subprocess.run(
            ["git", "-C", str(trainer_repository), "checkout", "--detach", settings.revision],
            check=True,
        )
    source_path = trainer_repository / settings.source_path
    patched = patch_flux2_trainer_source(source_path.read_text(encoding="utf-8"))
    generated_path = _resolve(repository_root, settings.generated_path)
    atomic_write_text(generated_path, patched)
    atomic_write_json(
        generated_path.with_suffix(".json"),
        {
            "schema_version": 1,
            "source_commit": settings.revision,
            "source_path": settings.source_path.as_posix(),
            "source_sha256": sha256_file(source_path),
            "generated_sha256": sha256_file(generated_path),
        },
    )
    return generated_path


def flux2_training_command(
    config: Flux2LoraConfig,
    repository_root: Path,
    python_executable: Path,
    trainer_path: Path,
    dataset: MaterializedDataset,
    output_dir: Path,
    max_train_steps: int | None = None,
) -> list[str]:
    repository_root = repository_root.resolve()
    settings = config.trainer
    prompt_settings = config.prompt_cache
    steps = max_train_steps or settings.max_train_steps
    return [
        str(python_executable.resolve()),
        str(trainer_path.resolve()),
        "--pretrained_model_name_or_path",
        str(_resolve(repository_root, config.model.output_root).resolve()),
        "--instance_data_dir",
        str(dataset.images_dir.resolve()),
        "--output_dir",
        str(output_dir.resolve()),
        "--precomputed_prompt_embeddings",
        str(_resolve(repository_root, prompt_settings.output_path).resolve()),
        "--bnb_quantization_config_path",
        str(_resolve(repository_root, settings.quantization_config).resolve()),
        "--instance_prompt",
        config.dataset.instance_prompt,
        "--max_sequence_length",
        str(prompt_settings.max_sequence_length),
        "--text_encoder_out_layers",
        *[str(layer) for layer in prompt_settings.text_encoder_out_layers],
        "--aspect_ratio_buckets",
        settings.aspect_ratio_buckets,
        "--train_batch_size",
        str(settings.train_batch_size),
        "--gradient_accumulation_steps",
        str(settings.gradient_accumulation_steps),
        "--learning_rate",
        str(settings.learning_rate),
        "--rank",
        str(settings.rank),
        "--lora_alpha",
        str(settings.lora_alpha),
        "--max_train_steps",
        str(steps),
        "--seed",
        str(settings.seed),
        "--mixed_precision",
        settings.mixed_precision,
        "--guidance_scale",
        "1",
        "--lr_scheduler",
        "constant",
        "--lr_warmup_steps",
        "0",
        "--checkpointing_steps",
        str(steps + 1),
        "--report_to",
        "none",
        "--gradient_checkpointing",
        "--cache_latents",
        "--use_8bit_adam",
        "--offload",
        "--skip_final_inference",
    ]
