from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from mara_lab.artifacts import atomic_write_json, read_jsonl, sha256_file
from mara_lab.flux2_lora import (
    flux2_training_command,
    load_flux2_lora_config,
    materialize_flux2_base_model,
    materialize_flux2_lora_dataset,
    prepare_flux2_trainer,
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def cache_prompt_embeddings(config_path: Path, repository_root: Path) -> Path:
    config = load_flux2_lora_config(config_path)
    dataset = materialize_flux2_lora_dataset(config, repository_root)
    settings = config.prompt_cache
    output_path = (repository_root / settings.output_path).resolve()
    metadata_path = output_path.with_suffix(".json")
    if settings.mode == "single":
        prompts = [config.dataset.instance_prompt]
        image_filenames: list[str] = []
        expected_metadata = {
            "schema_version": 1,
            "prompt": config.dataset.instance_prompt,
            "max_sequence_length": settings.max_sequence_length,
            "text_encoder_out_layers": settings.text_encoder_out_layers,
            "text_encoder_root": settings.text_encoder_root.as_posix(),
        }
    else:
        records = sorted(
            read_jsonl(dataset.manifest_path),
            key=lambda record: Path(record["image_path"]).name,
        )
        image_filenames = [Path(record["image_path"]).name for record in records]
        prompts = [
            (dataset.root / record["caption_path"]).read_text(encoding="utf-8").strip()
            for record in records
        ]
        if len(prompts) != dataset.count:
            raise ValueError("per-image prompt count does not match the dataset")
        if any(config.dataset.trigger_token not in prompt for prompt in prompts):
            raise ValueError("a per-image prompt is missing the trigger token")
        expected_metadata = {
            "schema_version": 1,
            "mode": settings.mode,
            "dataset_manifest_sha256": sha256_file(dataset.manifest_path),
            "items": [
                {
                    "caption_sha256": record["caption_sha256"],
                    "image_filename": Path(record["image_path"]).name,
                }
                for record in records
            ],
            "max_sequence_length": settings.max_sequence_length,
            "text_encoder_out_layers": settings.text_encoder_out_layers,
            "text_encoder_root": settings.text_encoder_root.as_posix(),
        }
    if output_path.is_file() and metadata_path.is_file():
        found = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_with_hash = {**expected_metadata, "sha256": sha256_file(output_path)}
        if found == expected_with_hash:
            print(f"Reusing prompt cache: {output_path}", flush=True)
            return output_path
        raise FileExistsError(f"existing prompt cache metadata differs: {metadata_path}")

    import torch
    from diffusers import Flux2KleinPipeline
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3ForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("prompt caching requires CUDA")
    source_model_root = (repository_root / config.model.shared_components_root).resolve()
    text_encoder_root = (repository_root / settings.text_encoder_root).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        source_model_root / "tokenizer", local_files_only=True
    )
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print(
        f"Loading the 4-bit Qwen text encoder for {len(prompts)} prompt(s)...",
        flush=True,
    )
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        text_encoder_root,
        dtype=torch.bfloat16,
        device_map={"": 0},
        offload_state_dict=True,
        quantization_config=quantization_config,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval()
    pipeline = Flux2KleinPipeline.from_pretrained(
        source_model_root,
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=None,
        local_files_only=True,
    )
    try:
        prompt_embeds_parts = []
        text_ids_parts = []
        for index, prompt in enumerate(prompts, start=1):
            print(f"Encoding prompt {index}/{len(prompts)}...", flush=True)
            with torch.inference_mode():
                prompt_embeds, text_ids = pipeline.encode_prompt(
                    prompt=prompt,
                    device=torch.device("cuda"),
                    max_sequence_length=settings.max_sequence_length,
                    text_encoder_out_layers=tuple(settings.text_encoder_out_layers),
                )
            prompt_embeds_parts.append(prompt_embeds.cpu())
            text_ids_parts.append(text_ids.cpu())
        payload = {
            "mode": settings.mode,
            "max_sequence_length": settings.max_sequence_length,
            "text_encoder_out_layers": settings.text_encoder_out_layers,
            "prompt_embeds": torch.cat(prompt_embeds_parts, dim=0),
            "text_ids": torch.cat(text_ids_parts, dim=0),
        }
        if settings.mode == "single":
            payload["prompt"] = prompts[0]
        else:
            payload["captions"] = prompts
            payload["image_filenames"] = image_filenames
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, output_path)
        atomic_write_json(
            metadata_path,
            {**expected_metadata, "sha256": sha256_file(output_path)},
        )
        print(f"Prompt cache: {output_path}", flush=True)
        return output_path
    finally:
        del pipeline, text_encoder, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def run_training(args: argparse.Namespace) -> Path:
    repository_root = args.repository_root.resolve()
    config_path = args.config.resolve()
    config = load_flux2_lora_config(config_path)
    dataset = materialize_flux2_lora_dataset(config, repository_root)
    print(f"Curated dataset ready: {dataset.count} images", flush=True)
    model_root = materialize_flux2_base_model(config, repository_root)
    print(f"Base model ready: {model_root}", flush=True)
    trainer_path = prepare_flux2_trainer(config, repository_root)
    print(f"Patched official trainer ready: {trainer_path}", flush=True)

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(config_path),
            "--repository-root",
            str(repository_root),
            "--prompt-only",
        ],
        check=True,
        cwd=repository_root,
    )
    if args.prepare_only:
        return dataset.root

    output_dir = (repository_root / config.output_root / args.run_id).resolve()
    if output_dir.exists():
        raise FileExistsError(f"training run already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    command = flux2_training_command(
        config,
        repository_root,
        Path(sys.executable),
        trainer_path,
        dataset,
        output_dir,
        max_train_steps=args.steps,
    )
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_MODULE_LOADING": "LAZY",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    status_path = output_dir / "mara-status.json"
    log_path = output_dir / "training.log"
    started_at = utc_now()
    status = {
        "schema_version": 1,
        "state": "running",
        "run_id": args.run_id,
        "started_at": started_at,
        "finished_at": None,
        "max_train_steps": args.steps or config.trainer.max_train_steps,
        "command": command,
        "error": None,
    }
    atomic_write_json(status_path, status)
    start = time.perf_counter()
    print(f"Starting LoRA training: {status['max_train_steps']} optimizer steps", flush=True)
    try:
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            process = subprocess.Popen(
                command,
                cwd=repository_root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
        lora_path = output_dir / "pytorch_lora_weights.safetensors"
        if not lora_path.is_file():
            raise FileNotFoundError(f"trainer did not produce LoRA weights: {lora_path}")
        status.update(
            state="completed",
            finished_at=utc_now(),
            wall_seconds=round(time.perf_counter() - start, 3),
            lora_path=lora_path.as_posix(),
            lora_sha256=sha256_file(lora_path),
            packages={
                package: version(package)
                for package in ("accelerate", "bitsandbytes", "diffusers", "peft", "torch")
            },
        )
        atomic_write_json(status_path, status)
        print(f"LoRA smoke completed: {lora_path}", flush=True)
        return output_dir
    except Exception as error:
        status.update(
            state="failed",
            finished_at=utc_now(),
            wall_seconds=round(time.perf_counter() - start, 3),
            error=f"{type(error).__name__}: {error}",
        )
        atomic_write_json(status_path, status)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and run the Mara FLUX.2 Klein LoRA smoke."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/mara-flux2-klein-lora-smoke-v001.yaml"),
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prompt-only", action="store_true")
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if not args.prompt_only and not args.prepare_only and not args.run_id:
        parser.error("--run-id is required unless --prepare-only or --prompt-only is used")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.prompt_only:
        cache_prompt_embeddings(parsed.config.resolve(), parsed.repository_root.resolve())
    else:
        run_training(parsed)
