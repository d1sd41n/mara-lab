from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    Flux2KleinPipeline,
    Flux2Transformer2DModel,
)
from transformers import BitsAndBytesConfig

from mara_lab.artifacts import atomic_write_json, save_png_atomic, sha256_file
from mara_lab.contact_sheet import create_contact_sheet
from mara_lab.flux2_lora import load_flux2_lora_config


@dataclass(frozen=True)
class EvaluationRecord:
    artifact_id: str
    sequence: int
    seed: int
    width: int
    height: int
    image_path: str
    lora_enabled: bool
    wall_seconds: float
    sha256: str


def run(args: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("FLUX.2 LoRA evaluation requires CUDA")
    repository_root = args.repository_root.resolve()
    config = load_flux2_lora_config(args.config.resolve())
    model_root = (repository_root / config.model.output_root).resolve()
    prompt_cache_path = (repository_root / config.prompt_cache.output_path).resolve()
    adapter_root = (repository_root / config.output_root / args.run_id).resolve()
    adapter_path = adapter_root / "pytorch_lora_weights.safetensors"
    if not adapter_path.is_file():
        raise FileNotFoundError(adapter_path)
    output_dir = adapter_root / args.evaluation_id
    if output_dir.exists():
        raise FileExistsError(f"evaluation already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    prompt_cache = torch.load(prompt_cache_path, map_location="cpu", weights_only=True)
    if prompt_cache["prompt"] != config.dataset.instance_prompt:
        raise ValueError("prompt cache does not match the training prompt")
    prompt_embeds = prompt_cache["prompt_embeds"].to(device="cuda", dtype=torch.bfloat16)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print("Loading the 4-bit FLUX.2 Klein Base transformer...", flush=True)
    transformer = Flux2Transformer2DModel.from_pretrained(
        model_root,
        subfolder="transformer",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
    )
    vae = AutoencoderKLFlux2.from_pretrained(
        model_root,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_root, subfolder="scheduler", local_files_only=True
    )
    pipe = Flux2KleinPipeline(
        scheduler=scheduler,
        vae=vae,
        transformer=transformer,
        text_encoder=None,
        tokenizer=None,
        is_distilled=False,
    )
    pipe.vae.enable_tiling()
    pipe.load_lora_weights(adapter_root)

    records: list[EvaluationRecord] = []
    for sequence, (artifact_id, lora_enabled) in enumerate(
        (("base", False), ("lora-step-50", True)), start=1
    ):
        if lora_enabled:
            pipe.enable_lora()
        else:
            pipe.disable_lora()
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        print(f"Generating {artifact_id}...", flush=True)
        started = time.perf_counter()
        with torch.inference_mode():
            image = pipe(
                prompt=None,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=torch.zeros_like(prompt_embeds),
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                max_sequence_length=config.prompt_cache.max_sequence_length,
            ).images[0]
        wall_seconds = time.perf_counter() - started
        destination = output_dir / f"{artifact_id}-seed-{args.seed}.png"
        save_png_atomic(image, destination)
        records.append(
            EvaluationRecord(
                artifact_id=artifact_id,
                sequence=sequence,
                seed=args.seed,
                width=image.width,
                height=image.height,
                image_path=destination.relative_to(adapter_root).as_posix(),
                lora_enabled=lora_enabled,
                wall_seconds=round(wall_seconds, 3),
                sha256=sha256_file(destination),
            )
        )
        del image
        torch.cuda.empty_cache()

    sheet_path = output_dir / "comparison.png"
    create_contact_sheet(
        adapter_root,
        records,
        sheet_path,
        columns=2,
        thumbnail_width=288,
        labeler=lambda record: "Base" if not record.lora_enabled else "LoRA, 50 steps",
    )
    atomic_write_json(
        output_dir / "evaluation.json",
        {
            "schema_version": 1,
            "adapter_sha256": sha256_file(adapter_path),
            "guidance_scale": args.guidance_scale,
            "height": args.height,
            "model_id": config.model.model_id,
            "model_revision": config.model.revision,
            "peak_reserved_vram_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 4),
            "prompt": config.dataset.instance_prompt,
            "records": [asdict(record) for record in records],
            "seed": args.seed,
            "steps": args.steps,
            "width": args.width,
        },
    )
    print(f"Comparison: {sheet_path}", flush=True)
    return sheet_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare FLUX.2 Base with and without Mara LoRA.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/mara-flux2-klein-lora-smoke-v001.yaml"),
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", default="mara-flux2-klein-lora-smoke-v001")
    parser.add_argument("--evaluation-id", default="evaluation-v002")
    parser.add_argument("--seed", type=int, default=49001)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
