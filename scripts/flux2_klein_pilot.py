from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from PIL import Image

from mara_lab.artifacts import append_jsonl, atomic_write_json, save_png_atomic, sha256_file
from mara_lab.contact_sheet import create_contact_sheet
from mara_lab.flux2_klein import (
    DEFAULT_FP8_TRANSFORMER_ROOT,
    DEFAULT_MODEL_ROOT,
    DEFAULT_TEXT_ENCODER_ROOT,
    FP8_CONFIG_SHA256,
    FP8_TRANSFORMER_ID,
    FP8_TRANSFORMER_REVISION,
    FP8_TRANSFORMER_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    TORCHAO_VERSION,
)


@dataclass(frozen=True)
class PilotCase:
    case_id: str
    seed: int
    scene: str

    @property
    def prompt(self) -> str:
        return (
            "Create an unretouched realistic photograph of the same fictional adult woman "
            "shown in both reference images. Preserve her facial identity, apparent age, "
            f"dark brown hair, and natural features. {self.scene} "
            "Natural anatomy, realistic skin, ordinary camera rendering, no beauty filter."
        )


@dataclass(frozen=True)
class SheetRecord:
    artifact_id: str
    sequence: int
    seed: int
    width: int
    height: int
    image_path: str


CASES = (
    PilotCase(
        "apartment-full-body",
        47001,
        "Full-body view standing casually beside an apartment window, charcoal shirt, "
        "straight-leg jeans and simple sneakers, relaxed expression, soft daylight.",
    ),
    PilotCase(
        "morning-coffee",
        47002,
        "Medium candid view making morning coffee in a modest kitchen, gray pajama shirt, "
        "sleepy relaxed eyes, soft window light.",
    ),
    PilotCase(
        "grocery-shopping",
        47003,
        "Full-body candid view choosing vegetables in a neighborhood grocery aisle, muted "
        "green shirt and jeans, thoughtful side gaze, fluorescent store light.",
    ),
    PilotCase(
        "park-walk",
        47004,
        "Medium-full candid view walking through a neighborhood park, navy jacket and jeans, "
        "looking to the side with a soft smile, late-afternoon light.",
    ),
    PilotCase(
        "laptop-cafe",
        47005,
        "Seated medium view working on a laptop at a small cafe table, light blue shirt, "
        "focused downward gaze, side window light.",
    ),
    PilotCase(
        "reading-sofa",
        47006,
        "Relaxed three-quarter view reading a paperback on a living-room sofa, burgundy "
        "sweater and jeans, looking down, warm afternoon light.",
    ),
    PilotCase(
        "cooking-dinner",
        47007,
        "Medium candid view preparing vegetables for dinner in a simple kitchen, burgundy "
        "shirt, focused side gaze, warm practical light.",
    ),
    PilotCase(
        "balcony-plants",
        47008,
        "Full-body view watering potted plants on a small balcony, dark T-shirt and jeans, "
        "gentle closed-mouth smile, bright open shade.",
    ),
    PilotCase(
        "rainy-bus-stop",
        47009,
        "Chest-up candid view waiting at a rainy city bus stop, navy rain jacket, calm side "
        "gaze, damp loose hair, overcast daylight.",
    ),
    PilotCase(
        "evening-bookstore",
        47010,
        "Medium candid view browsing shelves in a quiet neighborhood bookstore, dark blue "
        "coat, relaxed eyes and subtle smile, warm shelf lighting.",
    ),
)


BODY_EXPRESSION_CASES = (
    PilotCase(
        "full-front-neutral",
        47101,
        "Full-body front view standing naturally against a plain apartment wall, fitted "
        "charcoal T-shirt, straight-leg jeans and sneakers, both arms relaxed and both hands "
        "visible, neutral expression, even daylight, head-to-toe framing.",
    ),
    PilotCase(
        "full-back-turn",
        47102,
        "Full-body rear view walking away on a quiet sidewalk while turning her head back "
        "toward the camera, navy jacket, jeans and sneakers, natural stride, open shade, "
        "head-to-toe framing.",
    ),
    PilotCase(
        "full-profile-walk",
        47103,
        "Full-body strict side-profile view walking across a pedestrian crossing, muted green "
        "shirt, jeans and flat shoes, natural arm swing and stride, overcast daylight.",
    ),
    PilotCase(
        "low-angle-stairs",
        47104,
        "Full-body candid view walking up outdoor concrete stairs, photographed from a modest "
        "low angle, burgundy sweater, dark jeans and sneakers, looking toward the next step.",
    ),
    PilotCase(
        "high-angle-hallway",
        47105,
        "Full-body view in an apartment hallway photographed from a high corner angle, looking "
        "up toward the camera, light blue shirt, black trousers and simple shoes, relaxed stance.",
    ),
    PilotCase(
        "one-leg-balance",
        47106,
        "Full-body candid view balancing on one leg while putting on a sneaker near the front "
        "door, gray T-shirt and jeans, one hand against the wall, focused downward gaze.",
    ),
    PilotCase(
        "chair-crossed-legs",
        47107,
        "Seated full-body view in a simple dining chair with legs crossed at the ankles, navy "
        "blouse, straight trousers and flats, hands resting naturally in her lap, soft daylight.",
    ),
    PilotCase(
        "sofa-sideways",
        47108,
        "Relaxed full-body view sitting sideways on a sofa with one knee bent and one foot on "
        "the floor, burgundy sweater and jeans, reading a message on her phone.",
    ),
    PilotCase(
        "floor-cross-legged",
        47109,
        "Full-body view sitting cross-legged on a living-room rug while sorting printed photos, "
        "gray long-sleeve shirt and jeans, both hands visible, looking down with concentration.",
    ),
    PilotCase(
        "crouch-tie-shoe",
        47110,
        "Full-body three-quarter view crouching to tie a shoelace on a park bench path, dark "
        "T-shirt, jeans and sneakers, both hands clearly interacting with the laces.",
    ),
    PilotCase(
        "reach-high-shelf",
        47111,
        "Medium-full side view reaching both arms overhead for a box on a high kitchen shelf, "
        "olive shirt and jeans, natural torso stretch, focused upward gaze.",
    ),
    PilotCase(
        "carry-grocery-bags",
        47112,
        "Full-body view entering an apartment carrying one paper grocery bag in each hand, navy "
        "jacket, jeans and sneakers, natural weight and shoulder posture, subtle effort.",
    ),
    PilotCase(
        "pour-water",
        47113,
        "Waist-up three-quarter view pouring water from a glass pitcher into a drinking glass, "
        "light blue shirt, both hands visible and correctly placed, eyes following the water.",
    ),
    PilotCase(
        "write-notebook",
        47114,
        "Seated medium view writing in a notebook at a cafe table, burgundy cardigan, pen held "
        "naturally in her right hand and left hand holding the page, focused downward gaze.",
    ),
    PilotCase(
        "open-jar",
        47115,
        "Waist-up candid kitchen view opening a stubborn glass jar with both hands, charcoal "
        "T-shirt, mild effort in her face, natural wrist and finger placement.",
    ),
    PilotCase(
        "fold-laundry",
        47116,
        "Medium-full view standing beside a bed and folding a blue cotton shirt with both hands, "
        "gray pajama top and dark lounge pants, calm downward gaze, morning window light.",
    ),
    PilotCase(
        "strict-profile-neutral",
        47117,
        "Shoulders-up strict left-profile portrait beside a window, loose dark brown hair tucked "
        "behind one ear, neutral mouth and relaxed eye, realistic skin texture.",
    ),
    PilotCase(
        "over-shoulder-gaze",
        47118,
        "Chest-up rear three-quarter portrait turning to look over her shoulder toward the "
        "camera, dark blue blouse, loose hair, calm closed-mouth expression, soft daylight.",
    ),
    PilotCase(
        "broad-toothy-laugh",
        47119,
        "Chest-up candid portrait during a genuine broad laugh with teeth visible, eyes naturally "
        "narrowed and cheeks raised, head tilted slightly back, warm indoor light.",
    ),
    PilotCase(
        "sunlight-squint",
        47120,
        "Chest-up outdoor portrait squinting naturally in bright sunlight while shading her eyes "
        "with one hand, closed-mouth half-smile, realistic facial creases.",
    ),
    PilotCase(
        "eyes-closed-calm",
        47121,
        "Shoulders-up portrait with both eyes fully closed and a peaceful neutral expression, "
        "face angled slightly upward toward soft window light, no smile.",
    ),
    PilotCase(
        "concerned-frown",
        47122,
        "Chest-up candid portrait reading a concerning message on her phone, subtle natural "
        "frown with knitted brows and compressed lips, gaze directed downward.",
    ),
    PilotCase(
        "pleasant-surprise",
        47123,
        "Chest-up candid portrait reacting with pleasant surprise, raised eyebrows, widened eyes "
        "and slightly open mouth, one hand touching her upper chest, ordinary indoor light.",
    ),
    PilotCase(
        "windblown-three-quarter",
        47124,
        "Chest-up outdoor three-quarter portrait in a light breeze, loose hair moving naturally "
        "across one cheek, looking past the camera with relaxed eyes and neutral lips.",
    ),
)


SUITES = {
    "everyday": CASES,
    "body-expression": BODY_EXPRESSION_CASES,
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def embedding_cache_path(cache_root: Path, prompt: str, max_length: int) -> Path:
    digest = hashlib.sha256(f"{MODEL_REVISION}\0{max_length}\0{prompt}".encode()).hexdigest()
    return cache_root / "prompt-embeddings" / f"{digest}.pt"


def validate_inputs(
    model_root: Path,
    text_encoder_root: Path,
    fp8_transformer_root: Path,
    references: list[Path],
) -> None:
    model_index = model_root / "model_index.json"
    text_encoder_index = text_encoder_root / "model.safetensors.index.json"
    text_encoder_manifest = text_encoder_root / "mara-reshard.json"
    vae_weights = model_root / "vae" / "diffusion_pytorch_model.safetensors"
    fp8_config = fp8_transformer_root / "config.json"
    fp8_checkpoint = fp8_transformer_root / "model_fp8_static.pt"
    required = (
        model_index,
        text_encoder_index,
        text_encoder_manifest,
        vae_weights,
        fp8_config,
        fp8_checkpoint,
    )
    missing = [str(path) for path in (*required, *references) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required pilot files: {', '.join(missing)}")
    config_hash = sha256_file(fp8_config)
    if config_hash != FP8_CONFIG_SHA256:
        raise ValueError(
            f"FLUX.2 FP8 config hash mismatch: expected {FP8_CONFIG_SHA256}, found {config_hash}"
        )
    transformer_hash = sha256_file(fp8_checkpoint)
    if transformer_hash != FP8_TRANSFORMER_SHA256:
        raise ValueError(
            "FLUX.2 FP8 transformer hash mismatch: "
            f"expected {FP8_TRANSFORMER_SHA256}, found {transformer_hash}"
        )


def encode_missing_prompts(
    cases: tuple[PilotCase, ...],
    model_root: Path,
    text_encoder_root: Path,
    cache_root: Path,
    max_length: int,
) -> dict[str, Path]:
    import torch

    cache_paths = {
        case.case_id: embedding_cache_path(cache_root, case.prompt, max_length) for case in cases
    }
    missing = [case for case in cases if not cache_paths[case.case_id].is_file()]
    if not missing:
        print("Reusing cached FLUX.2 prompt embeddings.", flush=True)
        return cache_paths

    from diffusers import Flux2KleinPipeline
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3ForCausalLM

    print(f"Loading Qwen3 text encoder for {len(missing)} prompt(s)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_root / "tokenizer", local_files_only=True)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
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

    try:
        for index, case in enumerate(missing, start=1):
            print(f"Encoding prompt {index}/{len(missing)}: {case.case_id}", flush=True)
            with torch.inference_mode():
                prompt_embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    prompt=case.prompt,
                    device=torch.device("cuda"),
                    max_sequence_length=max_length,
                    hidden_states_layers=(9, 18, 27),
                ).cpu()
            destination = cache_paths[case.case_id]
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            torch.save(prompt_embeds, temporary)
            os.replace(temporary, destination)
            del prompt_embeds
    finally:
        del text_encoder, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    return cache_paths


def load_generation_pipeline(model_root: Path, fp8_transformer_root: Path):
    import torch
    from diffusers import (
        AutoencoderKLFlux2,
        FlowMatchEulerDiscreteScheduler,
        Flux2KleinPipeline,
        Flux2Transformer2DModel,
    )

    installed_torchao = version("torchao")
    if installed_torchao != TORCHAO_VERSION:
        raise RuntimeError(
            f"FP8 checkpoint requires torchao {TORCHAO_VERSION}; found {installed_torchao}"
        )

    print("Loading memory-mapped FLUX.2 FP8 transformer...", flush=True)
    # The legacy TorchAO archive needs pickle, so validate its pinned SHA-256 first.
    checkpoint = torch.load(
        fp8_transformer_root / "model_fp8_static.pt",
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    required_keys = {"state_dict", "act_scales", "fp8_dtype"}
    if not isinstance(checkpoint, dict) or not required_keys.issubset(checkpoint):
        found_keys = checkpoint.keys() if isinstance(checkpoint, dict) else ()
        raise ValueError(f"FP8 checkpoint is missing keys: {sorted(required_keys - found_keys)}")
    config = json.loads((fp8_transformer_root / "config.json").read_text(encoding="utf-8"))
    with torch.device("meta"):
        transformer = Flux2Transformer2DModel.from_config(config)
    transformer.load_state_dict(checkpoint["state_dict"], strict=True, assign=True)
    transformer.eval().to("cuda")
    del checkpoint
    gc.collect()

    print("Loading FLUX.2 VAE...", flush=True)
    vae = AutoencoderKLFlux2.from_pretrained(
        model_root / "vae",
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_root / "scheduler", local_files_only=True
    )
    pipe = Flux2KleinPipeline(
        scheduler=scheduler,
        vae=vae,
        transformer=transformer,
        text_encoder=None,
        tokenizer=None,
        is_distilled=True,
    )
    pipe.vae.enable_tiling()
    return pipe


def run(args: argparse.Namespace) -> Path:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("FLUX.2 pilot requires CUDA")

    model_root = args.model_root.resolve()
    text_encoder_root = args.text_encoder_root.resolve()
    fp8_transformer_root = args.fp8_transformer_root.resolve()
    references = [path.resolve() for path in args.reference]
    suite_cases = SUITES[args.suite]
    selected_cases = suite_cases[: args.limit]
    validate_inputs(model_root, text_encoder_root, fp8_transformer_root, references)

    run_dir = (args.output_root / args.run_id).resolve()
    if run_dir.exists():
        raise FileExistsError(f"pilot run already exists: {run_dir}")
    output_dir = run_dir / "outputs"
    sheet_path = run_dir / "contact-sheets" / "pilot.png"
    output_dir.mkdir(parents=True)

    started_at = utc_now()
    status = {
        "schema_version": 1,
        "state": "running",
        "run_id": args.run_id,
        "suite": args.suite,
        "started_at": started_at,
        "finished_at": None,
        "generated_count": 0,
        "error": None,
    }
    atomic_write_json(run_dir / "status.json", status)
    start = time.perf_counter()
    records: list[SheetRecord] = []
    manifest_path = output_dir / "manifest.jsonl"

    try:
        cache_paths = encode_missing_prompts(
            selected_cases,
            model_root,
            text_encoder_root,
            args.cache_root.resolve(),
            args.max_sequence_length,
        )
        torch.cuda.reset_peak_memory_stats()
        pipe = load_generation_pipeline(model_root, fp8_transformer_root)
        reference_images: list[Image.Image] = []
        for path in references:
            with Image.open(path) as opened:
                reference_images.append(opened.convert("RGB").copy())

        for sequence, case in enumerate(selected_cases, start=1):
            print(f"Generating {sequence}/{len(selected_cases)}: {case.case_id}", flush=True)
            prompt_embeds = torch.load(
                cache_paths[case.case_id], map_location="cuda", weights_only=True
            )
            generator = torch.Generator(device="cuda").manual_seed(case.seed)
            case_start = time.perf_counter()
            with torch.inference_mode():
                image = pipe(
                    image=reference_images,
                    prompt=None,
                    prompt_embeds=prompt_embeds,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    max_sequence_length=args.max_sequence_length,
                ).images[0]
            wall_seconds = time.perf_counter() - case_start
            artifact_id = f"flux2-{case.case_id}-seed-{case.seed}"
            destination = output_dir / f"{artifact_id}.png"
            save_png_atomic(image, destination)
            record = SheetRecord(
                artifact_id=artifact_id,
                sequence=sequence,
                seed=case.seed,
                width=image.width,
                height=image.height,
                image_path=destination.relative_to(run_dir).as_posix(),
            )
            records.append(record)
            append_jsonl(
                manifest_path,
                {
                    **asdict(record),
                    "schema_version": 1,
                    "prompt": case.prompt,
                    "sha256": sha256_file(destination),
                    "wall_seconds": round(wall_seconds, 3),
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "fp8_transformer_id": FP8_TRANSFORMER_ID,
                    "fp8_transformer_revision": FP8_TRANSFORMER_REVISION,
                    "fp8_transformer_sha256": FP8_TRANSFORMER_SHA256,
                    "prompt_embedding_sha256": sha256_file(cache_paths[case.case_id]),
                    "reference_paths": [path.as_posix() for path in references],
                    "reference_sha256": [sha256_file(path) for path in references],
                },
            )
            status["generated_count"] = len(records)
            atomic_write_json(run_dir / "status.json", status)
            del image, prompt_embeds
            torch.cuda.empty_cache()

        create_contact_sheet(
            run_dir,
            records,
            sheet_path,
            columns=min(5, len(records)),
            thumbnail_width=190,
            labeler=lambda record: record.artifact_id.removeprefix("flux2-").split("-seed-")[0],
        )
        environment = {
            "schema_version": 1,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "fp8_transformer_id": FP8_TRANSFORMER_ID,
            "fp8_transformer_revision": FP8_TRANSFORMER_REVISION,
            "fp8_transformer_sha256": FP8_TRANSFORMER_SHA256,
            "fp8_config_sha256": FP8_CONFIG_SHA256,
            "torchao_version": version("torchao"),
            "vae_sha256": sha256_file(model_root / "vae" / "diffusion_pytorch_model.safetensors"),
            "text_encoder_index_sha256": sha256_file(
                text_encoder_root / "model.safetensors.index.json"
            ),
            "text_encoder_reshard_manifest_sha256": sha256_file(
                text_encoder_root / "mara-reshard.json"
            ),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "max_sequence_length": args.max_sequence_length,
            "suite": args.suite,
            "peak_reserved_vram_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 4),
            "wall_seconds": round(time.perf_counter() - start, 3),
        }
        atomic_write_json(run_dir / "environment.json", environment)
        status.update(
            state="completed",
            finished_at=utc_now(),
            generated_count=len(records),
        )
        atomic_write_json(run_dir / "status.json", status)
        print(f"Completed {len(records)} FLUX.2 pilot image(s).", flush=True)
        print(f"Run: {run_dir}", flush=True)
        print(f"Contact sheet: {sheet_path}", flush=True)
        print(f"Peak reserved VRAM: {environment['peak_reserved_vram_gib']:.2f} GiB", flush=True)
        return run_dir
    except Exception as exc:
        status.update(state="failed", finished_at=utc_now(), error=f"{type(exc).__name__}: {exc}")
        atomic_write_json(run_dir / "status.json", status)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the isolated Mara FLUX.2 Klein pilot.")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--cache-root", type=Path, default=Path(".cache/flux2-klein-pilot"))
    parser.add_argument(
        "--text-encoder-root",
        type=Path,
        default=DEFAULT_TEXT_ENCODER_ROOT,
    )
    parser.add_argument(
        "--fp8-transformer-root",
        type=Path,
        default=DEFAULT_FP8_TRANSFORMER_ROOT,
    )
    parser.add_argument("--output-root", type=Path, default=Path("experiments"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        action="append",
        default=None,
        help="Repeat for each identity reference image.",
    )
    parser.add_argument("--suite", choices=tuple(SUITES), default="everyday")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=832)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    args = parser.parse_args()
    if args.reference is None:
        args.reference = [
            Path("characters/mara/canonical/v001/master.png"),
            Path("characters/mara/canonical/v001/backup.png"),
        ]
    suite_size = len(SUITES[args.suite])
    if not 1 <= args.limit <= suite_size:
        parser.error(f"--limit must be between 1 and {suite_size} for suite {args.suite!r}")
    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception as error:
        print(f"Error: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
