from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mara_lab.config import ModelProfile
from mara_lab.errors import ModelMaterializationError
from mara_lab.model_materialization import materialize_diffusers_snapshot


@dataclass(frozen=True)
class PrepareModelSummary:
    output_dir: Path
    manifest_path: Path
    shard_count: int


def prepare_diffusers_model(
    model: ModelProfile, *, local_files_only: bool = False
) -> PrepareModelSummary:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ModelMaterializationError(
            "GPU dependencies are missing. Run `uv sync --extra gpu`."
        ) from exc

    revision = model.resolved_revision or model.revision
    allow_patterns = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model.fp16.safetensors",
        "text_encoder_2/config.json",
        "text_encoder_2/model.fp16.safetensors",
        "tokenizer/*",
        "tokenizer_2/*",
        "unet/config.json",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.fp16.safetensors",
    ]
    try:
        snapshot = Path(
            snapshot_download(
                model.model_id,
                revision=revision,
                cache_dir=model.cache_dir.resolve(),
                local_files_only=local_files_only,
                allow_patterns=allow_patterns,
            )
        )
    except Exception as exc:
        mode = "local model cache" if local_files_only else "Hugging Face"
        raise ModelMaterializationError(f"cannot load model components from {mode}: {exc}") from exc

    model_slug = "models--" + model.model_id.replace("/", "--")
    shard_mib = model.materialized_max_shard_mib
    output_dir = (
        model.materialized_cache_dir.resolve()
        / model_slug
        / "snapshots"
        / f"{revision}-fp16-{shard_mib}m"
    )
    result = materialize_diffusers_snapshot(
        snapshot,
        output_dir,
        model_id=model.model_id,
        revision=revision,
        max_shard_bytes=shard_mib * 1024 * 1024,
    )
    return PrepareModelSummary(
        output_dir=result.output_dir,
        manifest_path=result.manifest_path,
        shard_count=sum(len(component.shards) for component in result.component_results),
    )
