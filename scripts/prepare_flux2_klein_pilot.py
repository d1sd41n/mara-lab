from __future__ import annotations

import argparse
import json
from pathlib import Path

from mara_lab.artifacts import sha256_file
from mara_lab.errors import ModelMaterializationError
from mara_lab.flux2_klein import (
    DEFAULT_FP8_MODEL_ROOT,
    DEFAULT_MODEL_ROOT,
    DEFAULT_TEXT_ENCODER_ROOT,
    FP8_CONFIG_SHA256,
    FP8_FILES,
    FP8_TRANSFORMER_ID,
    FP8_TRANSFORMER_REVISION,
    FP8_TRANSFORMER_SHA256,
    MODEL_ALLOW_PATTERNS,
    MODEL_ID,
    MODEL_REVISION,
    TEXT_ENCODER_MAX_SHARD_BYTES,
)
from mara_lab.model_materialization import reshard_safetensors_checkpoint


def _source_model_is_complete(model_root: Path) -> bool:
    required = (
        model_root / "model_index.json",
        model_root / "scheduler" / "scheduler_config.json",
        model_root / "text_encoder" / "config.json",
        model_root / "text_encoder" / "model.safetensors.index.json",
        model_root / "tokenizer" / "tokenizer.json",
        model_root / "vae" / "config.json",
        model_root / "vae" / "diffusion_pytorch_model.safetensors",
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        index = json.loads(required[3].read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    weight_map = index.get("weight_map", {}) if isinstance(index, dict) else {}
    return (
        isinstance(weight_map, dict)
        and bool(weight_map)
        and all(isinstance(filename, str) for filename in weight_map.values())
        and all(
            (model_root / "text_encoder" / filename).is_file()
            for filename in set(weight_map.values())
        )
    )


def _download_sources(
    model_root: Path,
    fp8_model_root: Path,
    *,
    local_files_only: bool,
) -> None:
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError("GPU dependencies are missing; run bootstrap.ps1 -Gpu") from exc

    if not _source_model_is_complete(model_root):
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=model_root,
            allow_patterns=list(MODEL_ALLOW_PATTERNS),
            local_files_only=local_files_only,
        )
    for filename in FP8_FILES:
        if not (fp8_model_root / filename).is_file():
            hf_hub_download(
                repo_id=FP8_TRANSFORMER_ID,
                revision=FP8_TRANSFORMER_REVISION,
                filename=filename,
                local_dir=fp8_model_root,
                local_files_only=local_files_only,
            )


def _validate_fp8(fp8_model_root: Path) -> None:
    config = fp8_model_root / "transformer_fp8_static" / "config.json"
    checkpoint = fp8_model_root / "transformer_fp8_static" / "model_fp8_static.pt"
    if sha256_file(config) != FP8_CONFIG_SHA256:
        raise ModelMaterializationError("downloaded FLUX.2 FP8 config hash does not match")
    if sha256_file(checkpoint) != FP8_TRANSFORMER_SHA256:
        raise ModelMaterializationError("downloaded FLUX.2 FP8 checkpoint hash does not match")


def _validate_existing_text_encoder(source_dir: Path, output_dir: Path) -> int:
    manifest_path = output_dir / "mara-reshard.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelMaterializationError(
            f"existing text encoder has no valid reshard manifest: {output_dir}"
        ) from exc
    expected_source_hash = sha256_file(source_dir / "model.safetensors.index.json")
    if manifest.get("source_index_sha256") != expected_source_hash:
        raise ModelMaterializationError("existing text encoder came from a different source index")
    if manifest.get("max_shard_bytes") != TEXT_ENCODER_MAX_SHARD_BYTES:
        raise ModelMaterializationError("existing text encoder uses a different shard size")
    output_index = output_dir / "model.safetensors.index.json"
    if not output_index.is_file() or sha256_file(output_index) != manifest.get(
        "output_index_sha256"
    ):
        raise ModelMaterializationError("existing text encoder index is invalid")
    output_shards = manifest.get("output_shards")
    if not isinstance(output_shards, list) or not output_shards:
        raise ModelMaterializationError("existing text encoder manifest has no output shards")
    for record in output_shards:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ModelMaterializationError("existing text encoder manifest is malformed")
        relative = Path(record["path"])
        if relative.is_absolute() or len(relative.parts) != 1:
            raise ModelMaterializationError("existing text encoder manifest has an unsafe path")
        path = output_dir / relative
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ModelMaterializationError(f"existing text encoder shard is invalid: {path}")
    tensor_count = manifest.get("tensor_count")
    if not isinstance(tensor_count, int) or tensor_count < 1:
        raise ModelMaterializationError("existing text encoder tensor count is invalid")
    return tensor_count


def prepare(
    model_root: Path,
    fp8_model_root: Path,
    text_encoder_root: Path,
    *,
    local_files_only: bool,
) -> None:
    model_root = model_root.resolve()
    fp8_model_root = fp8_model_root.resolve()
    text_encoder_root = text_encoder_root.resolve()
    _download_sources(model_root, fp8_model_root, local_files_only=local_files_only)
    if not _source_model_is_complete(model_root):
        raise ModelMaterializationError("FLUX.2 source snapshot is incomplete")
    _validate_fp8(fp8_model_root)

    source_text_encoder = model_root / "text_encoder"
    if text_encoder_root.exists():
        tensor_count = _validate_existing_text_encoder(source_text_encoder, text_encoder_root)
        print(f"Reusing verified text encoder with {tensor_count} tensors: {text_encoder_root}")
        return

    result = reshard_safetensors_checkpoint(
        source_text_encoder,
        text_encoder_root,
        max_shard_bytes=TEXT_ENCODER_MAX_SHARD_BYTES,
        auxiliary_files=(Path("config.json"), Path("generation_config.json")),
    )
    print(
        f"Prepared {result.tensor_count} text-encoder tensors in "
        f"{len(result.shards)} shards: {result.output_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the pinned FLUX.2 Klein pilot files.")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--fp8-model-root", type=Path, default=DEFAULT_FP8_MODEL_ROOT)
    parser.add_argument("--text-encoder-root", type=Path, default=DEFAULT_TEXT_ENCODER_ROOT)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    prepare(
        arguments.model_root,
        arguments.fp8_model_root,
        arguments.text_encoder_root,
        local_files_only=arguments.local_files_only,
    )
