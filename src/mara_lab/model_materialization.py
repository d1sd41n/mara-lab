from __future__ import annotations

import json
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from mara_lab.artifacts import atomic_write_json, sha256_file
from mara_lab.errors import ModelMaterializationError

_HEADER_PREFIX_BYTES = 8
_MAX_HEADER_BYTES = 100 * 1024 * 1024
_COPY_CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class TensorRecord:
    name: str
    dtype: str
    shape: list[int]
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class SafeTensorLayout:
    metadata: dict[str, str]
    tensors: list[TensorRecord]
    data_start: int
    payload_size: int


@dataclass(frozen=True)
class ShardResult:
    source_path: Path
    source_sha256: str
    index_path: Path
    shards: list[Path]
    tensor_count: int
    total_size: int


@dataclass(frozen=True)
class MaterializationResult:
    output_dir: Path
    manifest_path: Path
    component_results: list[ShardResult]


@dataclass(frozen=True)
class CheckpointReshardResult:
    output_dir: Path
    index_path: Path
    manifest_path: Path
    shards: list[Path]
    tensor_count: int
    total_size: int


def read_safetensors_layout(path: Path) -> SafeTensorLayout:
    """Read and validate a safetensors header without mapping its tensor payload."""
    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            prefix = handle.read(_HEADER_PREFIX_BYTES)
            if len(prefix) != _HEADER_PREFIX_BYTES:
                raise ModelMaterializationError(f"safetensors header is truncated: {path}")
            (header_size,) = struct.unpack("<Q", prefix)
            if header_size < 2 or header_size > _MAX_HEADER_BYTES:
                raise ModelMaterializationError(
                    f"safetensors header size is invalid ({header_size} bytes): {path}"
                )
            if _HEADER_PREFIX_BYTES + header_size > file_size:
                raise ModelMaterializationError(f"safetensors header exceeds file size: {path}")
            encoded_header = handle.read(header_size)
    except OSError as exc:
        raise ModelMaterializationError(f"cannot read safetensors file {path}: {exc}") from exc

    try:
        raw_header = json.loads(encoded_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelMaterializationError(
            f"invalid safetensors JSON header in {path}: {exc}"
        ) from exc
    if not isinstance(raw_header, dict):
        raise ModelMaterializationError(f"safetensors header must be an object: {path}")

    raw_metadata = raw_header.pop("__metadata__", {})
    if not isinstance(raw_metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in raw_metadata.items()
    ):
        raise ModelMaterializationError(f"invalid safetensors metadata in {path}")

    payload_size = file_size - (_HEADER_PREFIX_BYTES + header_size)
    tensors: list[TensorRecord] = []
    spans: list[tuple[int, int, str]] = []
    for name, raw_tensor in raw_header.items():
        if not isinstance(name, str) or not isinstance(raw_tensor, dict):
            raise ModelMaterializationError(f"invalid tensor entry in {path}")
        dtype = raw_tensor.get("dtype")
        shape = raw_tensor.get("shape")
        offsets = raw_tensor.get("data_offsets")
        if (
            not isinstance(dtype, str)
            or not isinstance(shape, list)
            or not all(isinstance(item, int) and item >= 0 for item in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(item, int) for item in offsets)
        ):
            raise ModelMaterializationError(f"invalid tensor metadata for {name!r} in {path}")
        start, end = offsets
        if start < 0 or end < start or end > payload_size:
            raise ModelMaterializationError(f"invalid tensor offsets for {name!r} in {path}")
        tensors.append(TensorRecord(name, dtype, shape, start, end))
        spans.append((start, end, name))

    if not tensors:
        raise ModelMaterializationError(f"safetensors file contains no tensors: {path}")
    spans.sort()
    cursor = 0
    for start, end, name in spans:
        if start != cursor:
            raise ModelMaterializationError(
                f"tensor payload is not contiguous before {name!r} in {path}"
            )
        cursor = end
    if cursor != payload_size:
        raise ModelMaterializationError(f"tensor payload does not fill file: {path}")

    return SafeTensorLayout(
        metadata=raw_metadata,
        tensors=tensors,
        data_start=_HEADER_PREFIX_BYTES + header_size,
        payload_size=payload_size,
    )


def _partition_tensors(
    tensors: list[TensorRecord], max_shard_bytes: int
) -> list[list[TensorRecord]]:
    if max_shard_bytes < 1:
        raise ValueError("max_shard_bytes must be positive")
    groups: list[list[TensorRecord]] = []
    current: list[TensorRecord] = []
    current_size = 0
    for tensor in tensors:
        if current and current_size + tensor.size > max_shard_bytes:
            groups.append(current)
            current = []
            current_size = 0
        current.append(tensor)
        current_size += tensor.size
    if current:
        groups.append(current)
    return groups


def _encoded_header(metadata: dict[str, str], tensors: list[TensorRecord]) -> bytes:
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = metadata
    cursor = 0
    for tensor in tensors:
        header[tensor.name] = {
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "data_offsets": [cursor, cursor + tensor.size],
        }
        cursor += tensor.size
    encoded = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    return encoded


def _copy_exact_range(source: BinaryIO, target: BinaryIO, start: int, size: int) -> None:
    source.seek(start)
    remaining = size
    while remaining:
        chunk = source.read(min(remaining, _COPY_CHUNK_BYTES))
        if not chunk:
            raise ModelMaterializationError("source tensor payload ended unexpectedly")
        target.write(chunk)
        remaining -= len(chunk)


def shard_safetensors(
    source_path: Path,
    output_dir: Path,
    *,
    max_shard_bytes: int,
    variant: str | None = None,
) -> ShardResult:
    """Split one safetensors file by copying tensor bytes with bounded memory use."""
    source_path = source_path.resolve()
    output_dir = output_dir.resolve()
    layout = read_safetensors_layout(source_path)
    groups = _partition_tensors(layout.tensors, max_shard_bytes)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source_path.name.removesuffix(".safetensors")
    shard_count = len(groups)
    shard_paths: list[Path] = []
    weight_map: dict[str, str] = {}

    try:
        with source_path.open("rb") as source:
            for sequence, tensors in enumerate(groups, start=1):
                shard_name = f"{stem}-{sequence:05d}-of-{shard_count:05d}.safetensors"
                shard_path = output_dir / shard_name
                temporary = shard_path.with_name(f".{shard_path.name}.tmp")
                encoded_header = _encoded_header(layout.metadata, tensors)
                try:
                    with temporary.open("wb") as target:
                        target.write(struct.pack("<Q", len(encoded_header)))
                        target.write(encoded_header)
                        for tensor in tensors:
                            _copy_exact_range(
                                source,
                                target,
                                layout.data_start + tensor.start,
                                tensor.size,
                            )
                            weight_map[tensor.name] = shard_name
                        target.flush()
                        os.fsync(target.fileno())
                    os.replace(temporary, shard_path)
                finally:
                    temporary.unlink(missing_ok=True)
                shard_paths.append(shard_path)
    except OSError as exc:
        raise ModelMaterializationError(f"cannot write model shards: {exc}") from exc

    if variant is None:
        index_name = f"{source_path.name}.index.json"
    else:
        variant_suffix = f".{variant}.safetensors"
        if not source_path.name.endswith(variant_suffix):
            raise ModelMaterializationError(
                f"source filename does not contain the {variant!r} variant: {source_path.name}"
            )
        base_name = source_path.name.removesuffix(variant_suffix)
        index_name = f"{base_name}.safetensors.index.{variant}.json"
        (output_dir / f"{source_path.name}.index.json").unlink(missing_ok=True)
    index_path = output_dir / index_name
    atomic_write_json(
        index_path,
        {
            "metadata": {"total_size": layout.payload_size},
            "weight_map": weight_map,
        },
    )
    return ShardResult(
        source_path=source_path,
        source_sha256=sha256_file(source_path),
        index_path=index_path,
        shards=shard_paths,
        tensor_count=len(layout.tensors),
        total_size=layout.payload_size,
    )


def _read_safetensors_index(path: Path) -> tuple[int, dict[str, str]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelMaterializationError(f"cannot read safetensors index {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelMaterializationError(f"safetensors index must be an object: {path}")

    metadata = value.get("metadata")
    weight_map = value.get("weight_map")
    total_size = metadata.get("total_size") if isinstance(metadata, dict) else None
    if not isinstance(total_size, int) or total_size < 1:
        raise ModelMaterializationError(f"safetensors index has invalid total_size: {path}")
    if (
        not isinstance(weight_map, dict)
        or not weight_map
        or not all(
            isinstance(key, str) and isinstance(filename, str)
            for key, filename in weight_map.items()
        )
    ):
        raise ModelMaterializationError(f"safetensors index has invalid weight_map: {path}")
    return total_size, weight_map


def reshard_safetensors_checkpoint(
    source_dir: Path,
    output_dir: Path,
    *,
    max_shard_bytes: int,
    auxiliary_files: tuple[Path, ...] = (),
) -> CheckpointReshardResult:
    """Re-shard an indexed safetensors checkpoint without decoding tensor payloads."""
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if max_shard_bytes < 1:
        raise ValueError("max_shard_bytes must be positive")
    if source_dir == output_dir or source_dir in output_dir.parents:
        raise ModelMaterializationError("re-sharded checkpoint must be outside its source")
    if output_dir.exists():
        raise ModelMaterializationError(f"re-sharded checkpoint already exists: {output_dir}")

    source_index_path = source_dir / "model.safetensors.index.json"
    total_size, source_weight_map = _read_safetensors_index(source_index_path)
    source_filenames = sorted(set(source_weight_map.values()))
    for filename in source_filenames:
        relative = Path(filename)
        if relative.is_absolute() or len(relative.parts) != 1 or relative.suffix != ".safetensors":
            raise ModelMaterializationError(
                f"safetensors index contains an unsafe shard path: {filename!r}"
            )
        if not (source_dir / relative).is_file():
            raise ModelMaterializationError(f"source checkpoint shard is missing: {filename}")
    for relative in auxiliary_files:
        if relative.is_absolute() or ".." in relative.parts:
            raise ModelMaterializationError(f"unsafe auxiliary path: {relative}")
        if not (source_dir / relative).is_file():
            raise ModelMaterializationError(f"auxiliary checkpoint file is missing: {relative}")

    output_dir.mkdir(parents=True, exist_ok=False)
    combined_weight_map: dict[str, str] = {}
    shards: list[Path] = []
    source_records: list[dict[str, object]] = []
    try:
        for filename in source_filenames:
            result = shard_safetensors(
                source_dir / filename,
                output_dir,
                max_shard_bytes=max_shard_bytes,
            )
            _, generated_weight_map = _read_safetensors_index(result.index_path)
            result.index_path.unlink()
            for key, generated_filename in generated_weight_map.items():
                if source_weight_map.get(key) != filename:
                    raise ModelMaterializationError(
                        f"source index maps tensor {key!r} to the wrong shard"
                    )
                if key in combined_weight_map:
                    raise ModelMaterializationError(f"duplicate tensor in source checkpoint: {key}")
                combined_weight_map[key] = generated_filename
            shards.extend(result.shards)
            source_records.append(
                {
                    "path": filename,
                    "sha256": result.source_sha256,
                    "tensor_count": result.tensor_count,
                    "tensor_payload_bytes": result.total_size,
                }
            )

        if set(combined_weight_map) != set(source_weight_map):
            missing = sorted(set(source_weight_map) - set(combined_weight_map))
            raise ModelMaterializationError(
                f"re-sharded checkpoint is missing {len(missing)} tensor(s)"
            )
        actual_total_size = sum(record["tensor_payload_bytes"] for record in source_records)
        if actual_total_size != total_size:
            raise ModelMaterializationError(
                "checkpoint payload size mismatch: "
                f"expected {total_size}, found {actual_total_size}"
            )

        for relative in auxiliary_files:
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_dir / relative, destination)

        index_path = output_dir / "model.safetensors.index.json"
        atomic_write_json(
            index_path,
            {"metadata": {"total_size": total_size}, "weight_map": combined_weight_map},
        )
        manifest_path = output_dir / "mara-reshard.json"
        atomic_write_json(
            manifest_path,
            {
                "schema_version": 1,
                "source_index": source_index_path.as_posix(),
                "source_index_sha256": sha256_file(source_index_path),
                "max_shard_bytes": max_shard_bytes,
                "tensor_count": len(combined_weight_map),
                "tensor_payload_bytes": total_size,
                "source_shards": source_records,
                "output_index_sha256": sha256_file(index_path),
                "output_shards": [
                    {
                        "path": shard.name,
                        "bytes": shard.stat().st_size,
                        "sha256": sha256_file(shard),
                    }
                    for shard in shards
                ],
            },
        )
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise

    return CheckpointReshardResult(
        output_dir=output_dir,
        index_path=index_path,
        manifest_path=manifest_path,
        shards=shards,
        tensor_count=len(combined_weight_map),
        total_size=total_size,
    )


_COMPONENT_WEIGHTS = (
    Path("text_encoder/model.fp16.safetensors"),
    Path("text_encoder_2/model.fp16.safetensors"),
    Path("unet/diffusion_pytorch_model.fp16.safetensors"),
    Path("vae/diffusion_pytorch_model.fp16.safetensors"),
)
_REQUIRED_FILES = (
    Path("model_index.json"),
    Path("scheduler/scheduler_config.json"),
    Path("text_encoder/config.json"),
    Path("text_encoder_2/config.json"),
    Path("unet/config.json"),
    Path("vae/config.json"),
)
_COPY_DIRECTORIES = (Path("tokenizer"), Path("tokenizer_2"))


def materialize_diffusers_snapshot(
    source_dir: Path,
    output_dir: Path,
    *,
    model_id: str,
    revision: str,
    max_shard_bytes: int,
) -> MaterializationResult:
    """Create a minimal, sharded FP16 Diffusers snapshot from official component files."""
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if source_dir == output_dir or source_dir in output_dir.parents:
        raise ModelMaterializationError("materialized model must be outside the source snapshot")

    required = [*_REQUIRED_FILES, *_COMPONENT_WEIGHTS]
    missing = [path.as_posix() for path in required if not (source_dir / path).is_file()]
    if missing:
        raise ModelMaterializationError(
            "source snapshot is incomplete; missing: " + ", ".join(missing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for relative_path in _REQUIRED_FILES:
            destination = output_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_dir / relative_path, destination)
        for relative_dir in _COPY_DIRECTORIES:
            for source_path in sorted((source_dir / relative_dir).glob("*")):
                if source_path.is_file():
                    destination = output_dir / relative_dir / source_path.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, destination)
    except OSError as exc:
        raise ModelMaterializationError(f"cannot copy Diffusers configuration: {exc}") from exc

    results = [
        shard_safetensors(
            source_dir / relative_path,
            output_dir / relative_path.parent,
            max_shard_bytes=max_shard_bytes,
            variant="fp16",
        )
        for relative_path in _COMPONENT_WEIGHTS
    ]
    manifest = {
        "schema_version": 1,
        "model_id": model_id,
        "revision": revision,
        "variant": "fp16",
        "max_shard_bytes": max_shard_bytes,
        "source_snapshot": source_dir.as_posix(),
        "components": [
            {
                "source": result.source_path.relative_to(source_dir).as_posix(),
                "source_sha256": result.source_sha256,
                "tensor_count": result.tensor_count,
                "tensor_payload_bytes": result.total_size,
                "index": result.index_path.relative_to(output_dir).as_posix(),
                "shards": [
                    {
                        "path": shard.relative_to(output_dir).as_posix(),
                        "bytes": shard.stat().st_size,
                        "sha256": sha256_file(shard),
                    }
                    for shard in result.shards
                ],
            }
            for result in results
        ],
    }
    manifest_path = output_dir / "mara-materialization.json"
    atomic_write_json(manifest_path, manifest)
    return MaterializationResult(output_dir, manifest_path, results)
