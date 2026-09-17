from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from mara_lab.errors import ModelMaterializationError
from mara_lab.model_materialization import (
    read_safetensors_layout,
    reshard_safetensors_checkpoint,
    shard_safetensors,
)


def _write_tensors(path: Path, tensors: dict[str, bytes]) -> None:
    cursor = 0
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    for name, payload in tensors.items():
        header[name] = {
            "dtype": "U8",
            "shape": [len(payload)],
            "data_offsets": [cursor, cursor + len(payload)],
        }
        cursor += len(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for payload in tensors.values():
            handle.write(payload)


def _write_fixture(path: Path) -> dict[str, bytes]:
    tensors = {
        "first": b"abcdefgh",
        "second": b"ijklmnopqrst",
        "third": b"uvwxyz0123456789",
    }
    _write_tensors(path, tensors)
    return tensors


def _tensor_payloads(path: Path) -> dict[str, bytes]:
    layout = read_safetensors_layout(path)
    payloads: dict[str, bytes] = {}
    with path.open("rb") as handle:
        for tensor in layout.tensors:
            handle.seek(layout.data_start + tensor.start)
            payloads[tensor.name] = handle.read(tensor.size)
    return payloads


def test_shards_without_changing_tensor_payloads(tmp_path: Path) -> None:
    source = tmp_path / "model.fp16.safetensors"
    expected = _write_fixture(source)

    result = shard_safetensors(source, tmp_path / "output", max_shard_bytes=16, variant="fp16")

    assert len(result.shards) == 3
    assert result.index_path.name == "model.safetensors.index.fp16.json"
    assert result.tensor_count == 3
    assert result.total_size == sum(map(len, expected.values()))
    index = json.loads(result.index_path.read_text(encoding="utf-8"))
    assert index["metadata"]["total_size"] == result.total_size
    actual: dict[str, bytes] = {}
    for shard in result.shards:
        layout = read_safetensors_layout(shard)
        assert layout.metadata == {"format": "pt"}
        actual.update(_tensor_payloads(shard))
    assert actual == expected
    assert set(index["weight_map"]) == set(expected)


def test_rejects_non_contiguous_payload(tmp_path: Path) -> None:
    source = tmp_path / "broken.safetensors"
    header = {
        "value": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]},
    }
    encoded = json.dumps(header).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    source.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"xx")

    with pytest.raises(ModelMaterializationError, match="not contiguous"):
        read_safetensors_layout(source)


def test_reshards_indexed_checkpoint_without_decoding_payloads(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    first = {"layer.0": b"abcdefgh", "layer.1": b"ijklmnopqrst"}
    second = {"layer.2": b"uvwxyz0123456789", "layer.3": b"ABCDEFGHIJKLMNOP"}
    first_name = "model-00001-of-00002.safetensors"
    second_name = "model-00002-of-00002.safetensors"
    _write_tensors(source / first_name, first)
    _write_tensors(source / second_name, second)
    expected = {**first, **second}
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(map(len, expected.values()))},
                "weight_map": {
                    **dict.fromkeys(first, first_name),
                    **dict.fromkeys(second, second_name),
                },
            }
        ),
        encoding="utf-8",
    )
    (source / "config.json").write_text('{"model_type": "fixture"}\n', encoding="utf-8")

    result = reshard_safetensors_checkpoint(
        source,
        tmp_path / "output",
        max_shard_bytes=16,
        auxiliary_files=(Path("config.json"),),
    )

    assert result.tensor_count == len(expected)
    assert result.total_size == sum(map(len, expected.values()))
    assert len(result.shards) == 4
    assert (result.output_dir / "config.json").read_text(encoding="utf-8") == (
        '{"model_type": "fixture"}\n'
    )
    index = json.loads(result.index_path.read_text(encoding="utf-8"))
    assert set(index["weight_map"]) == set(expected)
    actual: dict[str, bytes] = {}
    for shard in result.shards:
        actual.update(_tensor_payloads(shard))
    assert actual == expected
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["tensor_count"] == len(expected)
    assert len(manifest["source_shards"]) == 2
    assert len(manifest["output_shards"]) == 4
