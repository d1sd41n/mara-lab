from pathlib import Path

from mara_lab.artifacts import (
    append_jsonl,
    atomic_write_json,
    environment_snapshot,
    read_jsonl,
    sha256_file,
)


def test_atomic_json_and_append_only_manifest(tmp_path: Path) -> None:
    document = tmp_path / "document.json"
    manifest = tmp_path / "manifest.jsonl"

    atomic_write_json(document, {"b": 2, "a": 1})
    append_jsonl(manifest, {"sequence": 1})
    append_jsonl(manifest, {"sequence": 2})

    assert document.read_text(encoding="utf-8").startswith('{\n  "a": 1')
    assert read_jsonl(manifest) == [{"sequence": 1}, {"sequence": 2}]
    assert len(sha256_file(document)) == 64

    environment = environment_snapshot([], lockfile=document)
    assert environment["lockfile"]["sha256"] == sha256_file(document)
