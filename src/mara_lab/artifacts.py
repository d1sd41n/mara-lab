from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def default_run_id(experiment_id: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{experiment_id}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def atomic_write_yaml(path: Path, data: Any) -> None:
    content = yaml.safe_dump(data, sort_keys=False, allow_unicode=False)
    atomic_write_text(path, content)


def append_jsonl(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(data, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_png_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    image.save(temporary, format="PNG", optimize=False)
    os.replace(temporary, path)


def environment_snapshot(
    packages: Iterable[str], lockfile: Path | None = Path("uv.lock")
) -> dict[str, Any]:
    package_versions: dict[str, str | None] = {}
    for package in packages:
        try:
            package_versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            package_versions[package] = None
    lockfile_record: dict[str, str | None] | None = None
    if lockfile is not None:
        resolved_lockfile = lockfile.resolve()
        lockfile_record = {
            "path": resolved_lockfile.as_posix(),
            "sha256": sha256_file(resolved_lockfile) if resolved_lockfile.is_file() else None,
        }
    return {
        "captured_at": utc_now(),
        "lockfile": lockfile_record,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "packages": package_versions,
    }
