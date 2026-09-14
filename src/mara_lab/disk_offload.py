from __future__ import annotations

import json
import os
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from mara_lab.errors import BackendUnavailableError


class _QuietProgress:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> _QuietProgress:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def set_postfix(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_description(self, *args: Any, **kwargs: Any) -> None:
        pass

    def update(self, *args: Any, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        pass


@contextmanager
def quiet_accelerate_progress():
    """Suppress the per-tensor progress stream while building a disk store."""
    from accelerate.utils import modeling

    original = modeling.tqdm
    modeling.tqdm = _QuietProgress
    try:
        yield
    finally:
        modeling.tqdm = original


def read_offload_index(offload_dir: Path) -> dict[str, dict[str, Any]]:
    index_path = offload_dir / "index.json"
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendUnavailableError(f"invalid disk-offload index {index_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise BackendUnavailableError(f"disk-offload index is not an object: {index_path}")
    return data


def add_cpu_values_to_store(model: Any, offload_dir: Path) -> None:
    from accelerate.utils import offload_weight, save_offload_index

    index_path = offload_dir / "index.json"
    index = read_offload_index(offload_dir) if index_path.is_file() else {}
    changed = False
    for name, value in [*model.named_parameters(), *model.named_buffers()]:
        if value.device.type != "meta" and name not in index:
            offload_weight(value.detach(), name, offload_dir, index)
            changed = True
    if changed:
        save_offload_index(index, offload_dir)


def prepare_model_store(
    model: Any,
    checkpoint_index: Path,
    offload_dir: Path,
    *,
    anchor: str,
    dtype: Any,
) -> None:
    from accelerate.utils import load_checkpoint_in_model

    offload_dir.mkdir(parents=True, exist_ok=False)
    model.to(dtype=dtype)
    with quiet_accelerate_progress():
        load_checkpoint_in_model(
            model,
            checkpoint_index,
            device_map={anchor: "cpu", "": "disk"},
            offload_folder=offload_dir,
            offload_buffers=True,
            dtype=dtype,
        )
    add_cpu_values_to_store(model, offload_dir)


def write_state_store(state: Mapping[str, Any], model: Any, offload_dir: Path) -> None:
    from accelerate.utils import offload_weight, save_offload_index

    offload_dir.mkdir(parents=True, exist_ok=False)
    index: dict[str, dict[str, Any]] = {}
    expected = {name for name, _value in model.named_parameters()}
    if set(state) != expected:
        missing = sorted(expected - set(state))[:5]
        unexpected = sorted(set(state) - expected)[:5]
        raise BackendUnavailableError(
            f"PhotoMaker ID encoder mismatch: missing={missing}, unexpected={unexpected}"
        )
    for name, value in state.items():
        offload_weight(value, name, offload_dir, index)
    save_offload_index(index, offload_dir)
    add_cpu_values_to_store(model, offload_dir)


def map_photomaker_lora_name(source_name: str) -> str:
    name = source_name.removeprefix("unet.").replace(".processor.", ".")
    name = name.replace("to_out_lora.", "to_out.0_lora.")
    name = name.replace("_lora.down.weight", ".lora_A.photomaker.weight")
    return name.replace("_lora.up.weight", ".lora_B.photomaker.weight")


def offload_photomaker_lora(
    model: Any,
    state: Mapping[str, Any],
    offload_dir: Path,
    *,
    dtype: Any,
) -> int:
    from accelerate.utils import offload_weight, save_offload_index

    index = read_offload_index(offload_dir)
    mapped = {map_photomaker_lora_name(name): value for name, value in state.items()}
    expected = {
        name
        for name, _value in model.named_parameters()
        if ".lora_A.photomaker." in name or ".lora_B.photomaker." in name
    }
    if set(mapped) != expected:
        missing = sorted(expected - set(mapped))[:5]
        unexpected = sorted(set(mapped) - expected)[:5]
        raise BackendUnavailableError(
            f"PhotoMaker LoRA mapping mismatch: missing={missing}, unexpected={unexpected}"
        )
    for name, value in mapped.items():
        if name not in index:
            offload_weight(value.to(dtype=dtype), name, offload_dir, index)
    save_offload_index(index, offload_dir)
    return len(mapped)


def alias_wrapped_base_weights(model: Any, offload_dir: Path) -> int:
    from accelerate.utils import save_offload_index

    index = read_offload_index(offload_dir)
    aliases = 0
    for name, _value in model.named_parameters():
        if ".base_layer." not in name or name in index:
            continue
        original_name = name.replace(".base_layer.", ".")
        if original_name not in index:
            continue
        source = offload_dir / f"{original_name}.dat"
        destination = offload_dir / f"{name}.dat"
        try:
            os.link(source, destination)
        except FileExistsError:
            pass
        except OSError as exc:
            raise BackendUnavailableError(
                f"cannot create a disk-offload hard link for {name}: {exc}"
            ) from exc
        index[name] = index[original_name]
        aliases += 1
    if aliases:
        save_offload_index(index, offload_dir)
    return aliases


def attach_disk_offload(model: Any, offload_dir: Path, *, device: Any) -> Any:
    from accelerate import disk_offload

    model = disk_offload(
        model,
        offload_dir,
        execution_device=device,
        offload_buffers=True,
    )
    model.eval()
    return model


def validate_offload_store(offload_dir: Path, expected_count: int | None = None) -> int:
    index = read_offload_index(offload_dir)
    if expected_count is not None and len(index) != expected_count:
        raise BackendUnavailableError(
            f"disk-offload store {offload_dir} has {len(index)} entries; expected {expected_count}"
        )
    missing = [name for name in index if not (offload_dir / f"{name}.dat").is_file()]
    if missing:
        raise BackendUnavailableError(
            f"disk-offload store {offload_dir} is missing {len(missing)} tensor file(s)"
        )
    return len(index)
