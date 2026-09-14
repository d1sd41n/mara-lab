from __future__ import annotations

import json
import os
import runpy
import sys
import time
from contextlib import suppress
from pathlib import Path


def _write_telemetry(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    if len(sys.argv) < 5 or sys.argv[1] != "--telemetry" or "--" not in sys.argv:
        print(
            "usage: trainer_entrypoint.py --telemetry PATH --script PATH -- [trainer args]",
            file=sys.stderr,
        )
        return 2
    separator = sys.argv.index("--")
    prefix = sys.argv[1:separator]
    if len(prefix) != 4 or prefix[0] != "--telemetry" or prefix[2] != "--script":
        print("invalid trainer entrypoint arguments", file=sys.stderr)
        return 2
    telemetry_path = Path(prefix[1]).resolve()
    script_path = Path(prefix[3]).resolve()

    import torch

    if not torch.cuda.is_available():
        print("CUDA is unavailable", file=sys.stderr)
        return 1
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    error: str | None = None
    try:
        sys.argv = [str(script_path), *sys.argv[separator + 1 :]]
        runpy.run_path(str(script_path), run_name="__main__")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        with suppress(Exception), torch.no_grad():
            torch.cuda.synchronize()
        gib = 1024**3
        _write_telemetry(
            telemetry_path,
            {
                "wall_seconds": round(time.perf_counter() - started, 4),
                "max_allocated_vram_gib": round(torch.cuda.max_memory_allocated() / gib, 4),
                "max_reserved_vram_gib": round(torch.cuda.max_memory_reserved() / gib, 4),
                "torch_version": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "error": error,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
