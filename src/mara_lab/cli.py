from __future__ import annotations

import importlib.util
import platform
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from mara_lab.config import BackendName, load_experiment_config
from mara_lab.errors import MaraLabError
from mara_lab.workflows.candidates import run_candidates
from mara_lab.workflows.reproduce import reproduce_candidate

app = typer.Typer(
    name="mara-lab",
    help="Headless, reproducible synthetic photography experiments.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def doctor() -> None:
    """Report whether the local host is ready for RealVisXL generation."""
    rows: list[tuple[str, str]] = [
        ("Python", platform.python_version()),
        ("diffusers installed", str(importlib.util.find_spec("diffusers") is not None)),
        ("torch installed", str(importlib.util.find_spec("torch") is not None)),
    ]
    ready = False
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        rows.extend(
            [
                ("PyTorch", torch.__version__),
                ("CUDA available", str(cuda_available)),
                ("CUDA runtime", str(torch.version.cuda)),
            ]
        )
        if cuda_available:
            properties = torch.cuda.get_device_properties(0)
            rows.extend(
                [
                    ("GPU", properties.name),
                    ("GPU VRAM", f"{properties.total_memory / 1024**3:.2f} GiB"),
                    ("BF16 supported", str(torch.cuda.is_bf16_supported())),
                ]
            )
            ready = importlib.util.find_spec("diffusers") is not None
    except ImportError:
        rows.append(("CUDA available", "False"))

    table = Table(title="Mara Lab host")
    table.add_column("Check")
    table.add_column("Value")
    for label, value in rows:
        table.add_row(label, value)
    console.print(table)
    if not ready:
        console.print("[yellow]GPU runtime is not ready. Run `uv sync --extra gpu`.[/yellow]")
        raise typer.Exit(code=1)


@app.command()
def candidates(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/experiments/mara-candidates-v001.yaml"),
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    backend: Annotated[BackendName | None, typer.Option()] = None,
    run_id: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Generate candidate identities and a labeled contact sheet."""
    try:
        resolved = load_experiment_config(config)
        overrides: dict[str, object] = {}
        if limit is not None:
            overrides["seed_limit"] = limit
        if backend is not None:
            overrides["backend"] = backend.value
            model_updates: dict[str, object] = {"backend": backend}
            if backend is BackendName.FAKE:
                model_updates.update(
                    {
                        "model_id": "mara-lab/fake",
                        "revision": "fake-v1",
                        "resolved_revision": "fake-v1",
                    }
                )
            resolved = resolved.model_copy(
                update={"model": resolved.model.model_copy(update=model_updates)}
            )
        resolved = resolved.model_copy(update={"cli_overrides": overrides})
        summary = run_candidates(resolved, seed_limit=limit, run_id=run_id)
    except (MaraLabError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"Generated [bold]{summary.generated_count}[/bold] candidate(s).")
    console.print(f"Run: {summary.run_dir}")
    console.print(f"Contact sheet: {summary.contact_sheet_path}")
    console.print(f"Peak reserved VRAM: {summary.max_reserved_vram_gib:.2f} GiB")


@app.command()
def reproduce(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    artifact_id: Annotated[str, typer.Option("--artifact-id")],
    allow_drift: Annotated[bool, typer.Option()] = False,
) -> None:
    """Regenerate one candidate from its pinned manifest entry."""
    try:
        summary = reproduce_candidate(run, artifact_id, strict=not allow_drift)
    except (MaraLabError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    result = "exact match" if summary.exact_match else "pixel drift detected"
    console.print(f"Reproduction: {summary.output_path}")
    console.print(f"Result: {result}")
