from __future__ import annotations

import importlib.util
import platform
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from mara_lab.config import (
    BackendName,
    ReferenceBackendName,
    load_benchmark_experiment_config,
    load_experiment_config,
    load_model_profile,
    load_reference_experiment_config,
    load_training_experiment_config,
)
from mara_lab.errors import MaraLabError
from mara_lab.trainers.sd_scripts import prepare_sd_scripts_trainer
from mara_lab.workflows.approve_references import (
    approve_reference_set,
    finalize_reference_set,
    load_reference_set_images,
)
from mara_lab.workflows.benchmark import run_benchmark
from mara_lab.workflows.candidates import run_candidates
from mara_lab.workflows.dataset import build_character_dataset
from mara_lab.workflows.generate import build_generation_config
from mara_lab.workflows.prepare_model import prepare_diffusers_model
from mara_lab.workflows.references import run_references
from mara_lab.workflows.reproduce import reproduce_candidate
from mara_lab.workflows.select import select_canon
from mara_lab.workflows.train import run_training

app = typer.Typer(
    name="mara-lab",
    help="Headless, reproducible synthetic photography experiments.",
    no_args_is_help=True,
)
console = Console()


@app.command("prepare-model")
def prepare_model(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/models/realvisxl-v5.yaml"),
    local_files_only: Annotated[bool, typer.Option("--local-files-only")] = False,
) -> None:
    """Build a low-memory, sharded FP16 Diffusers snapshot."""
    try:
        summary = prepare_diffusers_model(
            load_model_profile(config), local_files_only=local_files_only
        )
    except (MaraLabError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"Prepared [bold]{summary.shard_count}[/bold] model shard(s).")
    console.print(f"Model: {summary.output_dir}")
    console.print(f"Manifest: {summary.manifest_path}")


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
def references(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/experiments/mara-references-pass-a-v001.yaml"),
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    backend: Annotated[ReferenceBackendName | None, typer.Option()] = None,
    reference_set: Annotated[
        Path | None,
        typer.Option("--reference-set", exists=True, dir_okay=False, readable=True),
    ] = None,
    run_id: Annotated[str | None, typer.Option()] = None,
    resume: Annotated[bool, typer.Option("--resume")] = False,
) -> None:
    """Expand canonical references with a pinned identity conditioner."""
    try:
        resolved = load_reference_experiment_config(config)
        overrides: dict[str, object] = {}
        if backend is not None:
            overrides["backend"] = backend.value
            resolved = resolved.model_copy(
                update={
                    "conditioner": resolved.conditioner.model_copy(update={"backend": backend}),
                }
            )
        if reference_set is not None:
            overrides["reference_set"] = reference_set.as_posix()
            resolved = resolved.model_copy(
                update={
                    "reference_images": load_reference_set_images(
                        reference_set, character_id=resolved.character_id
                    )
                }
            )
        if resume:
            overrides["resume"] = True
        resolved = resolved.model_copy(update={"cli_overrides": overrides})
        summary = run_references(
            resolved,
            image_limit=limit,
            run_id=run_id,
            resume=resume,
        )
    except (MaraLabError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"Generated [bold]{summary.generated_count}[/bold] reference candidate(s).")
    console.print(f"Run: {summary.run_dir}")
    console.print(f"Contact sheet: {summary.contact_sheet_path}")
    console.print(f"Peak reserved VRAM: {summary.max_reserved_vram_gib:.2f} GiB")


@app.command("approve-references")
def approve_references(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    artifact_id: Annotated[list[str] | None, typer.Option("--artifact-id")] = None,
    characters_root: Annotated[Path, typer.Option("--characters-root")] = Path("characters"),
    reference_set_id: Annotated[str, typer.Option("--reference-set-id")] = "pass-a-v001",
) -> None:
    """Freeze two to four human-approved Pass A views for Pass B."""
    try:
        summary = approve_reference_set(
            run,
            artifact_id or [],
            characters_root=characters_root,
            reference_set_id=reference_set_id,
        )
    except (MaraLabError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Approved [bold]{len(summary.image_paths) - 1}[/bold] derived view(s).")
    console.print(f"Reference set: {summary.reference_set_dir}")
    console.print(f"Descriptor: {summary.descriptor_path}")


@app.command("finalize-references")
def finalize_references(
    base_reference_set: Annotated[
        Path,
        typer.Option("--base-reference-set", exists=True, dir_okay=False, readable=True),
    ],
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    artifact_id: Annotated[list[str] | None, typer.Option("--artifact-id")] = None,
    characters_root: Annotated[Path, typer.Option("--characters-root")] = Path("characters"),
    reference_set_id: Annotated[str, typer.Option("--reference-set-id")] = "canonical-v001",
) -> None:
    """Freeze seven human-approved references for dataset expansion."""
    try:
        summary = finalize_reference_set(
            base_reference_set,
            run,
            artifact_id or [],
            characters_root=characters_root,
            reference_set_id=reference_set_id,
        )
    except (MaraLabError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Finalized [bold]{len(summary.image_paths)}[/bold] canonical references.")
    console.print(f"Reference set: {summary.reference_set_dir}")
    console.print(f"Descriptor: {summary.descriptor_path}")


@app.command("build-dataset")
def build_dataset(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    train: Annotated[list[str] | None, typer.Option("--train")] = None,
    validation: Annotated[list[str] | None, typer.Option("--validation")] = None,
    dataset_root: Annotated[Path, typer.Option("--dataset-root")] = Path(
        "characters/mara/datasets/v001"
    ),
    dataset_id: Annotated[str, typer.Option("--dataset-id")] = "mara-v001",
    token: Annotated[str, typer.Option()] = "mara_v01",
    expected_train: Annotated[int, typer.Option(min=1)] = 28,
    expected_validation: Annotated[int, typer.Option(min=0)] = 6,
) -> None:
    """Freeze human-selected reference outputs into a captioned dataset."""
    try:
        summary = build_character_dataset(
            run,
            train or [],
            validation or [],
            dataset_root=dataset_root,
            dataset_id=dataset_id,
            token=token,
            expected_train_count=expected_train,
            expected_validation_count=expected_validation,
        )
    except (MaraLabError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"Dataset: [bold]{summary.train_count}[/bold] train, "
        f"[bold]{summary.validation_count}[/bold] validation image(s)."
    )
    console.print(f"Root: {summary.dataset_root}")
    console.print(f"Manifest: {summary.manifest_path}")


@app.command("prepare-trainer")
def prepare_trainer(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/training/mara-lora-smoke-v001.yaml"),
) -> None:
    """Create the isolated, pinned sd-scripts training runtime."""
    try:
        resolved = load_training_experiment_config(config)
        summary = prepare_sd_scripts_trainer(resolved.trainer)
    except (MaraLabError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Trainer source: {summary.source_dir}")
    console.print(f"Revision: {summary.source_revision}")
    console.print(f"Runtime: {summary.runtime_python}")


@app.command()
def train(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/training/mara-lora-smoke-v001.yaml"),
    max_steps: Annotated[int | None, typer.Option(min=1)] = None,
    run_id: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Train a traced SDXL character LoRA with the pinned local trainer."""
    try:
        resolved = load_training_experiment_config(config)
        summary = run_training(resolved, max_train_steps=max_steps, run_id=run_id)
    except (MaraLabError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"Trained [bold]{len(summary.adapters)}[/bold] adapter checkpoint(s) "
        f"in {summary.wall_seconds:.1f} s."
    )
    console.print(f"Run: {summary.run_dir}")
    console.print(f"Manifest: {summary.adapter_manifest_path}")
    console.print(f"Peak reserved VRAM: {summary.max_reserved_vram_gib:.2f} GiB")


@app.command()
def benchmark(
    adapter: Annotated[Path, typer.Option("--adapter", exists=True, dir_okay=False, readable=True)],
    config: Annotated[
        Path,
        typer.Option("--config", "-c", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/benchmarks/mara-lora-sentinels-v001.yaml"),
    adapter_weight: Annotated[float, typer.Option(min=0.01, max=2.0)] = 1.0,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    run_id: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Render fixed sentinel shots from a traced character adapter."""
    try:
        resolved = load_benchmark_experiment_config(config)
        summary = run_benchmark(
            resolved,
            adapter,
            adapter_weight=adapter_weight,
            case_limit=limit,
            run_id=run_id,
        )
    except (MaraLabError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Generated [bold]{summary.generated_count}[/bold] benchmark image(s).")
    console.print(f"Run: {summary.run_dir}")
    console.print(f"Contact sheet: {summary.contact_sheet_path}")
    console.print(f"Peak reserved VRAM: {summary.max_reserved_vram_gib:.2f} GiB")


@app.command()
def generate(
    scene: Annotated[
        str,
        typer.Option(
            "--scene",
            "--prompt",
            "-p",
            help="Describe the setting, framing, light, clothing, and expression.",
        ),
    ],
    adapter: Annotated[
        Path,
        typer.Option("--adapter", exists=True, dir_okay=False, readable=True),
    ] = Path(
        "characters/mara/adapters/v001/adapters/"
        "mara-v001-sdxl-lora-step00000800.safetensors"
    ),
    config: Annotated[
        Path,
        typer.Option("--config", "-c", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/benchmarks/mara-lora-final-v001.yaml"),
    seed: Annotated[int, typer.Option(min=0, max=2**63 - 1)] = 43001,
    adapter_weight: Annotated[
        float, typer.Option("--adapter-weight", min=0.01, max=2.0)
    ] = 0.8,
    run_id: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Generate one traced Mara photograph from a plain scene description."""
    try:
        resolved = build_generation_config(
            load_benchmark_experiment_config(config), scene, seed=seed
        )
        summary = run_benchmark(
            resolved,
            adapter,
            adapter_weight=adapter_weight,
            run_id=run_id,
        )
    except (MaraLabError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    image_path = next((summary.run_dir / "outputs").glob("*.png"))
    console.print(f"Generated image: {image_path}")
    console.print(f"Run: {summary.run_dir}")
    console.print(f"Manifest: {summary.manifest_path}")
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


@app.command("select")
def select_character_canon(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    master: Annotated[str, typer.Option("--master")],
    backup: Annotated[str, typer.Option("--backup")],
    canon_root: Annotated[Path, typer.Option("--canon-root")] = Path("characters"),
    version: Annotated[str, typer.Option("--version")] = "v001",
) -> None:
    """Archive a human-selected master and backup as the character canon."""
    try:
        summary = select_canon(
            run,
            master,
            backup,
            canon_root=canon_root,
            canonical_version=version,
        )
    except (MaraLabError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"Canon: {summary.character_dir}")
    console.print(f"Master: {summary.master_artifact_id} -> {summary.master_path}")
    console.print(f"Backup: {summary.backup_artifact_id} -> {summary.backup_path}")
    console.print(f"Manifest: {summary.manifest_path}")
