# Mara Lab

Mara Lab is a code-first experiment for creating a persistent fictional person
across ordinary, photographically plausible images. The current milestone
generates candidate identities, records complete lineage, and can reproduce a
selected image from its manifest. Mara's first human-approved canon is also
archived in the repository.

## Start on Windows

Requirements: Python 3.12 and, for real generation, an NVIDIA GPU with a working
driver. The bootstrap keeps `uv`, the virtual environment, and model caches
inside the project.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1 -Gpu
.\.venv\Scripts\mara-lab.exe doctor
```

Omit `-Gpu` when only exercising the deterministic fake backend.

## Commands

Run the tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Exercise the artifact workflow without an image-model download:

```powershell
.\.venv\Scripts\mara-lab.exe candidates `
  --config configs/experiments/mara-candidates-v001.yaml `
  --backend fake `
  --limit 8
```

Run an eight-image GPU smoke test, then the frozen 64-candidate experiment:

```powershell
.\.venv\Scripts\mara-lab.exe candidates `
  --config configs/experiments/mara-candidates-v001.yaml `
  --limit 8

.\.venv\Scripts\mara-lab.exe candidates `
  --config configs/experiments/mara-candidates-v001.yaml
```

Each run is written below `experiments/<run-id>/` with its resolved model
revision, environment, image manifest, hashes, VRAM measurements, log, status,
and labeled contact sheet. Run IDs are immutable: a collision fails instead of
overwriting evidence. The model cache is project-local under `.cache/`.

Reproduce one artifact from a completed run:

```powershell
.\.venv\Scripts\mara-lab.exe reproduce `
  --run experiments/<run-id> `
  --artifact-id candidate-0001-seed-11000
```

Archive a human-selected master and backup without overwriting an existing canon:

```powershell
.\.venv\Scripts\mara-lab.exe select `
  --run experiments/<run-id> `
  --master candidate-0063-seed-11062 `
  --backup candidate-0055-seed-11054
```

The command copies the exact PNGs to `characters/<character-id>/canonical/<version>/`
and stores their source records, resolved configuration, sanitized environment,
and hashes. Canon versions are immutable.

## Measured Gate

The first complete run on the reference RTX 4070 SUPER produced 64/64 images at
640 x 832 and 30 steps in 5 minutes 22 seconds, including model setup. Inference
averaged 4.51 seconds per image. Peak CUDA memory was 7.12 GiB allocated and
9.06 GiB reserved, below the 10.5 GiB soft ceiling. A separately regenerated
candidate matched its recorded PNG SHA-256 byte for byte.

Human selection on 2026-09-14 established candidate `#0063` as Mara's v001
master and candidate `#0055` as the backup. Both source and canonical file hashes
match exactly.

The technical decision and full experiment protocol are documented in
[`docs/mara-v0.1-technical-decision.md`](docs/mara-v0.1-technical-decision.md).
