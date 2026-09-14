# Mara Lab

Mara Lab is a code-first experiment for creating a persistent fictional person
across ordinary, photographically plausible images. It generates candidate
identities, archives a human-approved canon, and expands that canon through a
pinned PhotoMaker V2 reference workflow. Every output keeps enough lineage to
audit the model, identity source, prompt, seed, and runtime.

## Start on Windows

Requirements: Python 3.12 and, for real generation, an NVIDIA GPU with a working
driver. The bootstrap keeps `uv`, the virtual environment, and model caches
inside the project.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1 -Gpu
.\.venv\Scripts\mara-lab.exe doctor
```

Omit `-Gpu` when only exercising the deterministic fake backend. Install the
reference workflow and its pinned PhotoMaker source with:

```powershell
.\.tools\uv312\Scripts\uv.exe sync --extra reference --locked
```

The pretrained `buffalo_l` face-recognition pack used by InsightFace is for
non-commercial research under its distributor's terms.

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

Prepare the low-memory FP16 snapshot and run one real PhotoMaker control image:

```powershell
.\.venv\Scripts\mara-lab.exe prepare-model `
  --config configs/models/realvisxl-v5.yaml

.\.venv\Scripts\mara-lab.exe references `
  --config configs/experiments/mara-references-pass-a-v001.yaml `
  --limit 1
```

Omit `--limit 1` to run all 48 Pass A candidates. On a 12 GiB GPU the reference
backend automatically uses a verified, roughly 10 GB disk-backed weight store.
Its setup cost is paid on the first run and reused afterward.

Resume an interrupted reference run with its explicit ID:

```powershell
.\.venv\Scripts\mara-lab.exe references `
  --config configs/experiments/mara-references-pass-a-v001.yaml `
  --run-id mara-references-pass-a-v001-full `
  --resume
```

Resume verifies the original configuration and every existing manifest entry,
prompt, image hash, and source reference before generating only the missing
suffix. Completed runs remain immutable.

After visually selecting two to four consistent Pass A images, freeze them with
the canonical master into an immutable reference set:

```powershell
.\.venv\Scripts\mara-lab.exe approve-references `
  --run experiments/mara-references-pass-a-v001-full `
  --artifact-id reference-a1-03-seed-31002 `
  --artifact-id reference-a1-08-seed-31007
```

Use that set for the 32-candidate Pass B stress test:

```powershell
.\.venv\Scripts\mara-lab.exe references `
  --config configs/experiments/mara-references-pass-b-v001.yaml `
  --reference-set characters/mara/reference-sets/pass-a-v001/reference-set.yaml
```

Pass B covers expressions, camera height, hard daylight, warm tungsten,
fluorescent light, and dim phone-camera light. Select enough Pass B results to
complete a seven-image canonical set: the master plus six approved views across
the two passes. Repeat `--artifact-id` for every selected Pass B result.

```powershell
.\.venv\Scripts\mara-lab.exe finalize-references `
  --base-reference-set characters/mara/reference-sets/pass-a-v001/reference-set.yaml `
  --run experiments/<pass-b-run-id> `
  --artifact-id <pass-b-artifact-id>
```

Generate the separate 72-candidate dataset matrix from those seven references:

```powershell
.\.venv\Scripts\mara-lab.exe references `
  --config configs/experiments/mara-dataset-candidates-v001.yaml `
  --reference-set characters/mara/reference-sets/canonical-v001/reference-set.yaml
```

After human review, freeze 28 training images and 6 held-out validation images
by repeating `--train` and `--validation` with their artifact IDs:

```powershell
.\.venv\Scripts\mara-lab.exe build-dataset `
  --run experiments/<dataset-candidate-run-id> `
  --train <artifact-id> `
  --validation <artifact-id>
```

Prepare the independently pinned training runtime, then run the 200-step SDXL
LoRA smoke profile:

```powershell
.\.venv\Scripts\mara-lab.exe prepare-trainer `
  --config configs/training/mara-lora-smoke-v001.yaml

.\.venv\Scripts\mara-lab.exe train `
  --config configs/training/mara-lora-smoke-v001.yaml
```

Render the four frozen sentinel shots from the resulting adapter:

```powershell
.\.venv\Scripts\mara-lab.exe benchmark `
  --config configs/benchmarks/mara-lora-sentinels-v001.yaml `
  --adapter experiments/<training-run-id>/adapters/<adapter-file>.safetensors
```

The trainer uses rank/alpha 16, UNet-only BF16 training, 8-bit AdamW, SDPA,
gradient checkpointing, and disk-backed latent/text-encoder caches. It records
the exact trainer revision, runtime lock hash, dataset and checkpoint hashes,
command line, log, adapter hashes, timing, and peak VRAM. The benchmark refuses
an altered adapter or one trained against a different base checkpoint. Each run
stages its own training copy so `sd-scripts` cache files never modify the frozen
character dataset. Completed adapters must contain finite weights and at least
one non-zero `lora_up` tensor.

Each run is written below `experiments/<run-id>/` with its resolved model
revision, environment, source-tree fingerprint, image manifest, hashes, VRAM
measurements, log, status, and labeled contact sheet. Run IDs are immutable: a
collision fails instead of overwriting evidence. The model cache is
project-local under `.cache/`.

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

The first production PhotoMaker control at 640 x 832 and 30 steps completed in
91.52 seconds with 1.30 GiB peak reserved VRAM. It produced one upright,
single-face portrait with the requested framing and no collage artifact.

The complete Pass A matrix contains 48/48 unique, hash-verified outputs. An
interruption after image 45 was recovered by the validated resume path, which
preserved the existing files and generated only the final three images.

A real one-step trainer integration produced a loadable rank-16 LoRA with 2,166
finite tensors and all 722 `lora_up` tensors non-zero. Training peaked at 5.86
GiB reserved VRAM; loading that adapter and rendering a RealVisXL sentinel
peaked at 9.16 GiB.

The technical decision and full experiment protocol are documented in
[`docs/mara-v0.1-technical-decision.md`](docs/mara-v0.1-technical-decision.md).
