# Mara Lab

Mara Lab is a code-first experiment for creating a persistent fictional person
across ordinary, photographically plausible images. It generates candidate
identities, archives a human-approved canon, expands that canon through a pinned
PhotoMaker V2 reference workflow, and trains a reusable SDXL LoRA. Every output
keeps enough lineage to audit the model, identity source, prompt, seed, and
runtime.

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

## Generate Mara

Mara v0.2 is ready locally. Describe only the photograph you want; the command
adds the identity token and adult class phrase, uses the selected 600-step LoRA
at weight 0.7, applies the expression-aware negative prompt, and records the
prompt, seed, model, adapter hash, and runtime:

```powershell
.\.venv\Scripts\mara-lab.exe generate `
  --scene "reading beside a laundromat window in overcast daylight, medium shot" `
  --seed 43001
```

The PNG is written below `experiments/<run-id>/outputs/`. The 85 MB LoRA lives
at `characters/mara/adapters/v002/adapters/` and is intentionally excluded from
Git; its descriptor, provenance, SHA-256, and recommended weight are versioned.

## FLUX.2 Klein Pilot

The isolated FLUX.2 Klein path tests a newer generator without replacing the
stable SDXL/LoRA workflow. It conditions directly on Mara's `#0063` master and
`#0055` backup, so this inference pilot does not train or consume a FLUX LoRA.

Prepare the pinned official pipeline components, the hash-verified FP8
transformer, and a low-memory 512 MiB re-sharding of Qwen3:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_flux2_klein_pilot.py
```

Then render the frozen ten-scene everyday matrix:

```powershell
.\.venv\Scripts\python.exe scripts\flux2_klein_pilot.py `
  --run-id mara-flux2-klein-everyday-v001 `
  --limit 10
```

The FP8 checkpoint is loaded only after its pinned SHA-256 is verified. Prompt
embeddings are cached outside Git, and every image records the model revisions,
reference hashes, prompt, seed, timing, and output hash. See the
[`pilot decision`](docs/mara-flux2-klein-pilot-v001.md) for the measured result
and remaining limits, plus the frozen
[`contact sheet`](characters/mara/benchmarks/flux2-klein-pilot-v001/contact-sheet.png)
and [`evaluation`](characters/mara/benchmarks/flux2-klein-pilot-v001/evaluation.yaml).

Run the frozen 24-case body and expression gate, then record supporting facial
similarity measurements:

```powershell
.\.venv\Scripts\python.exe scripts\flux2_klein_pilot.py `
  --run-id mara-flux2-klein-body-expression-v001 `
  --suite body-expression `
  --limit 24

.\.venv\Scripts\python.exe scripts\score_flux2_identity.py `
  --run-dir experiments\mara-flux2-klein-body-expression-v001
```

The frozen gate passes body consistency, scene quality, and expression range,
but reference-only conditioning does not hold Mara's identity strongly enough
in profiles, rear turns, and broad laughter. See the
[`gate decision`](docs/mara-flux2-klein-body-expression-gate-v001.md),
[`contact sheet`](characters/mara/benchmarks/flux2-klein-body-expression-v001/contact-sheet.png),
and [`evaluation`](characters/mara/benchmarks/flux2-klein-body-expression-v001/evaluation.yaml).

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

Create a new immutable dataset version by replacing selected examples from a
completed correction run. Repeat each selection option as needed:

```powershell
.\.venv\Scripts\mara-lab.exe revise-dataset `
  --base-dataset characters/mara/datasets/v001 `
  --run experiments/<correction-run-id> `
  --drop-train <old-artifact-id> `
  --add-train <new-artifact-id>
```

Prepare the independently pinned training runtime, then run the 200-step SDXL
LoRA smoke profile:

```powershell
.\.venv\Scripts\mara-lab.exe prepare-trainer `
  --config configs/training/mara-lora-smoke-v001.yaml

.\.venv\Scripts\mara-lab.exe train `
  --config configs/training/mara-lora-smoke-v001.yaml
```

After the smoke adapter passes all four sentinels, run the frozen 1,200-step
profile. It saves every 200 steps so the useful 400-1,200 checkpoint region can
be compared without retraining:

```powershell
.\.venv\Scripts\mara-lab.exe train `
  --config configs/training/mara-lora-full-v001.yaml
```

Render the four frozen sentinel shots from the resulting adapter:

```powershell
.\.venv\Scripts\mara-lab.exe benchmark `
  --config configs/benchmarks/mara-lora-sentinels-v001.yaml `
  --adapter experiments/<training-run-id>/adapters/<adapter-file>.safetensors
```

The final benchmark freezes 20 ordinary photographs and unseen seeds:

```powershell
.\.venv\Scripts\mara-lab.exe benchmark `
  --config configs/benchmarks/mara-lora-final-v001.yaml `
  --adapter experiments/<training-run-id>/adapters/<selected-adapter>.safetensors
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

The production run trained 1,200 steps in 893.9 seconds and saved six checkpoint
snapshots plus the final adapter. A frozen comparison rendered 120 images across
steps 600 and 800 at weights 0.6, 0.8, and 1.0. Human review selected step 800
at weight 0.8. Its final 20-image benchmark passes the v0.1 identity-consistency
gate across varied lighting, backgrounds, viewpoints, close portraits, and
medium shots. The main known limitation is an older apparent age than the
canonical portrait; body consistency is encouraging but remains outside the
v0.1 acceptance criterion. See the frozen
[`evaluation`](characters/mara/benchmarks/v001/evaluation.yaml) and
[`contact sheet`](characters/mara/benchmarks/v001/final-contact-sheet.png).

Mara v0.2 targets the rigid, wide-eyed bias found in v0.1. A 22-image correction
run supplied eight replacement training examples and two replacement validation
examples while preserving the immutable 28/6 dataset split. The new run trained
800 steps in 584.6 seconds and peaked at 5.87 GiB reserved VRAM. Review across
checkpoints 400, 600, and 800 selected step 600 at weight 0.7. Its eight-case
expression benchmark and 20-case general benchmark show better downward gaze,
side gaze, smiles, squinting, and frontal relaxation while retaining identity and
medium-shot body consistency. Direct eye contact can still look more open than
requested, so this is a measured improvement rather than a complete cure. See the
v0.2 [`evaluation`](characters/mara/benchmarks/v002/evaluation.yaml),
[`general sheet`](characters/mara/benchmarks/v002/final-contact-sheet.png), and
[`expression sheet`](characters/mara/benchmarks/v002/expression-contact-sheet.png).

The FLUX.2 Klein pilot generated 10/10 multi-reference scenes at 640 x 832 in
65.54 seconds end to end on the same RTX 4070 SUPER, including uncached prompt
encoding and model setup. Generation used four steps and peaked at 7.06 GiB
reserved VRAM. Initial visual review found a clear improvement in contemporary
realism, hands, clothing, backgrounds, gaze variety, and full-body anatomy over
the SDXL preview. Human approval is still pending; stronger smiles show minor
identity drift, so FLUX remains experimental until a larger pose and expression
gate is frozen.

The technical decision and full experiment protocol are documented in
[`docs/mara-v0.1-technical-decision.md`](docs/mara-v0.1-technical-decision.md).
