# Mara FLUX.2 Klein LoRA Candidate v0.1

## Objective

Turn the successful 50-step feasibility smoke into a real Mara identity
candidate, then choose a checkpoint from evidence instead of assuming that the
last checkpoint is best.

## Controlled Change

The candidate keeps the smoke run's pinned Base model, 20 identity-clean
images, 512 x 384 bucket, rank/alpha 4, learning rate, BF16 compute, NF4
transformer, 8-bit optimizer, and four-step gradient accumulation. It changes
only the two variables needed for a useful candidate:

- Each image uses its own verified caption instead of one generic prompt.
- Training runs for 600 optimizer steps and saves at 200, 400, and 600.

The 20 captions are encoded once with the 4-bit Qwen text encoder. The resulting
cache has shape `20 x 256 x 7680`, occupies 78,814,933 bytes, and has SHA-256
`7bb56102fbd775192ca1ad3366632ad9fac0ad9e690fffb46fd9d4b499503e09`.
The patched official trainer matches every embedding to a stably sorted image
filename before training starts.

## Training Result

The one-step caption probe passed before the candidate run. The 600-step run
then completed end to end in 1,764.993 seconds, with 29 minutes 5 seconds spent
in the optimizer loop. It averaged about 2.91 seconds per optimizer step. A
live `nvidia-smi` sample showed 5,204 MiB total board memory in use at 100
percent GPU utilization and 69 C.

Every checkpoint adapter contains 120 finite tensors, and all 60 learned
`lora_B` tensors are non-zero. Each adapter is 8,375,144 bytes.

| Step | Adapter SHA-256 |
| ---: | --- |
| 200 | `065b0790f605055afad99686eb31078b6f0b8f4bc32f761246b514ef95a12c41` |
| 400 | `697526d3dbd7717c8f06e53744f5ecbe9cdd34b00b6c133a741b521476e24cf5` |
| 600 | `69603ef4ec299b43a2a37ed82c43e991c68bca774a3d5fbc2dcbbdd6b0eb0723` |

The final root adapter has different safetensors metadata and therefore a
different file hash, but its tensor keys and values are exactly equal to the
step-600 checkpoint.

## Checkpoint Comparison

All checkpoints were rendered with the caption attached to canonical `#0063`,
the same 384 x 512 resolution, 20 inference steps, guidance 4.0, and three
frozen seeds. The visual sheets put canonical `#0063` first; they never show an
unrelated Base-model person as though she were Mara.

InsightFace `buffalo_l` measured cosine similarity to the normalized centroid
of canonical `#0063` and backup `#0055`. This is supporting evidence, not a
calibrated acceptance threshold. The two references score 0.393380 against
each other.

| Step | Seed 49002 | Seed 49003 | Seed 49004 | Mean | Stddev |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 0.240409 | 0.252802 | 0.198968 | 0.230726 | 0.023019 |
| 400 | 0.284280 | 0.232743 | 0.340030 | 0.285684 | 0.043811 |
| 600 | **0.362937** | **0.323974** | **0.348554** | **0.345155** | **0.016087** |

Step 600 wins all three seeds, has the highest average, and has the lowest
variation of the two viable checkpoints. It is selected for the next gate.

Visual evidence:

- [`seed 49002`](../characters/mara/benchmarks/flux2-klein-lora-candidate-v001/comparison-seed-49002.png)
- [`seed 49003`](../characters/mara/benchmarks/flux2-klein-lora-candidate-v001/comparison-seed-49003.png)
- [`seed 49004`](../characters/mara/benchmarks/flux2-klein-lora-candidate-v001/comparison-seed-49004.png)
- [`frozen evaluation`](../characters/mara/benchmarks/flux2-klein-lora-candidate-v001/evaluation.yaml)

## Honest Limit

The identity signal is materially stronger by step 600, but this is not yet a
general-purpose Mara adapter. The evaluation uses one neutral close-portrait
prompt. The outputs are still more polished and illustration-like than the
canonical photographs, and the eyes remain too rigid in some seeds.

Training longer is not the next move. The highest-value next gate is to render
the selected step-600 adapter across the already frozen body-and-expression
matrix. That will reveal whether the LoRA preserves identity in full-body
shots, profiles, laughter, blinking, and ordinary lighting before any more
data or training is added.

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts\run_flux2_lora_smoke.py `
  --config configs\training\mara-flux2-klein-lora-candidate-v001.yaml `
  --run-id mara-flux2-klein-lora-candidate-v001

.\.venv\Scripts\python.exe scripts\evaluate_flux2_lora_checkpoints.py `
  --evaluation-id checkpoint-comparison-v001 `
  --seed 49002

.\.venv\Scripts\python.exe scripts\score_flux2_identity.py `
  --run-dir experiments\mara-flux2-klein-lora-candidate-v001\checkpoint-comparison-v001
```

Repeat the last two commands for seeds `49003` and `49004` with unique
evaluation IDs. The full frozen configuration lives in
`configs/training/mara-flux2-klein-lora-candidate-v001.yaml`.
