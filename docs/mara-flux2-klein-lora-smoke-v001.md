# Mara FLUX.2 Klein LoRA Smoke v0.1

## Objective

Prove that a FLUX.2 Klein Base identity LoRA can be trained locally on the RTX
4070 SUPER without giving up the newer model's body, expression, and scene
quality. This is a feasibility run, not the final Mara adapter.

## Curated Dataset

The run uses 20 identity-clean images: canonical candidates `#0063` and
`#0055`, plus 18 selected images from Mara dataset v002. The materializer
verifies every source hash, replaces `mara_v01` with the new trigger
`MARA_K1`, and keeps images and captions in separate cache directories so the
official DreamBooth loader sees image files only.

The frozen selection and source hashes live in
`configs/training/mara-flux2-klein-lora-smoke-v001.yaml`.

## Memory Strategy

- Train `black-forest-labs/FLUX.2-klein-base-4B` at revision
  `a3b4f4849157f664bdbc776fd7453c2783562f4d`.
- Quantize the transformer to 4-bit NF4 and use 8-bit Adam.
- Precompute the single training prompt with the existing 4-bit Qwen encoder,
  then unload Qwen before training.
- Split the 7.75 GB Base transformer into sixteen 512 MiB safetensors shards
  to avoid a Windows host-memory spike.
- Cache VAE latents, use gradient checkpointing, BF16 compute, batch size 1,
  four-step gradient accumulation, and a 512 x 384 portrait bucket.

The adaptation is a narrow, generated patch over the official Diffusers
`v0.39.0` trainer at commit
`a3608b512ed7248499a44c61d954965ed9bdae4d`. It only adds support for a
verified precomputed prompt cache and disabled metric logging.

## Result

The one-step probe completed before the full run. The 50-step smoke then
finished successfully in 165.33 seconds, including model loading and latent
caching. Training itself averaged about 2.93 seconds per optimizer step. A
live `nvidia-smi` sample showed 5,287 MiB total board memory in use while the
GPU was at 100 percent utilization.

The adapter contains 120 finite tensors; all 60 learned `lora_B` tensors are
non-zero. It is about 8.4 MB and has SHA-256:

`9e1da5634d6d5cfe9e10ce7069818faff53d5b00b28bfbe2f01f9897f50eda59`

It is stored locally at
`experiments/mara-flux2-klein-lora-smoke-v001/pytorch_lora_weights.safetensors`.

## Load Test

The adapter loads successfully into a 4-bit FLUX.2 Klein Base pipeline. A
same-seed comparison at 384 x 512 and 20 inference steps shows that enabling
the LoRA materially changes the generated person toward Mara, so the adapter
is active and the trigger association is learning. Evaluation reserved 2.71
GiB through PyTorch. The comparison uses a neutral zero embedding for the
Base model's negative branch because the large text encoder remains unloaded;
it is a load test, not a photographic-quality benchmark.

The 50-step image still has a plastic, illustration-like finish. That is not a
quality failure for this gate: the smoke only proves the complete training and
loading path. The comparison is at
`experiments/mara-flux2-klein-lora-smoke-v001/evaluation-v002/comparison.png`.

## Decision

The local FLUX.2 LoRA path passes the feasibility gate. Proceed to a real
candidate run of roughly 500 to 750 optimizer steps, save intermediate
checkpoints, and evaluate them against the frozen body-and-expression suite.
For the candidate run, precompute each image's descriptive caption instead of
using one generic prompt so identity is separated more cleanly from wardrobe,
pose, and background.
