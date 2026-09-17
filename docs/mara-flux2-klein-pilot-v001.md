# Mara FLUX.2 Klein Pilot v0.1

## Decision

Adopt FLUX.2 Klein 4B as Mara's experimental next-generation image path while
keeping the SDXL LoRA v0.2 workflow unchanged. Use the distilled four-step model
for inference and both canonical images as direct identity references. Do not
train a FLUX LoRA until a broader benchmark shows that multi-reference identity
conditioning is insufficient.

## Why This Is the Highest-ROI Step

- It tests the newer image prior immediately, without paying for a new dataset or
  training run.
- Multi-reference conditioning uses the approved `#0063` master and `#0055`
  backup directly.
- The FP8 transformer leaves enough memory for two references, the VAE, and a
  640 x 832 output on the 12 GiB RTX 4070 SUPER.
- The existing SDXL generator and Mara v0.2 LoRA remain available as a stable
  fallback.

## Reproducible Path

```text
#0063 master + #0055 backup
             |
             v
 FLUX.2 Klein multi-reference conditioning
             |
   cached Qwen3 prompt embedding
             |
  four-step FP8 transformer inference
             |
             v
 image + manifest + hashes + contact sheet
```

The pilot pins the official `black-forest-labs/FLUX.2-klein-4B` pipeline at
revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`. The compatible FP8
Diffusers transformer is pinned to
`Photoroom/FLUX.2-klein-4b-fp8-diffusers` revision
`408c457f3589e17a1be1dae5bf0dcaf09cd4985f`; its checkpoint SHA-256 is
`13b37a5ca5cd9cf190236e7e99a3f086cf24618682f74e27a6f00cb173c308c8`.
The checkpoint uses TorchAO's older serialized FP8 representation, so the pilot
pins TorchAO 0.16.0 and verifies the checkpoint hash before loading it.

Qwen3's original first shard is 4.63 GiB and exceeded the available host-memory
margin during quantized loading. The preparation command re-shards the exact
tensor bytes toward a 512 MiB target; individual tensors remain intact, so the
largest output shard is about 742 MiB. This changes storage layout, not weights.

## Result

The frozen ten-scene run `mara-flux2-klein-everyday-v003` completed on
2026-09-16 with:

- 10/10 valid outputs at 640 x 832;
- four inference steps and guidance 1.0;
- 65.54 seconds end to end with uncached prompt encoding;
- 7.06 GiB peak reserved VRAM;
- deterministic seeds `47001` through `47010`;
- byte-identical PNG output across three reruns of seed `47001`;
- two identity references in every generation.

Initial model-assisted visual review found Mara recognizable throughout.
Downward and side gazes look natural, body proportions remain coherent in full
and medium views, and hands, clothes, lighting, and environments are materially
more convincing than the SDXL everyday preview. Human approval is still
pending; the frozen contact sheet is stored with the benchmark evaluation.

## Limits And Next Gate

- Stronger smiles produce a small but visible identity shift.
- Only two scenes show most of the body, so this is evidence of feasibility, not
  a complete body-consistency benchmark.
- The FP8 checkpoint format depends on TorchAO 0.16.0 and should eventually be
  migrated to a current safe serialization format.
- The next gate should freeze 20 to 30 scenes emphasizing full-body poses,
  seated anatomy, hands, profiles, and stronger expressions. Train a FLUX.2
  Base LoRA only if that benchmark exposes identity drift that references alone
  cannot solve.
