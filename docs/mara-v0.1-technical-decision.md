# Mara v0.1 Technical Decision

## Executive decision

Use **RealVisXL V5.0 (the non-Lightning SDXL checkpoint)** as the primary image model. Establish Mara first as a small, model-independent set of approved canonical references. Compare two identity mechanisms on the same frozen benchmark:

1. **PhotoMaker V2** as a zero-shot, reference-conditioned baseline.
2. **A UNet-only SDXL LoRA** trained against the exact pinned RealVisXL checkpoint.

This is the lowest-risk route to a convincing recurring person on an RTX 4070 Super with 12 GiB VRAM. SDXL has the strongest combination of realistic-human quality, mature LoRA tooling, identity adapters, reproducible headless workflows, and memory behavior that does not depend on constant CPU/NVMe swapping.[^1][^2][^3]

Use **Realistic Vision V6.0 B1 / Stable Diffusion 1.5** only as the fallback. It has a materially lower realism, anatomy, resolution, and prompt-following ceiling, but its inference and LoRA training are comfortably inside the hardware envelope and its ecosystem is exceptionally mature.[^4][^5][^6]

Treat **FLUX.2 Klein 4B Base**, paired with its distilled sibling for inference, as the first challenger rather than the initial platform. Its Apache 2.0 license, native multi-reference editing, and explicit support for character LoRAs make it strategically attractive. The present evidence for 12 GiB is contradictory, however: Black Forest Labs reports a 12 GB training minimum and 9.2 GB base inference, while the model card and trainer guidance put common configurations closer to 13-24 GB unless weights are quantized or offloaded.[^7][^8][^9][^10] It should be promoted only after a short local feasibility gate defined below.

The v0.1 deliverable is therefore a **character experiment**, not a platform build. It should prove that approximately 20 varied, ordinary photographs are recognizably the same fictional adult woman before any world model, production service, or general-purpose generation framework is built.

**Evidence cutoff:** 2026-09-11. VRAM values marked *estimate* are engineering planning ranges, not vendor guarantees. They include the practical pipeline rather than merely dividing parameter count by weight precision.

## Model comparison

The hardware requirement is interpreted as more than "can start with offload." A comfortable local path should leave headroom for the Windows display stack and allocator fragmentation, avoid sustained block swapping, and complete repeat runs unattended. For v0.1, the soft ceiling is **10.5 GiB peak CUDA memory reserved**; 12 GiB is an emergency ceiling, not a target.

| Model / family | Parameters and resident components | Inference VRAM | Realistic LoRA-training VRAM | Native / useful resolution | Photoreal humans | Identity suitability | Expected 4070S speed | Training complexity | Ecosystem maturity | License | Main compromise |
|---|---|---:|---:|---|---|---|---|---|---|---|---|
| **RealVisXL V5.0 / SDXL** **PRIMARY**[^1][^2] | About 3B checkpoint; SDXL also loads dual CLIP encoders and VAE | **9.06 GiB reserved / 7.12 GiB allocated**, measured at 640 x 832, batch 1; staged 768 x 1024 reached 10.80 GiB reserved | **10-12 GiB** at 768 with frozen encoders, cached latents/text outputs, checkpointing, and 8-bit optimizer (*estimate*); official Diffusers recipe is under 16 GB at 1024[^3] | SDXL native 1024; train v0.1 at 768-area buckets, infer at the measured-safe 640 x 832 | Strong, especially skin and ordinary photography; better starting point than raw SDXL | Excellent practical ecosystem: LoRA, PhotoMaker, InstantID, IP-Adapter, PuLID | **4.34-4.81 s/image**, measured at 30 steps and 640 x 832 | Low-medium; stable recipes and many diagnostics | Very mature | OpenRAIL++ | Slower/larger than SD1.5; LoRA is tied to the pinned base/fine-tune and may not transfer cleanly to another SDXL checkpoint |
| **FLUX.2 Klein 4B Base + 4B distilled** **CHALLENGER**[^7][^8] | 4B generator plus Qwen3-family text encoder; headline count understates resident pipeline | Base reported at **9.2-13 GB**; distilled 8.4 GB vendor figure; int8 can be around 8 GB[^7][^9] | Vendor minimum **12 GB**; standard LoRA commonly under 24 GB, with 12 GB requiring quantization and a carefully optimized trainer[^8][^9][^10] | 1024-class; supports text-to-image and single/multi-reference editing | Potentially strongest raw fidelity and prompt following in this shortlist | Excellent architecture fit: native multi-reference editing and base model intended for LoRA | Base likely 25-60 s/image; distilled roughly 2-6 s/image (*4070S estimates*) | Medium-high on 12 GiB; newer stack, quantization choices matter | Growing rapidly, less battle-tested than SDXL | Apache 2.0 for 4B Base and distilled | Best long-term option may still be operationally marginal today; a failed 12 GiB run would waste the first experiment on systems work |
| **Stable Diffusion 3.5 Medium**[^11][^12] | About 2.5B MMDiT plus three text encoders, including T5-XXL | Full pipeline exceeds the comfortable envelope; quantized/offloaded modes fit roughly **8-12 GiB** | Roughly **10-12 GiB** only with NF4, 512 px, batch 1, and aggressive savings; 16+ GiB is more realistic[^12] | 1024 native; low-memory training guidance falls to 512 | Good prompt adherence and respectable humans, but not a decisive gain here | Fewer mature identity tools and character recipes than SDXL | Moderate (*estimate*) | High; text-encoder footprint and schedule sensitivity add risk | Medium | Stability Community License; free under its stated annual-revenue threshold | Requires compromises at exactly the point v0.1 needs dependable training; license is less permissive than Apache 2.0 |
| **SANA 1.5 1.6B**[^13][^14] | 1.6B diffusion transformer plus Gemma 2 text encoder | Officially about **12 GB** in BF16 and under 8 GB at 4-bit[^13] | Official guidance is approximately **32 GB**; current trainer guidance treats 24 GB as practical[^14] | 1024; efficient 32x latent compression | Good for its size and fast; less proven for close recurring faces | LoRA exists, but personalization ecosystem and evidence are thin | Fast inference; exact 4070S figure requires measurement | High on this GPU | Early-medium | Apache 2.0 model; Gemma terms also apply to its text encoder | Small denoiser does not translate into small end-to-end training memory |
| **Lumina-Image 2.0**[^15][^16] | 2.6B transformer plus Gemma 2 text encoder | Approximately **10-14 GiB** or CPU offload (*estimate*) | Roughly **12-14 GiB** for a minimal rank-16 LoRA; more headroom is recommended[^16] | 1024-class | Promising composition and text rendering; less evidence for ordinary-person fidelity | General LoRA path exists; little mature character-specific tooling | Moderate (*estimate*) | Medium-high on 12 GiB | Early | Apache 2.0 model; Gemma terms also apply | Marginal memory with no clear identity-quality advantage over the safer SDXL route |
| **Z-Image Base 6B**[^17][^18] | 6B generator plus Qwen-family encoder | Quantized configurations can approach **10-12 GiB**; full precision is above the target | About **10-12 GiB** only with NF4 and aggressive savings; 16-24 GB for int8 and 32-40 GB unquantized are more realistic[^18] | 1024 native; documented 512-2048 range | Strong emerging quality | Trainable base is attractive, but identity recipes are young | Moderate-slow (*estimate*) | High on 12 GiB | Early | Apache 2.0 | "Fits" only in the most constrained configuration; too little operational margin for the first proof |
| **Realistic Vision V6.0 B1 / SD1.5** **FALLBACK**[^4][^5] | About 0.86-0.9B UNet plus one CLIP encoder and VAE | **3-5 GiB** at 512-768 (*estimate*) | **5-8 GiB** optimized; official Diffusers LoRA training ran on an 11 GB 2080 Ti without memory tricks[^6] | Native 512; useful up to about 640-768 with careful crops/upscaling | Good at 512 for its generation, visibly behind modern 1024 models in skin, anatomy, and scene coherence | Very mature LoRA, DreamBooth, ControlNet, and face-adapter tooling | Fast: roughly 1-3 s/image at 512, 25 steps (*estimate*) | Low | Extremely mature | CreativeML OpenRAIL-M | Comfortable compute, but a lower quality ceiling and weaker generalization make false success more likely |

### What was screened out

- **Qwen-Image, FLUX.1-dev, FLUX.2-dev, HiDream, Kandinsky 5 Image, and ZLab i1** are interesting quality references, not sensible first training backends on 12 GiB. Their full pipelines or documented practical LoRA paths generally target 16-40+ GB.[^19][^20]
- **Juggernaut XL** is a credible SDXL photorealistic alternative and advertises 8 GB inference, but it does not improve the compute case over RealVisXL and its model card adds a commercial API restriction beyond the base OpenRAIL terms.[^21]
- Distilled "turbo" checkpoints are useful for fast exploration but are not the first training base. Train against the non-distilled model, then test whether the resulting identity asset behaves acceptably on a compatible distilled sibling.

### Why parameter count is misleading

Training memory includes more than trainable weights: frozen model weights, text encoders, VAE, activations, gradients for the adapted layers, optimizer states, temporary attention buffers, and allocator fragmentation. A 1.6B model with a large language encoder can therefore be less comfortable than a 3B SDXL pipeline whose text outputs and image latents are cached. Quantization may lower resident weights while introducing trainer constraints or quality uncertainty. Model selection should consequently use a measured end-to-end peak, not a parameter-count heuristic.

### Why identity conditioning and LoRA are both tested

PhotoMaker uses stacked identity embeddings from one or more face images and requires no per-person training; its paper reports stronger identity preservation and diversity than several test-time tuning baselines.[^22][^23] PhotoMaker V2 supports SDXL checkpoints and reports an 11 GB minimum, making it viable, though close to the ceiling.[^22]

A LoRA can encode identity more compactly and can work without reference images at inference, but it can also bind Mara to training backgrounds, camera distance, expression, or hair. DreamBooth research identifies context entanglement and language drift as core failure modes, especially with small subject datasets.[^24] Neither mechanism should win by assumption. Mara's persistent asset may legitimately be canonical images plus a conditioner configuration if that passes the benchmark more convincingly than the trained LoRA.

There is one licensing caveat: PhotoMaker V2 and several face-ID adapters rely on InsightFace. InsightFace's code is MIT, but its distributed pretrained face-recognition model packs are restricted to non-commercial research.[^25] That is acceptable for this stated experiment, but it must not silently become a commercial production dependency. InstantID has a similar research-only checkpoint constraint.[^26]

## v0.1 architecture

### Scope boundary

Build only enough structure to run, reproduce, compare, and audit the Mara experiment. Do not build a server, UI, job queue, database, cloud integration, general world simulator, or end-user prompt product. The first implementation should expose five headless commands: `candidates`, `references`, `train`, `benchmark`, and `evaluate`.

```text
character canon -----------+----------------------+
  images + manifest        |                      |
                            v                      v
shot specs + camera specs -> generation backend -> output manifest
                            ^                      |
identity asset ------------+                      v
  references or LoRA                         evaluators
                                                   |
training dataset -> trainer backend -> adapter ----+
                                                   |
frozen benchmark specs ----------------------------+
```

The arrows carry typed specifications and artifact references. They do not carry backend-specific prompt strings as domain objects.

### Proposed package layout

```text
pyproject.toml
uv.lock
src/mara_lab/
  cli.py
  config.py
  domain.py
  artifacts.py
  manifests.py
  backends/
    base.py
    sdxl_diffusers.py
    photomaker_v2.py
  trainers/
    base.py
    sdxl_lora.py
  evaluation/
    identity.py
    memorization.py
    realism.py
  workflows/
    candidates.py
    references.py
    dataset.py
    benchmark.py
configs/
  compute/local-4070s.yaml
  models/realvisxl-v5.yaml
  models/realistic-vision-v6.yaml
characters/mara/
  character.yaml
  canonical/manifest.jsonl
  datasets/v001/manifest.jsonl
  identity-assets/<backend>/<run-id>/asset.yaml
benchmarks/selfies-v001.yaml
experiments/<run-id>/
  resolved-config.yaml
  environment.json
  events.jsonl
  outputs/manifest.jsonl
  metrics.json
  contact-sheets/
```

Use **Pydantic models** for validation, YAML for concise human-authored configuration, JSONL for appendable image/result manifests, and JSON for immutable resolved snapshots. Image files stay in the filesystem and are addressed by relative path plus SHA-256. A database would add failure modes without helping a one-character experiment.

### Core domain boundaries

- `CharacterCanon`: a stable character ID, policy metadata (`fictional`, `adult`, `no_nudity_v001`), and approved canonical image references. It contains no Diffusers class names or model tokens.
- `ShotSpec`: framing, pose, expression, action, setting, wardrobe, lighting intent, and occlusion. This answers *what is happening*.
- `CameraSpec`: capture device class, focal-length equivalent, depth of field, exposure behavior, flash, noise, motion blur, and processing. This answers *how it was photographed*.
- `GenerationSpec`: references the character, shot, camera, model profile, identity asset, seed list, and output policy. A backend compiles it into prompts and arguments.
- `IdentityAsset`: either reference images plus conditioner settings or an adapter file. Every adapter records its exact base-model repository, revision, weight hash, trainer revision, and dataset-manifest hash.
- `ExperimentManifest`: the immutable resolved inputs and environment from which outputs and metrics descend.

Backends should be capability-based. A text-to-image backend need not pretend it supports multiple references; a reference-conditioned backend advertises that capability explicitly. Backend-specific settings belong in a namespaced `backend_options` object and must be copied into the resolved experiment manifest.

Do not create a speculative `ComputeBackend` hierarchy in v0.1. A validated `ExecutionProfile` is enough to describe the local GPU, precision, memory ceiling, process count, and cache paths. Each workflow is an ordinary non-interactive command with explicit inputs and outputs. A future remote runner can execute those same commands on disposable infrastructure without changing the image-model, character, or experiment contracts.

The first LoRA trainer should be a thin Python/CLI wrapper around a pinned, upstream SDXL trainer such as `kohya-ss/sd-scripts` or the official Diffusers DreamBooth LoRA script, not a fork of its training loop.[^3][^27] The wrapper owns validation, manifests, subprocess invocation, checkpoint discovery, and telemetry. The upstream trainer owns optimization. This makes replacing the trainer much cheaper than replacing character assets or experiment history.

### Reproducibility record

Every generated image must have a sidecar/manifest row containing:

- character, shot, camera, and experiment IDs;
- source reference hashes and adapter hash, if any;
- model repository, exact revision, weight-file hash, VAE, scheduler, and precision;
- positive and negative prompts after backend compilation;
- width, height, steps, guidance, seed, sampler, and identity strength;
- Python, package-lock, PyTorch, CUDA, driver, GPU, and operating-system versions;
- wall time, peak allocated VRAM, peak reserved VRAM, and output SHA-256;
- parent artifact IDs and human-review status.

Archive the selected outputs. Seeds and software versions enable repeatability, but identical pixels across different CUDA kernels are not guaranteed. The acceptance test is byte-identical reruns on the pinned host for three sentinel cases; on a changed host, retain the same semantic result and report pixel drift rather than hiding it.

### Explicitly deferred

- Full-body, hands-first, nudity, persistent wardrobe, tattoo placement, and body-shape continuity.
- A general world representation, location memory, narrative timeline, or multi-character scenes.
- Automated aesthetic selection without a human gate.
- REST APIs, queues, web UIs, ComfyUI graphs, Docker/Kubernetes, cloud inference, and experiment SaaS.
- Cross-base LoRA conversion. An adapter is derived, replaceable, and base-revision-bound.

## Mara v0.1 experiment protocol

### 0. Freeze the experiment

Before generating Mara, commit:

- the 20 benchmark cases below, including their seed list;
- the compute ceiling of 10.5 GiB peak reserved VRAM;
- exact model and trainer revisions;
- the config schema and manifest schema;
- the human-review rubric;
- the policy that all people are fictional adults and v0.1 contains no nudity.

Assume one RTX 4070 Super 12 GiB, at least 32 GB system RAM, 50 GB free local storage, Python 3.12, and one sequential GPU process. Start on native Windows with PyTorch SDPA to minimize setup. Use WSL2 only if a pinned trainer dependency proves incompatible; host choice must not leak into character or shot schemas.

### 1. Generate candidate identities

Generate **64 images**, one image per seed, with RealVisXL V5.0 at the measured-safe 640 x 832 resolution, 30 steps, DPM++ SDE Karras, CFG 6.0, and seeds `11000` through `11063`. Use a neutral prompt template whose fixed content is:

```text
unretouched smartphone portrait of {identity}, chest-up, looking at camera, neutral
expression, dark crew-neck shirt, plain apartment wall, indirect window light,
natural skin texture, realistic exposure
```

The negative prompt should reject minors, celebrity resemblance, multiple people, plastic/airbrushed skin, CGI/illustration, extreme makeup, text/watermarks, and malformed facial anatomy. Do not specify a celebrity, famous character, or real individual. The first unconstrained smoke test converged too strongly on one facial type, so `{identity}` rotates through eight frozen descriptions spanning skin tone, face structure, and hair. These descriptions widen the candidate search and do not define Mara; the human-selected image defines her initial canon. The exact variants live in the versioned experiment config.

Automatically reject unreadable files, zero or multiple detected faces, a face occupying less than 18% of image area, and severe blur. Produce a static contact sheet labeled only with candidate IDs. Human selection chooses one master and one backup based on ordinary distinctiveness, clearly adult appearance, visible facial geometry, plausible skin, mild asymmetry, and absence of artifacts. Record the chosen image's complete lineage; the name "Mara" is assigned only after selection.

#### Measured implementation gate (2026-09-11)

- Host: NVIDIA GeForce RTX 4070 SUPER, 11.99 GiB VRAM, CUDA 12.8, PyTorch 2.11.0, native Windows.
- Weights: `RealVisXL_V5.0_fp16.safetensors`, SHA-256 `6a35a7855770ae9820a3c931d4964c3817b6d9e3c6f9c4dabb5b3a94e5643b80`, pinned repository revision `ac93e0dda1f6d448cae19bbfab8c5e720a5e48bc`.
- A staged 768 x 1024 smoke image reached 10.80 GiB reserved and failed the 10.5 GiB soft gate. At 640 x 832, 64/64 images completed with 7.12 GiB peak allocated and 9.06 GiB peak reserved.
- End-to-end time was 5 minutes 22 seconds, including setup; generation averaged 4.51 seconds per image.
- Candidate `candidate-0037-seed-11036` was regenerated from its manifest and matched the original PNG SHA-256 byte for byte.
- Human review on 2026-09-14 selected `candidate-0063-seed-11062` as the v001 master and `candidate-0055-seed-11054` as its backup. Their exact PNGs and complete source records are archived under `characters/mara/canonical/v001/`.

### 2. Expand references without identity recursion

Use PhotoMaker V2 with the master image as the only identity input for **pass A: 48 candidates**. Generate eight seeds for each of six cells:

| Cell | View / expression | Light |
|---|---|---|
| A1 | frontal, neutral | indirect window |
| A2 | frontal, soft smile | overcast outdoor |
| A3 | left three-quarter, neutral | indirect window |
| A4 | right three-quarter, neutral | indirect window |
| A5 | left near-profile, relaxed | soft indoor practical |
| A6 | right near-profile, relaxed | open shade |

Keep clothing and backgrounds plain. A derived image may never become a reference merely because an automated metric likes it. Human review approves two to four strong, non-duplicate views.

Then run **pass B: 32 candidates**, using the master plus only those approved views. Generate four seeds for eight cells that add a clear smile, serious expression, slightly high camera, slightly low camera, harder daylight, warm kitchen light, fluorescent light, and dim phone-camera light. This two-pass rule limits self-reinforcing facial distortions.

### 3. Curate the canonical set

Select **seven canonical images**: the master, a frontal alternate, left and right three-quarter views, one near-profile from each side, and one expression alternate. Require exactly one detectable face, useful sharpness, no visible identity-changing artifact, and meaningful view diversity.

Use AdaFace/CVLFace as the primary face-embedding evaluator and InsightFace only as a secondary diagnostic, because PhotoMaker itself relies on InsightFace-family features.[^25][^28] Using the conditioner's own embedding as the acceptance judge would reward its blind spots. Use DINOv2 only for whole-image similarity and duplicate diagnostics, not as identity truth.[^29]

Treat every downloaded evaluator checkpoint as a separately licensed dependency. CVLFace is appropriate for this research comparison, but its chosen pretrained weight and training-data terms still require an audit before commercial deployment.

Retain all rejected candidates and their reasons. Canonical images are immutable; replacements create a new canon version.

### 4. Build the LoRA dataset

From the seven canonical references, generate **72 candidates** with PhotoMaker V2:

- 24 close/selfie frames across six ordinary scene-light combinations and four seeds;
- 32 chest-up/medium frames across eight combinations and four seeds;
- 16 three-quarter frames across four combinations and four seeds.

Scenes should include bedroom morning light, bathroom fluorescent light, parked-car daylight, kitchen tungsten, office fluorescent, overcast outdoors, hard noon sun, evening street light, elevator light, cafe window light, dim monitor light, and direct compact-camera flash. Use simple everyday tops in restrained colors. No setting may contribute more than three retained images; no outfit may contribute more than four.

Human-curate **28 training images** and **six held-out identity-validation images**. Target approximately 40% frontal, 40% left/right three-quarter, and 20% near-profile or candid angles; 50% neutral, 30% smile, and 20% serious/relaxed alternatives. Prefer consistency over filling a quota. A technically wrong face is never rescued by dataset size.

Create captions from the known structured shot and camera specs, then have a human skim every caption. Use the rare token `mara_v01` with the class phrase `adult woman`. The template is:

```text
a [capture style] [framing] photo of mara_v01, an adult woman, [view], [expression],
wearing [clothing], in [setting], lit by [lighting], [camera behavior]
```

Caption attributes that should remain controllable: view, expression, framing, hair state, clothing, setting, light, flash, noise, and motion. Do not repeatedly caption stable facial traits; those are what the identity token must learn. Do not use an image captioner to invent metadata already known from generation. A local vision-language model may flag mismatches, but it does not author the source of truth.

### 5. Run the 12 GiB training smoke test

Before the full run, train for **200 steps** and generate four sentinel prompts. The initial SDXL LoRA configuration is:

| Setting | Value |
|---|---|
| Base | Exact pinned RealVisXL V5.0 non-Lightning revision |
| Train target | UNet attention LoRA only; text encoders frozen |
| Rank / alpha | 16 / 16 |
| Resolution | 768-area aspect-ratio buckets; min 512, max 1024 |
| Batch / accumulation | 1 / 1 |
| Precision | BF16; fall back to FP16 only for a demonstrated incompatibility |
| Optimizer | 8-bit AdamW |
| Learning rate | `1e-4`, constant with 100-step warmup |
| Memory features | cache latents, cache text-encoder outputs, gradient checkpointing, SDPA or xFormers |
| Seed | `240311` |
| Compilation / offload | `torch.compile` off; no block/NVMe offload |

Proceed only if peak reserved VRAM is at most 10.5 GiB, loss is finite, all four sentinel outputs decode, and throughput does not collapse from memory paging. If the peak is 10.5-12 GiB, reduce training area to 640 before adding more memory machinery. If it still fails, run the fallback SD1.5 protocol at 512-640. The experiment should not normalize a fragile, swap-heavy path.

Run the same 200-step gate for FLUX.2 Klein 4B Base only after the primary SDXL path is operational. Use the official 12 GB-oriented trainer settings, rank 16, 512 or 768 area, frozen/cached text features, and its recommended quantization.[^8][^9] Promote it for a later iteration only if it meets the same memory ceiling without block offload and its four sentinel outputs are at least as realistic as SDXL. This is a feasibility probe, not a second v0.1 platform.

### 6. Train and sweep checkpoints

Continue the SDXL run to **1,200 steps**, saving at steps 400, 600, 800, 1,000, and 1,200. Do not train the text encoders. Do not add prior-preservation images in run 1; the low-rank adapter, diverse captions, frozen benchmark, and checkpoint sweep keep the first causal comparison interpretable. If outputs show class leakage or severe pose/background binding, run 2 adds 100 generic adult-woman prior images while changing no other major variable. Prior preservation is a targeted remedy, not a default pile-on.[^24]

Checkpoint selection has two stages:

1. **Sentinel sweep:** 8 benchmark cases x 5 checkpoints x LoRA weights `0.7` and `1.0` = 80 outputs.
2. **Final sweep:** the best 2 checkpoints x all 20 cases x weights `0.6`, `0.8`, and `1.0` = 120 outputs.

Use the same seeds and compiled prompts for every competing identity method. Also render all 20 cases with no identity adapter and with PhotoMaker V2. The base-only set is a negative control; PhotoMaker is the substantive zero-shot baseline.

### 7. Frozen 20-image benchmark

All cases use everyday, non-revealing clothing and exactly one fictional adult. Prompts and seeds are frozen before LoRA training.

| ID | Shot | Camera profile |
|---|---|---|
| B01 | Close frontal, neutral, apartment window light | modern phone auto, 26 mm equivalent |
| B02 | Close left three-quarter, soft smile, open shade | modern phone auto |
| B03 | Close right three-quarter, serious, soft indoor light | modern phone auto |
| B04 | Left near-profile, wind moving a few hairs, overcast | modern phone auto |
| B05 | Right near-profile, hard noon side light | modern phone auto, restrained HDR |
| B06 | Just awake, slightly messy hair, bedroom morning | early phone, mild noise |
| B07 | Bathroom mirror selfie, cool fluorescent light | modern phone auto |
| B08 | Elevator mirror selfie, mixed overhead light | modern phone auto |
| B09 | Seated in a parked car, side-window daylight | modern phone auto |
| B10 | Kitchen snapshot, warm tungsten practicals | cheap compact, auto white balance |
| B11 | Office desk, flat fluorescent light | modern phone auto |
| B12 | Night sidewalk, storefront and street lighting | phone low-light, visible noise |
| B13 | Desk lit mostly by a monitor | early phone, high ISO |
| B14 | Plain hallway, direct on-camera flash | cheap compact, 35 mm equivalent |
| B15 | Waiting at a rainy bus stop, overcast | modern phone auto |
| B16 | Seated at a cafe table, candid half-smile | modern phone auto |
| B17 | Standing medium shot in an apartment hallway | cheap compact, ambient light |
| B18 | Slightly low, imperfect crop, neutral expression | early phone auto |
| B19 | Walking candid with mild subject motion blur | phone low-light |
| B20 | Face partly occluded by loose hair, side light | modern phone auto |

Store these as structured `ShotSpec` and `CameraSpec` records. The backend may translate them into prose, but neither record contains model-specific trigger syntax.

### 8. Evaluate identity, realism, and overfitting separately

Do not collapse the experiment to one score. Personalization benchmarks show that CLIP/DINO-style similarities can disagree with human judgment and over-reward copied shape or color; identity and prompt adherence form a trade-off rather than one axis.[^30]

**Identity diagnostics**

- Detect one face where a face is expected and record face area, yaw bin, and detector confidence.
- Compute AdaFace/CVLFace similarity to each canonical reference and to the canonical centroid. Report median, worst quartile, minimum, and values by front/three-quarter/profile view.
- Calibrate `tau_id` rather than importing a universal cosine threshold. Use canonical plus six held-out Mara images as positives and ten rejected candidate identities as negatives; choose and record the equal-error-rate threshold. If positive and negative distributions overlap heavily, mark the metric inconclusive.
- Report InsightFace similarity separately as a conditioner-adjacent diagnostic, never as the sole acceptance metric.
- Run a blind human same-person review on shuffled images.

**Realism and prompt diagnostics**

- Human reviewers score each image from 1 to 5: `1` obvious synthetic failure, `3` plausible only at a glance, `5` ordinary unretouched photograph with no salient synthesis cue.
- Tag face geometry, eyes/teeth, hair boundary, skin texture, hands if visible, reflections, lighting, background geometry, text, and camera plausibility.
- Score required pose, expression, framing, setting, lighting, and camera behavior separately. A recognizable Mara in the wrong shot is not a pass.

**Overfitting diagnostics**

- Compare every benchmark image to every training image with perceptual hash, LPIPS, and DINOv2 similarity.[^29]
- Flag exact/near-exact crops, duplicated backgrounds, repeated hair silhouettes, and suspiciously identical skin details for human inspection.
- Require unseen prompts and seeds, at least five settings, five lighting regimes, frontal/three-quarter/profile views, and three expression states.
- Do not use FID on a 20-image set and do not use an "AI detector" as an acceptance gate; neither answers the experiment's actual question at this sample size.

### 9. Acceptance criteria

The final selected method, checkpoint, and identity strength passes Mara v0.1 only if all of the following hold:

| Dimension | Pass condition |
|---|---|
| Set identity | At least 4 of 5 blind reviewers say the shuffled 20-image set depicts one woman |
| Per-image identity | At least 16/20 images receive a same-Mara majority; no required view or lighting category fails completely |
| Automated identity | At least 16/20 images meet the calibrated AdaFace/CVLFace threshold, with no more than one extreme outlier; metric must have separated calibration distributions |
| Face validity | Exactly one usable face in at least 19/20 images where one is expected |
| Realism | At least 16/20 score 4 or 5; median score at least 4; no more than two obvious plastic/CGI failures and no more than two severe visible anatomy failures |
| Prompt adherence | At least 18/20 satisfy pose, framing, setting, and lighting; at least 16/20 satisfy the intended camera behavior |
| Generalization | All prompts and seeds are unseen in training; required view, expression, setting, and lighting coverage is present; no benchmark image is judged a training-image copy |
| Reproducibility | 20/20 have complete lineage; three sentinel reruns are byte-identical on the pinned host, or any documented kernel-level drift is below the predeclared LPIPS tolerance |
| Operations | Every stage runs from one CLI command with config files; no GUI graph, cloud service, manual image editing, or steady-state memory swapping is required |
| Safety/scope | Every depicted person is clearly an adult; no nudity; no real-person or celebrity source imagery |

Body consistency, fixed outfits, tattoos, jewelry continuity, perfect hands, and cinematic beauty are explicitly not v0.1 pass criteria.

### 10. Decision rule

Select a winner on the **Pareto frontier of identity, realism, prompt adherence, and operational cost**. A LoRA wins only if it improves identity over PhotoMaker without materially reducing realism or shot diversity. If PhotoMaker passes and the LoRA does not, v0.1 can still succeed: Mara's persistent identity asset is the canonical set plus the pinned conditioner configuration, while LoRA training remains an unresolved experiment.

If both methods fail, locate the first divergence:

- failure in pass-A reference expansion means the master or conditioner is unsuitable;
- coherent references but a drifting dataset means generation/curation is unsuitable;
- coherent dataset but a failing adapter means the training recipe or base is unsuitable;
- automated pass but human failure means the evaluator is unsuitable.

Do not respond to failure by indiscriminately adding images, ranks, steps, or identity strength. Change one causal layer at a time.

## Expected cost and risks

On the stated GPU, the SDXL path should require roughly tens of minutes for candidate/reference generation, one to several hours for training, and tens of minutes for checkpoint sweeps; these are planning estimates and the smoke test must replace them with measured throughput. Human curation is likely to dominate elapsed effort. Model caches and experiment artifacts should fit comfortably inside 50 GB if rejected intermediates are stored as PNG/JPEG plus manifests rather than duplicated model environments.

The largest risks are:

1. **Synthetic-reference feedback:** generated defects become Mara. Mitigation: immutable master, no unreviewed recursive conditioning, and lineage on every image.
2. **Embedding circularity:** the conditioner grades itself. Mitigation: AdaFace/CVLFace primary evaluation and InsightFace secondary reporting.
3. **LoRA entanglement:** backgrounds, clothing, or pose become identity. Mitigation: balanced dataset, captions for controllable variables, held-out shot matrix, and checkpoint/weight sweeps.
4. **False 12 GiB compatibility:** a model starts only by paging or offloading. Mitigation: measured 10.5 GiB gate and throughput check.
5. **Adapter lock-in:** an identity asset silently depends on one base. Mitigation: canonical images are authoritative; adapters are versioned derived artifacts tied to model hashes.
6. **License migration:** research-only face-recognition weights enter a commercial path. Mitigation: record weight licenses in manifests and replace/audit InsightFace-dependent components before commercial use.

## Recommendation to the project team

Implement only the experiment spine described here. The first engineering milestone is not "generate Mara"; it is **produce a fully traced contact sheet from a frozen config and regenerate a selected image from its manifest**. The second is the 200-step SDXL memory smoke test. Only after those pass should the team spend time curating 28 images and running the complete benchmark.

Revisit the primary model when any of these becomes true:

- FLUX.2 Klein Base passes the local memory gate and beats SDXL on the frozen identity benchmark;
- a commercially permissive, non-InsightFace identity conditioner matches PhotoMaker's quality;
- the project adds full-body or multi-character continuity, where SDXL's current advantage may not hold;
- RealVisXL's license or exact weight provenance conflicts with deployment requirements.

This approach leaves Mara portable even if every inference and training backend changes: her canon and experiment history remain stable, while conditioners and adapters can be replaced and re-evaluated against the same benchmark.

## Sources

[^1]: SG161222, [RealVisXL V5.0 model card](https://huggingface.co/SG161222/RealVisXL_V5.0), Hugging Face. Creator documentation for photorealism focus, SDXL lineage, inference guidance, and OpenRAIL++ terms.
[^2]: Stability AI, [Stable Diffusion XL Base 1.0 model card](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), Hugging Face. Architecture, resolution, limitations, and license.
[^3]: Hugging Face Diffusers, [SDXL DreamBooth LoRA training guide](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sdxl.md). Documents sub-16 GB training using mixed precision, checkpointing, memory-efficient attention, 8-bit Adam, and cached text outputs.
[^4]: SG161222, [Realistic Vision V6.0 B1 model card](https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE), Hugging Face.
[^5]: Runway/Stable Diffusion community mirror, [Stable Diffusion v1.5 model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), Hugging Face. Architecture, 512 px training resolution, limitations, and OpenRAIL-M terms.
[^6]: Hugging Face Diffusers, [LoRA training documentation](https://huggingface.co/docs/diffusers/v0.21.0/training/lora). Reports full LoRA training on an 11 GB RTX 2080 Ti without memory-saving options.
[^7]: Black Forest Labs, [FLUX.2 Klein](https://bfl.ai/models/flux-2-klein). Official model variants, inference-memory figures, latency figures, editing capabilities, and licensing.
[^8]: Black Forest Labs, [FLUX.2 Klein training guide](https://docs.bfl.ai/flux_2/flux2_klein_training). Official LoRA training scope and 12 GB VRAM / 32 GB RAM minimum guidance.
[^9]: SimpleTuner, [FLUX.2 training quickstart](https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/FLUX2.md). End-to-end memory guidance for BF16 and int8 Klein configurations.
[^10]: Black Forest Labs and Hugging Face, [Train your own FLUX.2 Klein LoRA](https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora). Practical standard-LoRA memory and RTX 4090 training guidance.
[^11]: Stability AI, [Stable Diffusion 3.5 Medium model card](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium), Hugging Face. Architecture, quantized inference example, and Community License.
[^12]: SimpleTuner, [Stable Diffusion 3 training quickstart](https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/SD3.md). Low-VRAM NF4 training constraints and practical memory guidance.
[^13]: NVIDIA Research, [SANA repository and documentation](https://github.com/NVlabs/Sana). Official inference memory and training hardware guidance.
[^14]: SimpleTuner, [SANA training quickstart](https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/SANA.md). Practical minimum/recommended trainer memory.
[^15]: Alpha-VLLM, [Lumina-Image 2.0 repository](https://github.com/Alpha-VLLM/Lumina-Image-2.0). Official 2.6B model architecture, Apache 2.0 license, inference and training entry points.
[^16]: SimpleTuner, [Lumina-Image 2.0 training quickstart](https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/LUMINA2.md). Practical LoRA memory guidance.
[^17]: Tongyi-MAI, [Z-Image model card](https://huggingface.co/Tongyi-MAI/Z-Image). Official 6B base model, resolutions, training suitability, and Apache 2.0 license.
[^18]: SimpleTuner, [Z-Image training quickstart](https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/ZIMAGE.md). BF16, int8, and NF4 memory ranges.
[^19]: Kandinsky Lab, [Kandinsky 5 repository](https://github.com/kandinskylab/kandinsky-5), and SimpleTuner, [Kandinsky 5 Image training guide](https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/KANDINSKY5_IMAGE.md). Architecture and practical training memory.
[^20]: ZLab Princeton, [i1 repository](https://github.com/zlab-princeton/i1), and SimpleTuner, [i1 training guide](https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/ZLAB_i1.md). Open training recipe and practical 1024 px LoRA memory.
[^21]: RunDiffusion, [Juggernaut XL v9 model card](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9), Hugging Face. Inference claims, SDXL lineage, and additional usage terms.
[^22]: TencentARC, [PhotoMaker repository](https://github.com/TencentARC/PhotoMaker) and [PhotoMaker V2 model card](https://huggingface.co/TencentARC/PhotoMaker-V2). Identity workflow, SDXL integration, V2 requirements, and memory guidance.
[^23]: Zhen Li et al., [PhotoMaker: Customizing Realistic Human Photos via Stacked ID Embedding](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_PhotoMaker_Customizing_Realistic_Human_Photos_via_Stacked_ID_Embedding_CVPR_2024_paper.pdf), CVPR 2024.
[^24]: Nataniel Ruiz et al., [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://openaccess.thecvf.com/content/CVPR2023/papers/Ruiz_DreamBooth_Fine_Tuning_Text-to-Image_Diffusion_Models_for_Subject-Driven_Generation_CVPR_2023_paper.pdf), CVPR 2023. Context entanglement and prior-preservation rationale.
[^25]: DeepInsight, [InsightFace repository and licensing notice](https://github.com/deepinsight/insightface). Distinguishes MIT code from non-commercial-research pretrained model packs.
[^26]: InstantX, [InstantID repository](https://github.com/instantX-research/InstantID). Zero-shot SDXL identity conditioning and checkpoint restrictions.
[^27]: kohya-ss, [SDXL network/LoRA training documentation](https://github.com/kohya-ss/sd-scripts/blob/main/docs/sdxl_train_network.md). Caching, precision, and gradient-checkpointing guidance.
[^28]: Minchul Kim et al., [CVLFace repository](https://github.com/mk-minchul/CVLface). AdaFace-family face-recognition implementations and pretrained evaluators.
[^29]: Meta AI Research, [DINOv2 repository](https://github.com/facebookresearch/dinov2). General-purpose visual features and Apache 2.0 code/weights.
[^30]: DreamBench++ authors, [DreamBench++: A Human-Aligned Benchmark for Personalized Image Generation](https://arxiv.org/abs/2406.16855), 2024. Evidence that common automatic personalization metrics do not fully align with human judgments.
