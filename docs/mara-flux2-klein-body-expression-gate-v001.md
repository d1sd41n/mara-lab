# Mara FLUX.2 Body And Expression Gate v0.1

## Objective

Decide whether FLUX.2 Klein's two-reference mode is sufficient for Mara, or
whether the project should invest in a new LoRA. The gate freezes 24 difficult
scenes instead of judging a few attractive portraits.

| Block | Cases | Stress tested |
| --- | ---: | --- |
| Full body and viewpoint | 6 | Front, rear turn, strict profile, low angle, high angle, one-leg balance |
| Seated and dynamic anatomy | 6 | Chair, sofa, floor, crouch, overhead reach, weighted bags |
| Hands and objects | 4 | Pouring, writing, opening a jar, folding clothes |
| Face and expression | 8 | Profile, over-shoulder, broad laugh, squint, closed eyes, concern, surprise, wind |

All cases use canonical candidate `#0063` as the master and `#0055` as the
backup, seeds `47101` through `47124`, 640 x 832 output, four inference steps,
and guidance 1.0.

## Result

The run produced all 24 images in 144.95 seconds and reserved 7.06 GiB of VRAM
on the RTX 4070 SUPER. Body proportions remain plausible across the full-body,
seated, crouched, and reaching cases. Hands and object interactions are also
materially stronger than the old SDXL path, with one clear failure in the
pouring scene where fingers around the glass become anatomically ambiguous.

Expression range passes. Mara can now close her eyes, squint, frown, laugh, and
look away instead of repeating one wide-eyed frontal expression. The strongest
expressions also reveal the limiting factor: the broad laugh changes identity,
and the rear turn, strict profile, and over-shoulder view move facial structure
or apparent age away from the canonical references.

## Supporting Identity Measurement

The reproducible InsightFace report detected one face in every output. Cosine
similarity to the normalized two-reference centroid ranged from 0.0027 to
0.6032, with a median of 0.3638 and mean of 0.3482. The easier ten-image pilot
had a nearly identical mean of 0.3466.

These numbers are recorded for comparison, not used as a universal pass/fail
threshold. The two canonical references score only 0.3934 against each other,
and strict profiles are strongly penalized by the recognizer. Visual review is
therefore still required.

## Decision

FLUX.2 Klein passes the body, scene, photographic-quality, and expression
gates. Reference-only identity control does not pass a production gate.
Proceed to a low-memory FLUX.2 Base LoRA feasibility spike and compare every
candidate adapter against this exact frozen suite.

Do not train on the visibly drifted benchmark outputs. The first training spike
should use a small identity-clean set, prove that training fits the 12 GiB GPU,
and test whether identity improves without collapsing the expression and body
flexibility gained here.
