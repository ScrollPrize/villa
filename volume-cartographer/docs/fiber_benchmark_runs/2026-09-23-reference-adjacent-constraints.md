# 1024³ crop: constraints between consecutive reference fibers

## Finding

All 25 adjacent orthogonal reference pairs are connected with correct sign/order.
Only 10 of 24 adjacent parallel pairs have any direct constraint; one of these
has no target-1 measurement after scaling, leaving just 9 of 24 with target-1
support under the current reduced policy.

| Diagnostic | Orthogonal, GT step 0.5 | Parallel, GT step 1.0 |
| --- | ---: | ---: |
| Pairs with any measurement | 25/25 (100%) | 10/24 (41.67%) |
| Missing pairs | 0/25 (0%) | 14/24 (58.33%) |
| Piece-pair measurements | 103 | 156 |
| Correct unscaled quantized distance | 72/103 (69.90%) | 105/156 (67.31%) |
| Correct distance after current 0.822 scaling | 95/103 (92.23%) | 119/156 (76.28%) |
| Wrong direction classification | 0 | 0 |
| Wrong sign/order | 0 | 0 |

Counts are unique extracted piece-pair measurements, **not independent fibers**
or percentages of fiber length. Several measurements can connect the same
reference pair. Zero count means no direct extracted relation; it does not rule
out an indirect path through other fibers.

The present reduced loss uses hard orthogonal order only, not orthogonal
distance. Therefore the eight overestimated orthogonal distances below do not
add an incorrect distance penalty in this experiment. Parallel magnitude has
weight only for target 1, so measured target 0/2/3 terms are excluded (parallel
sign is independently disabled). Numerical weights also include existing
confidence/distance scaling; the tables below concern target classes, not
summed confidence or loss.

## Method / reproducibility

Existing detailed diagnostic, no code changes and no inference or tracing.
Reused the corrected small-crop config and appended
`--reference-constraint-details`; output prefix:

`/ephemeral/hendrik/fiber_crop_bench/20260923_ref_pair_diagnostics/bp1024-reference-pairs`

The `.run.json` records full command, binary/library SHA, source revision,
reference/input hashes and timing; `.source.patch` captures code state.
Completed with return code 0 in 50.9032 s. The existing CLI recomputes BP while
printing details; no dedicated new extraction implementation was introduced.

Same 26 references, normal volume, piece length 512, 1024³ crop and weight policy
as [the corrected control](2026-09-23-inactive-winding-factors.md).
Use the **first** reference diagnostic block, before oracle pruning.
There are 513 total GT-to-GT extracted measurements over 93 source pairs;
259 measurements belong to the 49 adjacent pairs discussed here.

GT labels are the established filename-order half-step assignment.
“Raw targets” below means quantization before measurement-scale correction,
with the existing global normal-sign calibration already applied.
“Scaled targets” applies the configured 0.822 scale before the native
integer / half-integer quantization. The detailed CLI prints calibrated values
to three decimals; none of these 259 adjacent measurements is close enough to a
quantization boundary for that rounding to change its target classification.

## Orthogonal adjacent pairs

| Pair | N | Wrong class | Raw targets (value:count) | Scaled targets (value:count) | Correct % | Wrong order |
|---|---:|---:|---|---|---:|---:|
| 0–0.5 | 4 | 0 | 0.5:4 | 0.5:4 | 100.0% | 0 |
| 0.5–1 | 5 | 0 | 0.5:1, 1.5:4 | 0.5:4, 1.5:1 | 80.0% | 0 |
| 1–1.5 | 6 | 0 | 0.5:6 | 0.5:6 | 100.0% | 0 |
| 1.5–2 | 4 | 0 | 1.5:4 | 0.5:4 | 100.0% | 0 |
| 2–2.5 | 4 | 0 | 0.5:3, 1.5:1 | 0.5:4 | 100.0% | 0 |
| 2.5–3 | 4 | 0 | 0.5:4 | 0.5:4 | 100.0% | 0 |
| 3–3.5 | 4 | 0 | 0.5:3, 1.5:1 | 0.5:4 | 100.0% | 0 |
| 3.5–4 | 4 | 0 | 1.5:4 | 1.5:4 | 0.0% | 0 |
| 4–4.5 | 4 | 0 | 0.5:4 | 0.5:4 | 100.0% | 0 |
| 4.5–5 | 4 | 0 | 0.5:1, 1.5:3 | 0.5:4 | 100.0% | 0 |
| 5–5.5 | 3 | 0 | 0.5:3 | 0.5:3 | 100.0% | 0 |
| 5.5–6 | 4 | 0 | 0.5:1, 1.5:3 | 0.5:4 | 100.0% | 0 |
| 6–6.5 | 5 | 0 | 0.5:3, 1.5:2 | 0.5:5 | 100.0% | 0 |
| 6.5–7 | 4 | 0 | 1.5:4 | 0.5:1, 1.5:3 | 25.0% | 0 |
| 7–7.5 | 4 | 0 | 0.5:4 | 0.5:4 | 100.0% | 0 |
| 7.5–8 | 4 | 0 | 0.5:4 | 0.5:4 | 100.0% | 0 |
| 8–8.5 | 4 | 0 | 0.5:4 | 0.5:4 | 100.0% | 0 |
| 8.5–9 | 4 | 0 | 0.5:4 | 0.5:4 | 100.0% | 0 |
| 9–9.5 | 5 | 0 | 0.5:4, 1.5:1 | 0.5:5 | 100.0% | 0 |
| 9.5–10 | 3 | 0 | 0.5:3 | 0.5:3 | 100.0% | 0 |
| 10–10.5 | 4 | 0 | 0.5:4 | 0.5:4 | 100.0% | 0 |
| 10.5–11 | 4 | 0 | 0.5:4 | 0.5:4 | 100.0% | 0 |
| 11–11.5 | 4 | 0 | 1.5:4 | 0.5:4 | 100.0% | 0 |
| 11.5–12 | 4 | 0 | 0.5:4 | 0.5:4 | 100.0% | 0 |
| 12–12.5 | 4 | 0 | 0.5:4 | 0.5:4 | 100.0% | 0 |

Only three pairs contain wrong distances after scaling:

- 0.5–1: one of five measurements says 1.5 instead of 0.5.
- 3.5–4: all four say 1.5 instead of 0.5.
- 6.5–7: three of four say 1.5 instead of 0.5.

All 103 have the correct order. In the current hard-order-only policy all these
pairs retain usable order evidence; these magnitude errors matter only if
orthogonal magnitude is enabled.

## Parallel adjacent pairs

| Pair | N | Wrong class | Raw targets (value:count) | Scaled targets (value:count) | Correct % | Wrong order |
|---|---:|---:|---|---|---:|---:|
| 0–1 | 22 | 0 | 1:10, 2:12 | 1:14, 2:8 | 63.6% | 0 |
| 0.5–1.5 | 16 | 0 | 1:6, 2:10 | 1:7, 2:9 | 43.8% | 0 |
| 1–2 | 0 | 0 | — | — | — | 0 |
| 1.5–2.5 | 0 | 0 | — | — | — | 0 |
| 2–3 | 13 | 0 | 1:13 | 1:13 | 100.0% | 0 |
| 2.5–3.5 | 16 | 0 | 1:13, 2:3 | 1:16 | 100.0% | 0 |
| 3–4 | 0 | 0 | — | — | — | 0 |
| 3.5–4.5 | 0 | 0 | — | — | — | 0 |
| 4–5 | 0 | 0 | — | — | — | 0 |
| 4.5–5.5 | 0 | 0 | — | — | — | 0 |
| 5–6 | 0 | 0 | — | — | — | 0 |
| 5.5–6.5 | 19 | 0 | 0:3, 1:16 | 0:6, 1:13 | 68.4% | 0 |
| 6–7 | 0 | 0 | — | — | — | 0 |
| 6.5–7.5 | 19 | 0 | 1:19 | 0:1, 1:18 | 94.7% | 0 |
| 7–8 | 0 | 0 | — | — | — | 0 |
| 7.5–8.5 | 0 | 0 | — | — | — | 0 |
| 8–9 | 0 | 0 | — | — | — | 0 |
| 8.5–9.5 | 16 | 0 | 1:13, 2:3 | 1:16 | 100.0% | 0 |
| 9–10 | 0 | 0 | — | — | — | 0 |
| 9.5–10.5 | 0 | 0 | — | — | — | 0 |
| 10–11 | 15 | 0 | 2:15 | 1:7, 2:8 | 46.7% | 0 |
| 10.5–11.5 | 4 | 0 | 3:4 | 2:2, 3:2 | 0.0% | 0 |
| 11–12 | 0 | 0 | — | — | — | 0 |
| 11.5–12.5 | 16 | 0 | 1:15, 2:1 | 0:1, 1:15 | 93.8% | 0 |

Missing direct pairs:
1–2, 1.5–2.5, 3–4, 3.5–4.5, 4–5, 4.5–5.5, 5–6, 6–7,
7–8, 7.5–8.5, 8–9, 9–10, 9.5–10.5, 11–12.

10.5–11.5 is present geometrically but all four measurements say 2 or 3 after
scaling, so none supplies the currently enabled target-1 loss.

## Relation to observed winding failures

The corrected 1024 result is wrong at GT 8→7, 11.5→12.5 and 12→8.

- GT 8 has **neither** adjacent parallel link: 7–8 and 8–9 are missing.
- GT 12 lacks 11–12; the next parallel reference, 13, is outside this cohort.
- GT 11.5 lacks usable target-1 evidence toward 10.5, although 11.5–12.5 has
  15 correct target-1 measurements out of 16.

This supports investigating missing/overestimated parallel links. It is not a
causal proof of the beam result: these measurements are extracted between
**reference geometry**, whereas the ordinary solver operates on predicted
beam-trace pieces and reference evaluation uses cross-constraints.
The fully correct adjacent orthogonal order chain on the GT references would
reject some reported wrong assignments if imposed directly on those references.
Thus “GT-to-GT constraint exists” cannot be equated with “equivalent correct
beam-to-beam/cross evidence is present and retained.”

## Reference identity

| Winding | Reference JSON stem |
| --- | --- |
| 0 | waldkauz_20260828T064452831_000002 |
| 0.5 | waldkauz_20260828T064812002_000003 |
| 1 | waldkauz_20260828T065143381_000004 |
| 1.5 | waldkauz_20260828T065643794_000005 |
| 2 | waldkauz_20260828T072253909_000006 |
| 2.5 | waldkauz_20260828T072717899_000007 |
| 3 | waldkauz_20260828T073017594_000008 |
| 3.5 | waldkauz_20260828T074703971_000009 |
| 4 | waldkauz_20260901T102631471_000010 |
| 4.5 | waldkauz_20260901T105412524_000011 |
| 5 | waldkauz_20260901T105602115_000012 |
| 5.5 | waldkauz_20260901T160540656_000013 |
| 6 | waldkauz_20260901T160642323_000014 |
| 6.5 | waldkauz_20260901T160716356_000015 |
| 7 | waldkauz_20260901T160812289_000016 |
| 7.5 | waldkauz_20260901T161032692_000017 |
| 8 | waldkauz_20260901T161219869_000018 |
| 8.5 | waldkauz_20260901T162005620_000019 |
| 9 | waldkauz_20260901T162100631_000020 |
| 9.5 | waldkauz_20260901T162112825_000021 |
| 10 | waldkauz_20260901T162212264_000022 |
| 10.5 | waldkauz_20260901T162324568_000023 |
| 11 | waldkauz_20260901T162413303_000024 |
| 11.5 | waldkauz_20260901T162425442_000025 |
| 12 | waldkauz_20260901T162507413_000026 |
| 12.5 | waldkauz_20260901T162541128_000027 |

