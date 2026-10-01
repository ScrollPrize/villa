# Reviewed-reference beam tracing — 2026-09-30

## Corrected annotations: v3 supersedes v2

After user correction, all91 source files carry `hendrik_hq_approved` and none
contain Lasagna spans. New byte-identical cohort/results:
`/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3`.
All91 now yield one eligible piece each (182directions); no exclusions. Eligible
length693.087803mm, directed1386.175605mm. Same models/protocol/cache parameters,
including frozen Gaussianu17700, not a refreshed live training checkpoint.
GPUs0–5 only. Tables below for v2 are historical and not matched v3 comparisons.

V3 all complete:lowres29 restarts/46.206mm mean,previous fullres36/37.464mm,
Gaussian32/42.005mm. No infrastructure failures. Current full per-scroll tables
and exact results: [corrected-cohort report](reviewed_fibers_corrected_20260930.md).

## Cohort and protocol

Source: `/home/hendrik/data/2026-09-30_fibers_approved_refined/`.
Completed transfer:91 annotations (PHerc0139=22,PHerc0332=1,PHerc1299=11,
PHerc1451=1,PHercParis4=56). Do not supplement with older references.
Freeze byte-identical annotations and SHA256 inventory under
`/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v2/cohort`.
New/modified source files require a new version, not a silent resume.

Use the existing native reference replay evaluator, both directions, beam8,
lookahead2, step4 prediction voxels, no embedding (historical model-specific
prediction scales retained). Acceptance thickness40/width80 base voxels,
excursion grace160, existing restart-episode filtering and CP/Lasagna eligibility.
Keep existing inference producer settings and namespaces; all eight GPUs are
originally configured through demand-scoped manager ownership, with192 CPU trace workers.
User restricted evaluations to GPUs0–5 before lowres completion. Lowres completed
entirely from cache (no worker pool), so used no GPUs. Subsequent runs request0–5 only.
No rendering, parameter tuning or new scheduler implementation.

## Models

1. Best lowres run `20260915_212757`, EMA999; frozen SHA256
   `67edd3a1bde162c7105ee37d849a0f0b60a9b840dd37fc7ff6e17e0c3a2e9742`.
2. Previous fullres run `20260916_213753`, best EMA999; frozen SHA256
   `5ec278f71b15b27187b55805310176fe2fa3b00c5264fc528b95bc6d0690206e`.
3. Gaussian fullres run `20260929_092844`, `best_ema_0p999.pt`;
   EMA999 update17700,2265600 samples, selected by test_ema_0p999/loss_total
   (0.4442611531). Frozen SHA256
   `274acb0c686bfb030646d039f2740f7ad6349cd507a4de0843cf26e4f638ccec`.

Each model directory contains workload.json, full_bindings.json, full.json,
and full/summary.json. Workloads derive from the committed training-mining
lowres/fullres configs; inference parameters stay unchanged.

## Reproduction

Set scripts/training_failure_mining_control.json to
`{"action":"reviewed_eval","reviewed_model":"lowres"}` (or fullres/gaussian), then:

```bash
bash /home/hendrik/villa3/scripts/run_training_failure_mining
```

This delegates prepare, validate-full and full to the existing evaluation runner.
No mining is launched. Existing completed results are preserved and reused.
The previously completed bidirectional pilot remains the established smoke gate.

## Validation and interpretation

Seven focused Python tests pass (cohort byte preservation/resume, changed/invalid
input rejection, v3/v4 editor hints, completion and workload identity checks).
Final workspace RelWithDebInfo build:23 replay,86 tracing,24 reference tests,
plus4 coordinate-contract CLI tests pass. Includes v4 mixed-span eligibility.
Independent plan review approved reuse with frozen source/model provenance.

This is a selected, reviewed hard-region cohort, not the old50-fiber evaluation
and not an independently established training holdout. Report only matched
comparisons on these annotations; training/selection overlap is not ruled out.
Report total reference length once and evaluated length across both directions
separately. Failures are filtered restart episodes, not all visible retry events.

## Results

Initial attempt blocked before any tracing: the source file
`PHerc0139/0005_kb_20260901T224443942_000068_run1.json` is truncated
at32768bytes (unterminated string, line880). Transfer now complete and validated.
Initial v1 contains preparation only, no results; complete cohort uses v2.
Shared readers now accept validated optional CP direction/display-normal fields
and version4 object CPs without changing stored reference geometry/eligibility.
Source annotations and frozen copies retain all fields unchanged.

### Lowres212757 best EMA999

Completed194/194 directed cases (97 eligible runs from91 annotations), no
infrastructure failures. Eligible reference length686.126621mm, directed
length1372.253241mm. Native tracing12.157s; inference/cache coordinator23.539s,
all predictions reused, no GPU worker pool. These are warm-cache timings.
One PHerc0139 annotation has no eligible span under the unchanged policy.
That annotation is `0015_lt_20260904T062829219_000052_run2.json`:
two CPs, its sole segment explicitly marked `interp_mode=lasagna`.

| Scroll | Restarts | Failures/mm | Mean mm | Median mm | P5 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|
| PHerc0139 | 8 | 0.02889 | 30.773 | 3.598 | 1.373 | 9.690 |
| PHerc0332 | 0 | 0 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc1299 | 4 | 0.02713 | 29.485 | 4.702 | 1.957 | 15.539 |
| PHerc1451 | 0 | 0 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHercParis4 | 16 | 0.01727 | 54.485 | 5.266 | 1.448 | 24.289 |
| Combined | 28 | 0.02040 | 47.319 | 4.887 | 1.421 | 17.775 |

Mean is combined directed distance/(restarts+1); quantiles use individual
reference partitions including successful terminal segments, unchanged.

### Previous fullres213753 best EMA999

Completed194/194 cases,30 restarts,zero infrastructure failures; same reference
length as lowres. Native29.467s/coordinator40.951s,18 new inference tiles in6
batches; remaining prediction data reused. GPU requests restricted to0–5.

| Scroll | Restarts | Failures/mm | Mean mm | Median mm | P5 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|
| PHerc0139 | 10 | 0.03611 | 25.178 | 3.420 | 1.378 | 9.605 |
| PHerc0332 | 0 | 0 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc1299 | 4 | 0.02713 | 29.485 | 4.811 | 1.535 | 15.539 |
| PHerc1451 | 0 | 0 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHercParis4 | 16 | 0.01727 | 54.485 | 5.187 | 1.484 | 24.660 |
| Combined | 30 | 0.02186 | 44.266 | 4.778 | 1.397 | 17.777 |

Two additional restarts relative to lowres, both in0139.

### Gaussian fullres092844 best EMA999 update17700

Completed194/194 cases,32 restarts,zero infrastructure failures. Native217.056s
(includes inference waits),coordinator228.780s.4942 tiles/1295 batches with a
fresh model cache; cannot compare timing directly to warm historical models.
GPU0–5 only; GPUs6/7 observed empty. Same cohort hashes, normal-manifest hashes
and tracing protocol verified exactly across all three results.

| Scroll | Restarts | Failures/mm | Mean mm | Median mm | P5 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|
| PHerc0139 | 11 | 0.03972 | 23.080 | 3.525 | 1.371 | 8.883 |
| PHerc0332 | 0 | 0 | 11.129 | 5.564 | 5.564 | 5.564 |
| PHerc1299 | 4 | 0.02713 | 29.485 | 4.811 | 1.003 | 15.539 |
| PHerc1451 | 0 | 0 | 10.496 | 5.248 | 5.248 | 5.248 |
| PHercParis4 | 17 | 0.01835 | 51.458 | 5.159 | 1.503 | 24.565 |
| Combined | 32 | 0.02332 | 41.583 | 4.763 | 1.371 | 17.777 |

### Matched combined comparison

| Model | Restarts | Mean mm | Median mm | P5 mm | P95 mm |
|---|---:|---:|---:|---:|---:|
| Lowres212757 best EMA999 | 28 | 47.319 | 4.887 | 1.421 | 17.775 |
| Fullres213753 best EMA999 | 30 | 44.266 | 4.778 | 1.397 | 17.777 |
| Gaussian092844 best EMA999 u17700 | 32 | 41.583 | 4.763 | 1.371 | 17.777 |

Lowres retains the fewest restarts on this selected cohort. Gaussian adds four
relative to lowres (three0139,oneParis4),14.29% more; adds two relative to prior
fullres. No parameter tuning performed. Exact machine-readable values and
source summary paths: [reviewed_fibers_20260930.json](reviewed_fibers_20260930.json).
