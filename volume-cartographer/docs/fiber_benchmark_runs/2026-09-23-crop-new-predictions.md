# Paris4 crop: new predictions versus Classic (2026-09-23)

## Scope and reproducibility

Base-XYZ crop `[10240,22016,6144] .. [11264,23040,7168)`;
references `/home/hendrik/test_datasets/2026-09-01_fiber_stack2`, tag
`hendrik_crop1`: 26 sources, 24 in-crop runs, 48 directed cases,
101.035764287893 mm directed length (2.4 µm/base voxel).

New prediction: Lasagna run `7ff0ce6c`, best EMA999 u22000 from
`5scroll_json_eligible_refinement_128_sd1_16x8_ema_20260915_212757`,
checkpoint SHA256 `67edd3a1bde162c7105ee37d849a0f0b60a9b840dd37fc7ff6e17e0c3a2e9742`.
Classic prediction: `d74cca79`; normal prediction: `5b17ff6c`.
Classic combined fiberlets `e7aa1a99` must not be used for new-model results.

Build: workspace `volume-cartographer/build/fiber-evaluation`, GCC13,
RelWithDebInfo. Initial source revision `46850d3f064de92e2a88568d745abf61265f37b7`
plus recorded crop integration patches. One repetition, CPU only; timings are
observations, not controlled speed comparisons with historical hardware/builds.

Artifacts: `/ephemeral/hendrik/fiber_crop_bench/20260923_v1/`.
Each command has `.log`, `.resources.txt`, `.run.json` with command, binary SHA,
reference/manifest SHA, revision and timing; later commands also include
`.source.patch`. Native replay results are `.json`.

## Completed replay results

| Prediction / route | Failures | Directions with failures / 48 | Mean segment, mm | Historical score, % | Wall, s | User, s | System, s | Peak RSS, KiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Classic direct greedy | 13 | 12 | 7.21684 | 7.14286 | 2.55 | 12.39 | 56.81 | 722176 |
| New direct greedy | 14 | 9 | 6.73572 | 6.66667 | 1.89 | 10.84 | 58.31 | 725760 |
| **Classic VC3D beam** | **1** | **1** | **50.51788** | **50.00000** | 1.69 | 26.35 | 87.18 | 1179136 |
| **New VC3D beam** | **2** | **2** | **33.67859** | **33.33333** | 1.72 | 29.20 | 89.03 | 1159236 |
| Classic fiberlet (supplied v2) | 6 | 5 | 14.43368 | 14.28571 | 28.08 | 127.23 | 86.29 | 18927652 |
| Classic fiberlet (regenerated v3, includes generation) | 6 | 5 | 14.43368 | 14.28571 | 657.51 | 49112.96 | 2179.04 | 645968336 |
| New fiberlet (includes generation) | 39 | 9 | 2.52589 | 2.50000 | 287.21 | 25725.42 | 290.06 | 171042384 |

All cases completed; the new fiberlet route credited 91.579851 / 101.035764 mm
(90.6410%), versus 100% for the other three rows. Its failures comprise 19
distance violations, 10 graph-exhausted, 6 missing-seed-gap and 4 no-usable-seed
events. Corpus reference length is not the same as credited trace length.
The two beam rows use actual VC3D defaults: width 8, lookahead 2, step 4 trace
voxels (8 base voxels), cone 25 degrees, inference exponent 2, no embeddings.
The GUI does not override these defaults. No unrelated full-resolution sweep
parameters were used. Greedy rows retain width 1/lookahead 1 only as controls.

The sole headline metric is the documented **100/(failures+1)**. The conflicting
per-direction denominator was removed from the native implementation and this
task's existing JSON summaries were corrected programmatically without changing
any failures or case geometry. Schema v6 identifies the restored corpus metric.

Historical controls: Classic direct 13 failures / 7.217 mm, unstaged fiberlet
6 / 14.434 mm, staged fiberlet 7 / 12.629 mm. The current controls reproduce
their total failure counts, not necessarily the distribution among directions.

## Commands

Build with the existing approved runner:

```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
```

Run each committed config through the existing approved evaluation entrypoint:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_direct_20260923.json
```

Other configs in that directory: `crop_benchmark_classic_fiberlet_20260923.json`,
`crop_benchmark_new_fiberlet_20260923.json`, `crop_benchmark_new_bp_20260923.json`,
and `crop_benchmark_native_tests_20260923.json`.

For the corrected VC3D beam comparison, use `crop_benchmark_beam_20260923.json`.
Run `crop_benchmark_publish_20260923.json` through the same entrypoint to update
the existing historical plot data and SVGs from recorded results. That command
also restores the historical metric in this task's older replay summaries.

Additional controls: `crop_benchmark_classic_bp_20260923.json`,
`crop_benchmark_unassisted_bp_20260923.json`,
`crop_benchmark_classic_regenerated_20260923.json`, and
`crop_benchmark_classic_regenerated_bp_20260923.json`.

## BP protocol

BP is a winding/constraint benchmark, not restart-distance evaluation. Match
the historical ambiguity-margin 0.40, radius 20, quality threshold 0.21,
piece length 512 and sum-product-mixed protocol. Historical reference-conditioned
diagnostic: 24 exact / 1 wrong / 1 missing of 26 (92.31%). This uses references
in optimization and is not a held-out accuracy claim. See
[the historical run](2026-09-03-07dba8fef-ambiguity-m040-q021.md).

Current supplied-Classic reference-assisted round zero: 22 exact, 3 wrong,
1 missing (84.62%). New: 23 exact, 2 wrong, 1 missing (88.46%). Neither reached
zero required errors; both stopped at the configured one-round limit.
Unassisted BP on all retained pieces gives supplied Classic 11 exact / 14 wrong /
1 missing (42.31%) and new 12 / 13 / 1 (46.15%); both hit the configured 500-message
limit. They match the initial unconditioned solutions recorded by the oracle
diagnostic. Gauge alignment uses reference labels for scoring, but no reference
pruning or clamped-reference optimization is applied in these rows.
Unassisted wall times were 41.61 s and 13.85 s. Regenerated Classic comparison
uses the supplied v2 cache here. Historical normals were `las008_s1_full`, not the current manifest.

Classic crop generation accepted 2895 traces in 148.21 s; new accepted 674 in
28.81 s (both warm fiberlet caches). BP including oracle diagnostics took
55.58 s and 14.84 s respectively. These are one-shot timings; generation and
other control jobs overlapped, so they are not isolated speed measurements.

## Generation-version control

The supplied Classic cache has generation contract v2. Current new-model
generation is v3. Scientific anchor/path parameter values and prediction scale
match, but v3 fixes authoritative producer consistency and resamples endpoint
evidence for graph scoring. Therefore the supplied-Classic/new comparison is
not solely a model ablation. Regenerated Classic v3 nevertheless reproduced
6 failures (14.286%) through the identical on-demand path. No existing cache was overwritten.

## Validation and progress

Native replay suite: 18 tests passed; storage suite: 43 tests passed; Python
recording/migration runner: 4 tests passed; plot tests: 7 passed. Independent final review found no blocking
correctness issue. First generation integration attempt rejected mismatched
storage profiles before tracing; fixed by supplying the canonical quantized
anchor view, not relaxing compatibility checks. Prior attempts are archived.

Explicit limitation: generated split caches do not support transient `--stage`
filters; those retain the existing Combined-dataset route. This benchmark did
not request a new staged sweep. No GPU or new neural inference was needed.

Final approved inspection runner: native suites 23+86+18+14 passed;
Python 298 passed, 8 skipped, 2 failures in unchanged scratch-cleanup and
demand-GPU cache-validation tests. The replay metric/eligible-span tests pass.
