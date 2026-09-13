# Embedding-only tracer tuning — frozen paired update 4300

## Result

Best tested: weight **0.25**, history spacing **128 base voxels**, window **1024 base voxels**, count **8**, **Gaussian** falloff.
**82 restarts (38 forward /44 reverse)** versus86 embedding-off and90 original embedding settings:
4.65% fewer than off;8.89% fewer than original. Final replay matched exactly.
Counts8/16/32/64 give the identical replay hash; count8 is the smallest tested matching cap.

This is a local tuning result on one hard held-out fiber, not a global optimum or an independent test-set gain.
No production defaults, training settings, geometric step, beam settings, or other costs changed.
The selected count/window/spacing limit history naturally; count64 does not imply64 references are available.
Gaussian weights are exp(-2*(age/window)^2), normalized by weight sum; references reset with replay segments.

## Frozen data and execution

Raw update4300 /550400 patches; SHA
`ee272228afaee9f4163338b5d803338f834ddd7bdfcbdb6e34c03a08e76b6f0f`.
PHerc1299 (dataset label PHerc1229), held-out
`lt_20260825T075316065_000069.json`, whole fiber both directions.
Base voxel2.399um: spacing128=0.307mm, window1024=2.457mm.
InputL1, BF16 model, tile256/border32/overlap96, signed FP32 embedding cache.
Native dev-quickbuild-gcc binary, unchanged for all trials. Restarts use existing total_failures;
all selected traces must complete both directions.

Approved launcher: `/home/hendrik/villa/vesuvius/scripts/run_fiber3d_experiment`.
Commands: `embedding-freeze-current`, `embedding-trace-pilot`, then
`embedding-trace-sweep` for history, weight, and refinement configurations.
Shared nonexclusive manager allocation oneGPU20GiB/16CPU; training retained.
All non-embedding trace arguments are preserved by the sweep.

Outputs under `/ephemeral/hendrik/fiber3d_train2/evaluations/`:
- `embedding_trace_paired_tune_20260913_v1`: frozen-checkpoint off/on pilot.
- `embedding_paired_tune_history_20260913_v1`:33 histories +2 controls +repeat.
- `embedding_paired_tune_weight_20260913_v1`:weight descent on selected history +controls/repeat.
- `embedding_paired_tune_refine_20260913_v1`:count and local spacing/window refinements +controls/repeat.

Each report.json retains exact variant settings, replay hash, timing, cache-extension counts,
and full metrics; each variant JSON is an executable evaluation configuration.
Source cache identities/checkpoint hashes were checked; controls and each stage winner repeated exactly.
Final replay hash:
`cc6b696a8632cc1523e638f7a91744331068eee8e8b711f9dc7e68e5e3ea1b57`.

## History grid at weight0.25, count64

Spacing8/window1024 or4096 is count-limited to roughly512 base voxels;
spacing32/window4096 is limited to roughly2048. These are not full-window dense tests.
Spacing128 covers the requested long windows without the count cap truncating them.

| Spacing | Window | Uniform restarts | Linear | Gaussian |
|---:|---:|---:|---:|---:|
|8|64|92|88|90|
|8|256|87|89|91|
|8|1024|89|86|91|
|8|4096|89|86|89|
|32|64|84|89|89|
|32|256|90|93|88|
|32|1024|90|90|84|
|32|4096|92|90|88|
|128|256|95|92|86|
|128|1024|88|87|82|
|128|4096|89|92|91|

## Weight refinement (spacing128/window1024/count64/Gaussian)

| Weight | Restarts |
|---:|---:|
|0.25|82|
|0.2|87|
|0.3|96|

Strict .05 neighbor descent: both neighbors worse than.25, so retained.25.
Ties do not drive further descent. This does not exclude distant optima or interactions outside the tested grid.

## Count and local-history refinement at weight0.25/Gaussian

| Spacing | Window | Count | Restarts |
|---:|---:|---:|---:|
|128|1024|1|90|
|128|1024|2|92|
|128|1024|4|85|
|128|1024|8|82|
|128|1024|16|82|
|128|1024|32|82|
|128|1024|64|82|
|64|1024|64|86|
|256|1024|64|90|
|128|512|64|87|
|128|2048|64|91|

## Validation and timings

112 regression tests passed33.32s via approved cooperative test runner
(`cooperative-tests/20260913_193258_569190`). No native/training numerics changed.
Sweep code generalized explicit embedding-only candidates and selected-history weight search;
legacy search defaults retained, count limited to native1–64.

Pilot off/on85.38s/130.62s, including inference. Cached sweep timings include process setup,
native tracing and occasional manager-authorized cache extension; not pure GPU/model throughput.

| Stage | Replays incl. controls/repeat | Mean seconds | Min / median / max | New tiles |
|---|---:|---:|---|---:|
|history|36|47.27|3.33 / 48.32 / 60.97|210|
|weight|6|31.83|3.31 / 33.93 / 45.62|0|
|refine|14|39.27|4.10 / 42.67 / 47.18|26|

The final repeat needed no new inference. No unseen-fiber validation was run.
Use these embedding flags for a subsequent validation; they are not installed as defaults:

```text
--embedding-weight 0.25 --embedding-history-spacing-base-voxels 128 --embedding-history-window-base-voxels 1024 --embedding-history-count 8 --embedding-history-falloff gaussian
```

