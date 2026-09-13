# EMA99 checkpoint: tracer-reference comparison

## Fixed workload

Current model EMA0.99, update6500 /832000 patches, frozen SHA256
`0a2651b711e7e5ee6d445ac8c68ff482c1951798ced2f44896c25602e1c50417`.
Paired combined-five-scroll training continues during testing.
Whole held-out PHerc1299 fiber `lt_20260825T075316065_000069.json` (dataset alias
PHerc1229), both directions. Same geometry, normal costs and failure ellipse.
Inference tile256/border32/overlap96, BF16 forward, FP32 signed embeddings.
Native GCC QuickBuild O3, no compiler precision changes. One nonexclusive
manager GPU20GiB plus16 logical CPUs; timings include concurrent training.

## Pre-change controls

| Reference | Forward | Reverse | Total restarts |
|---|---:|---:|---:|
| Off |41|46|87|
| Original mean, .25 /32 /256 /8 /uniform |35|47|82|
| Previous raw winner, .25 /128 /1024 /8 /Gaussian |35|52|87|

The raw4300 winner does not transfer as winner to EMA99 at6500. This is not a
controlled raw-versus-EMA comparison because training steps also differ.
Pre-change off/original cached replays exactly match their pilot hashes;
original repeats exactly. Fresh pilot walls80.28s/126.86s include inference.
Previous-best control46.83s includes72 newly inferred tiles.

## Experiments

Reference EMA is distinct from the EMA model weights. It is updated per selected
beam child, scored against parent first, using normalized interpolation with
decay `exp2(-step_distance/half_life)`. Half-lives128/512/2048 base voxels.
Modes: plain, gated, seed-anchored, gated+seed. Seed cost averaged equally with
EMA cost. Gate requires presence>=.5 and cosine>=.5; failed gate only prevents
update. Cancellation retains parent. Every restart resets references.
Best-history uses maximum cosine, not mean, over retained history with no age
discount. Windows256/1024/4096, spacing128, count64. All start at weight.25.
Keep both original and previous-winner controls. Refine winning weight by.05
neighbors until neither strictly improves; ties retain current weight.

Only embedding settings change. Inference cache is reused with manager-gated
extension when a new route visits missing coverage. Repeat winner exactly.
One difficult fiber is an in-sample tuning workload, not validation across
scrolls. Defaults are not changed by this experiment.

## Reproduction

Settings live in `vesuvius/scripts/fiber3d_experiment.json`; saved result JSONs
preserve the actual settings for each stage. Commands:

```bash
/home/hendrik/villa/vesuvius/scripts/run_fiber3d_experiment embedding-freeze-current
/home/hendrik/villa/vesuvius/scripts/run_fiber3d_experiment embedding-trace-pilot
/home/hendrik/villa/vesuvius/scripts/run_fiber3d_cooperative_tests
/home/hendrik/villa/vesuvius/scripts/run_fiber3d_experiment embedding-trace-sweep
```

Frozen snapshot and pilot: `/ephemeral/hendrik/fiber3d_train2/evaluations/embedding_ema99_reference_20260913_v1` (frozen copy under sibling `-frozen`).
Pre-change controls: `embedding_ema99_controls_20260913_v1/report.json` under same root.
Modes: `embedding_ema99_modes_20260913_v1/report.json`.
Each result preserves per-direction traces, summary, exact replay hash, and
inference coverage. New reruns require a fresh output directory.

## Completed mode sweep

| Mode | Half-life / window (base voxels) | Restarts | Wall seconds | Added tiles |
|---|---:|---:|---:|---:|
| Off | — | 87 | 3.29 | 0 |
| Original mean | 256 | 82 | 31.75 | 0 |
| Previous-best mean | 1024 | 87 | 35.56 | 0 |
| ema | 128 | 81 | 47.40 | 12 |
| ema | 512 | **69** | 50.00 | 45 |
| ema | 2048 | 82 | 58.28 | 35 |
| ema-gated | 128 | 84 | 53.78 | 12 |
| ema-gated | 512 | 75 | 38.73 | 0 |
| ema-gated | 2048 | 75 | 51.54 | 8 |
| ema-seed | 128 | 78 | 52.21 | 44 |
| ema-seed | 512 | 85 | 37.34 | 0 |
| ema-seed | 2048 | 83 | 47.38 | 0 |
| ema-seed-gated | 128 | 77 | 49.80 | 26 |
| ema-seed-gated | 512 | 82 | 37.50 | 0 |
| ema-seed-gated | 2048 | 79 | 43.81 | 0 |
| best | 256 | 76 | 41.72 | 0 |
| best | 1024 | 79 | 51.38 | 18 |
| best | 4096 | 77 | 43.76 | 0 |
| Repeat EMA | 512 | **69** | 38.16 | 0 |

19 bidirectional replays; all completed. Walls sum 813.38s,
mean 42.81s, min/median/max 3.29/43.81/58.28s.
Times include cached tracing and any model startup/cache extension, not isolated
native kernel timings. Total added tiles 200.
Winner EMA512 weight.25:28 forward+41 reverse=69,20.7% fewer than off87,
15.9% fewer than original82. Exact winning-repeat hash:
`6e590851a9a00555dbc14edc3aa6f1fdccfc57a47b4a8c3c85a22ba900050484`.

## Weight refinement and validation

EMA512 weights.20/.25/.30 give80/69/84 restarts. Both neighbors worsen,
so.25 is the local minimum on the.05 grid, not a global optimum claim.
Winning final repeat again has the exact hash above; all controls reproduce.
Weight-stage report: `embedding_ema99_weight_20260913_v1/report.json`.
Six replays total190.69s, mean31.78s, min/median/max3.30/35.97/42.93s;
8 additional tiles. Winner cached-only repeat37.44s.
Combined19+6 mode/weight replays all complete; no variants skipped.
History spacing/window/count/falloff do not control EMA modes.

Native build/tests through approved runner: both native suites pass1.63s,
113 Python regressions pass31.96s; logs
`/ephemeral/hendrik/fiber3d_train2/cooperative-tests/20260913_203814_067611`.
Tests cover default behavior, beam-local memory, EMA formula/gates/cancellation,
invalid mode/options, sweep parameter preservation and manager admission.
Independent review checked implementation and report. `git diff --check` clean.

Use with existing trace arguments:

```text
--embedding-weight 0.25 --embedding-reference-mode ema --embedding-ema-half-life-base-voxels 512
```

No inference or tracer defaults were changed. Portable standard C++ math;
measured only on this Linux GCC host. No global/unseen-fiber quality guarantee.

## Next experiment

User requested a bounded fiber-appearance model of multiple representative,
mutually distinct anchors instead of one fixed seed. Favor coverage of observed
embeddings while preserving separate appearance modes, not just outliers.
Recorded for a separate experiment after this completed chain; not implemented
or mixed into these results.
