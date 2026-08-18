# VC3D Rendering Performance Gate

## Purpose

The `render-benchmark` job in `.github/workflows/vc3d-ci.yml` protects VC3D's
production `ChunkedPlaneSampler` fine-to-coarse rendering path. It builds a
generic Release binary, validates native multi-thread execution, then asks
Ninja to generate a fresh eight-case Valgrind graph:

- serial and four-worker parallel fixtures;
- `full_res`, `fallback_3`, `mixed_correlated`, and `mixed_shuffled`;
- separate-thread Callgrind profiles for all eight cases;
- complete DRD dependency graphs for the four parallel cases;
- a relative modeled-runtime score and exact rendering checksum per case.

The fixture renders through the production `ChunkCache`. A deterministic fake
`IChunkFetcher` preloads all resident and missing states before measurement;
the benchmark fails if a timed render reaches the fetcher. Storage, decode, and
download work are therefore excluded while production cache lookup, locking,
and request-context handling remain measured.

Every score must stay at or below the reference plus the one-sided slowdown
tolerance stored in `core/test/data/render_valgrind_ci_reference.json`. Faster
scores pass. This is a regression score, not estimated native wall time.

## Running Locally

The deterministic gate supports Linux amd64. The CI configuration below is the
normal local setup, but compiler and Valgrind versions are diagnostic metadata,
not historical reference gates. Configure once:

```bash
cmake -S volume-cartographer -B volume-cartographer/build/ci-render-benchmark \
  -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-15 -DCMAKE_CXX_COMPILER=g++-15 \
  -DVC_MARCH_NATIVE=OFF -DVC_TESTING=ON \
  -DVC_RUN_RENDER_BENCHMARKS=ON \
  -DVC_BUILD_APPS=OFF -DVC_BUILD_UI_TRACER=OFF \
  -DVC_BUILD_FLATBOI=OFF
```

Run the native correctness check and complete gate using all available logical
CPUs:

```bash
jobs=$(nproc)
cmake --build volume-cartographer/build/ci-render-benchmark \
  --target bench_render_synthetic bench_thread_sync_replay --parallel "$jobs"
ctest --test-dir volume-cartographer/build/ci-render-benchmark \
  --output-on-failure \
  -R '^(test_render_synthetic_fixture|test_thread_sync_replay_native)$'
cmake --build volume-cartographer/build/ci-render-benchmark \
  --target render_valgrind_ci --parallel "$jobs"
```

Set `jobs` to a smaller positive number to limit local CPU or memory use. This
only controls how many independent Ninja commands run at once. It does not
change the fixed four-worker renderer fixture or five-core replay model.

The regular estimate contains no Python process. Ninja invokes Valgrind
directly, then `bench_thread_sync_replay evaluate-render` parses the raw
Callgrind profiles and DRD log, validates the pair, attributes costs, replays
the graph, and writes the result in C++.

Artifacts are under
`build/ci-render-benchmark/render-valgrind-ci/<fixture>/<scenario>/`:

- `callgrind/callgrind.out.*`, benchmark metadata, and a collection stamp;
- `drd/drd.log`, benchmark metadata, and a collection stamp for parallel cases;
- `evaluation.json` for the ungated score;
- `checked.json` after a passing comparison;

Start failure diagnosis with the raw profiles, DRD log, and `evaluation.json`.
Historical compiler, model, checksum, cache, fixture, repetition, and profiler
changes do not fail the reference gate. Current-run parse errors or
Callgrind/DRD metadata inconsistency still fail because they prevent a valid
score. A score above the allowed slowdown is a performance regression requiring
investigation or an intentional reference update.

Parallel collection uses the same fair scheduler and 10,000-basic-block
quantum in both Valgrind runs. Periodic Callgrind deltas are matched to the DRD
measured window by canonical worker work rank. Equal DRD worker signatures are
accepted only when their total costs differ by at most 5% and their normalized
32-bin chronological shapes differ by at most 2%. Every admissible mapping is
replayed and the maximum makespan is the score.

## CI Activation

The workflow job runs on qualifying pull requests and pushes to `main`. The
rendering job is selected when changes touch VC3D/core build inputs such as
`volume-cartographer/core/**`, `volume-cartographer/scripts/**`, CMake files,
VC3D sources, shared utilities/libraries, or `.github/workflows/vc3d-ci.yml`.
Documentation-only changes under `volume-cartographer/docs/**` run the workflow
path filter but do not select this expensive rendering job.

To make the job mandatory before merge, add
the `Synthetic rendering regression (GCC Release / Valgrind replay)` check to
the repository branch ruleset or branch protection. Merging alone runs the
check but does not make it a required status check.

GitHub Actions derives Ninja concurrency with `nproc`. The artifact graph has
at most twelve simultaneous collection commands, so runners with more CPUs are
naturally bounded by available graph work.

## Maintenance Rules

Model calibration, case references, and tolerance are independent controls:

- **Recalibrate** only when changing how Callgrind/DRD events map to the score.
- **Refresh references** when an understood code or toolchain change shifts the
  expected score while the model remains valid.
- **Change tolerance** when intentionally changing regression sensitivity.

Do not recalibrate or widen tolerance merely to make an unexplained regression
pass. Each operation requires a focused diff and its reason in the changelog or
pull-request description.

## Recalibrating The Model

Recalibration includes native timing, unlike the CI gate. It must run on the
documented one-CCD calibration host with CPU frequency pinned and at least five
sequential native processes per case. Build
`bench_thread_pool_dispatch` and `bench_thread_sync_replay` in Release first.

```bash
sudo volume-cartographer/scripts/run_with_fixed_cpu_frequency.py set
trap 'sudo volume-cartographer/scripts/run_with_fixed_cpu_frequency.py restore' EXIT
jobs=$(nproc)
cmake --build volume-cartographer/build-release \
  --target bench_thread_pool_dispatch bench_thread_sync_replay \
  --parallel "$jobs"
```

Collect the base synthetic event observations, then the expanded access-density
fit and untouched holdout:

```bash
python3 volume-cartographer/scripts/calibrate_synthetic_event_costs.py \
  --benchmark volume-cartographer/build-release/bin/bench_thread_pool_dispatch \
  --output-dir /tmp/render-event-base --native-trials 5
python3 volume-cartographer/scripts/calibrate_synthetic_event_features.py \
  --phase fit \
  --benchmark volume-cartographer/build-release/bin/bench_thread_pool_dispatch \
  --base-observations /tmp/render-event-base/observations.json \
  --output-dir /tmp/render-event-features --native-trials 5
python3 volume-cartographer/scripts/calibrate_synthetic_event_features.py \
  --phase holdout \
  --benchmark volume-cartographer/build-release/bin/bench_thread_pool_dispatch \
  --base-observations /tmp/render-event-base/observations.json \
  --output-dir /tmp/render-event-features --native-trials 5
```

The normal path continues only if the fit and holdout gates accept
`model_data_reads.json`. Use that event model for synthetic synchronization
calibration:

```bash
python3 volume-cartographer/scripts/calibrate_thread_sync_synthetic.py \
  --runner volume-cartographer/scripts/run_thread_sync_replay.py \
  --benchmark volume-cartographer/build-release/bin/bench_thread_pool_dispatch \
  --replay-engine volume-cartographer/build-release/bin/bench_thread_sync_replay \
  --event-cost-model /tmp/render-event-features/model_data_reads.json \
  --output-dir /tmp/render-thread-sync --trace-trials 3 --native-trials 5
```

Review all fit, holdout, frequency, coefficient-bound, rank/correlation, and
candidate-acceptance diagnostics before producing the compact CI model:

```bash
python3 volume-cartographer/scripts/run_render_valgrind_ci.py freeze-model \
  --calibration /tmp/render-thread-sync/model.json \
  --model-id synthetic-data-reads-v2 \
  --output volume-cartographer/core/test/data/render_valgrind_ci_model.json
```

`freeze-model` requires synthetic-only provenance and the exact seven-feature
data-read basis. It rejects an unaccepted calibration. The exceptional
`--allow-unpromoted` option exists only for an explicitly reviewed experimental
promotion; using it keeps `timing_claims_enabled=false` and must be justified in
the pull request.

Always restore the saved CPU policy, including after a failed calibration:

```bash
sudo volume-cartographer/scripts/run_with_fixed_cpu_frequency.py restore
trap - EXIT
```

A model change is recorded but does not fail by identity. Refresh all eight
references only when an understood model change intentionally shifts the score
baseline. Never use renderer observations to fit these coefficients.

## Refreshing Performance References

Use this when the model remains unchanged but an understood implementation,
compiler, or Valgrind change alters expected scores. Use a new build directory
to guarantee fresh Callgrind and DRD collection, and match the intended CI
toolchain exactly:

```bash
build=volume-cartographer/build/render-reference-$(date +%Y%m%d-%H%M%S)
jobs=$(nproc)
cmake -S volume-cartographer -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-15 -DCMAKE_CXX_COMPILER=g++-15 \
  -DVC_MARCH_NATIVE=OFF -DVC_TESTING=ON \
  -DVC_RUN_RENDER_BENCHMARKS=ON \
  -DVC_BUILD_APPS=OFF -DVC_BUILD_UI_TRACER=OFF \
  -DVC_BUILD_FLATBOI=OFF
cmake --build "$build" --target render_valgrind_ci_measure --parallel "$jobs"
python3 volume-cartographer/scripts/run_render_valgrind_ci.py freeze-reference \
  --model volume-cartographer/core/test/data/render_valgrind_ci_model.json \
  --tolerance 0.05 \
  --output volume-cartographer/core/test/data/render_valgrind_ci_reference.json \
  "$build"/render-valgrind-ci/*/*/evaluation.json
cmake --build "$build" --target render_valgrind_ci --parallel "$jobs"
```

The freeze command requires exactly all eight native evaluation artifacts and
records score, checksum, model hash, and tolerance. Legacy evaluation artifacts
also retain their environment/workload identity as diagnostic metadata. Only
score and tolerance affect historical pass/fail.
Review every old/new score ratio. Run a second fresh collection before accepting
references when parallel DRD replay variation is close to the chosen tolerance.

## Tightening Or Loosening Tolerance

Tolerance is a fraction in `[0, 1)` stored once at the top level of
`render_valgrind_ci_reference.json`. The current `0.05` accepts any finite,
positive ratio at or below `1.05`:

- lowering it to `0.03` tightens the maximum accepted ratio to `1.03`;
- raising it to `0.15` loosens the maximum accepted ratio to `1.15`.

Speedups do not fail the reference gate. A tolerance-only change must leave all
eight scores and diagnostic fields unchanged. Use the atomic policy-only command:

```bash
python3 volume-cartographer/scripts/run_render_valgrind_ci.py set-tolerance \
  --reference volume-cartographer/core/test/data/render_valgrind_ci_reference.json \
  --tolerance 0.05
git diff -- volume-cartographer/core/test/data/render_valgrind_ci_reference.json
cmake --build volume-cartographer/build/ci-render-benchmark \
  --target render_valgrind_ci --parallel "$(nproc)"
```

`set-tolerance` validates the range and preserves every other reference field.
Do not recalibrate the model, recollect profiles, or run `freeze-reference`
solely to change policy width.

## Required Validation

For any maintenance change, run:

```bash
cmake --build volume-cartographer/build/ci-render-benchmark \
  --target test_thread_sync_replay_native --parallel "$(nproc)"
volume-cartographer/build/ci-render-benchmark/bin/test_thread_sync_replay_native
cmake --build volume-cartographer/build/ci-render-benchmark \
  --target render_valgrind_ci --parallel "$(nproc)"
git diff --check
```

Reference-score changes require fresh collection. Model changes require fresh
references only when their intended score shift should become the new baseline.
Tolerance-only changes require an otherwise unchanged reference.
