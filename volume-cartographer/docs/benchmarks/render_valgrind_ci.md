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

Every score must remain within the symmetric tolerance stored in
`core/test/data/render_valgrind_ci_reference.json`. This is a regression score,
not estimated native wall time.

## Running Locally

The deterministic gate supports Linux amd64. Use the same compiler and
Valgrind versions recorded by the reference. Configure once:

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
  --target bench_render_synthetic --parallel "$jobs"
ctest --test-dir volume-cartographer/build/ci-render-benchmark \
  --output-on-failure -R '^test_render_synthetic_fixture$'
cmake --build volume-cartographer/build/ci-render-benchmark \
  --target render_valgrind_ci --parallel "$jobs"
```

Set `jobs` to a smaller positive number to limit local CPU or memory use. This
only controls how many independent Ninja commands run at once. It does not
change the fixed four-worker renderer fixture or five-core replay model.

Artifacts are under
`build/ci-render-benchmark/render-valgrind-ci/<fixture>/<scenario>/`:

- `callgrind/artifact.json` and raw per-thread profiles;
- `drd/artifact.json`, `drd.log`, and `events.jsonl` for parallel cases;
- `evaluation.json` for the ungated score;
- `checked.json` after a passing comparison;
- `checked.failed.json` after a failed comparison.

Start failure diagnosis with `checked.failed.json`. A checksum failure is a
rendering behavior change. An identity failure means compiler, Valgrind, model,
cache, fixture, or repetition settings differ. An incomplete DRD failure means
the trace must be recollected. A score ratio outside the reported interval is a
performance change requiring investigation or an intentional reference update.

## CI Activation

The workflow runs on qualifying pull requests and pushes to `main`. The
rendering job is selected when changes touch VC3D/core build inputs such as
`volume-cartographer/core/**`, `volume-cartographer/scripts/**`, CMake files,
VC3D sources, shared utilities/libraries, or `.github/workflows/vc3d-ci.yml`.
Documentation-only changes under `volume-cartographer/docs/**` run the workflow
path filter but do not select this expensive rendering job.

Merging the implementation into `main` therefore makes the job available for
all subsequent qualifying changes; the pull request containing the workflow
change should also execute it. To make passing it mandatory before merge, add
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

A model change changes its SHA-256 identity, so all eight references must then
be refreshed. Never use renderer observations to fit these coefficients.

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
  --tolerance 0.10 \
  --output volume-cartographer/core/test/data/render_valgrind_ci_reference.json \
  "$build"/render-valgrind-ci/*/*/evaluation.json
cmake --build "$build" --target render_valgrind_ci --parallel "$jobs"
```

The freeze command requires exactly all eight cases and records checksum,
environment/workload identity, score, model hash, and tolerance. Review every
old/new score ratio. Run a second fresh collection before accepting references
when parallel DRD replay variation is close to the chosen tolerance.

## Tightening Or Loosening Tolerance

Tolerance is a fraction in `[0, 1)` stored once at the top level of
`render_valgrind_ci_reference.json`. The default `0.10` accepts ratios from
`0.90` through `1.10`:

- lowering it to `0.05` tightens the gate to `[0.95, 1.05]`;
- raising it to `0.15` loosens the gate to `[0.85, 1.15]`.

Both unexpected slowdowns and speedups fail because either can indicate changed
work or a broken benchmark. A tolerance-only change must leave all eight scores,
checksums, identities, and the model hash unchanged. Use the atomic policy-only
command:

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
PYTHONPATH=volume-cartographer/scripts python3 -m unittest \
  volume-cartographer/core/test/test_run_render_valgrind_ci.py
cmake --build volume-cartographer/build/ci-render-benchmark \
  --target render_valgrind_ci --parallel "$(nproc)"
git diff --check
```

Model changes require fresh references. Reference-score changes require fresh
collection. Tolerance-only changes require an otherwise unchanged reference.
