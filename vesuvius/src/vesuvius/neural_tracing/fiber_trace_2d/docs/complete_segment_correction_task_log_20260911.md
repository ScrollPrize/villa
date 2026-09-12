# Complete-segment metric correction

- User confirmed restarts+1 per directed fiber (10mm/one restart =5mm).
- Prior reports excluded successful terminals from headline empirical statistics, and native summaries used F+1 across the whole corpus. Both are superseded by F+C.
- No changes to trace paths, failures, checkpoint, inference, or reset behavior; no expensive reruns.
- The referenced volume-cartographer/scripts/build_dependencies.sh does not exist in this checkout; reuse existing /tmp/vc-live-native-cmake build instead of installing dependencies.

## Implementation and findings

- Independent plan review by complete_segment_native_review: approved; preserve empty-native-summary numeric zeros with0segments (CLI rejects empty input).
- Fixed native per-source and replay-set mean/count to F+C, source schema5/replay-set2 and complete-reference-partition-v1 marker. Source failure/credited diagnostics remain unchanged.
- Added shared replay_metrics.py complete reference partitions and pooled quantiles. Updated existing saved-artifact analysis script; old quantities now explicitly nested under legacy_diagnostics, not headline metrics. Percentiles linearly interpolate at(n−1)q. Zero-failure diagnostic groups handled safely.
- Added correct_saved_replay_metrics.py: read-only default, preflight validation, immutable byte-exact original backups, atomic summary replacement, idempotent repeat. User-approved scoped external write corrected25completed summaries under evaluations; missing/incomplete runs had no complete summary and received no fabricated metrics. Raw trace artifacts untouched.
- Audited all300saved case artifacts for the three comparable50-fiber/100-direction runs against saved failure and credited-length accounting. No inference/tracing/training rerun. Original and current hashes recorded in docs/benchmarks/complete_segments_migration_20260911.json.
- Corrected global means: classic3.857545613mm(F1995,C100); previous collection17k2.707389635mm(F2885,C100); latest collection3k3.601407335mm(F2144,C100). Paris4 now classic4.295342807mm(F1226,C60), latest4.223096980mm(F1248,C60), resolving the contradiction.
- New full mean/median/P05/P95/count tables in docs/benchmarks/complete_segments_20260911.md/.json. Updated the3original JSON reports; retained historical narrative tables clearly labelled superseded and linked corrected current results. Performance/snapshot provenance unchanged.

## Validation commands

```bash
cmake --build /tmp/vc-live-native-cmake --target test_fiber_reference_replay_benchmark vc_fiber_trace_metric -j 8
/tmp/vc-live-native-cmake/bin/test_fiber_reference_replay_benchmark
/home/hendrik/.venv_las/bin/python -m pytest --noconftest -o 'pythonpath=/home/hendrik/villa2/vesuvius/src /home/hendrik/villa2' -q vesuvius/tests/neural_tracing/test_replay_metrics.py
AGENTS_AGENT_MODE=1 VC_REPLAY_TEST_BINARY=/tmp/vc-live-native-cmake/bin/vc_fiber_trace_metric VC_CACHE_ROOT_BINARY=/tmp/vc-live-native-cmake/bin/vc_cache_root PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=/home/hendrik/villa2/vesuvius/src:/home/hendrik/villa2 /home/hendrik/.venv_las/bin/python -m pytest --noconftest -q vesuvius/tests/neural_tracing/test_native_replay_evaluation.py
PYTHONPATH=vesuvius/src /home/hendrik/.venv_las/bin/python vesuvius/scripts/correct_saved_replay_metrics.py /ephemeral/hendrik/fiber3d_train2/evaluations --apply
```

- GCC13 RelWithDebInfo build and11native tests passed;9metric/migration tests passed;16native CLI tests passed in9.28s. First sandbox CLI-test attempt stalled in Zarr async fixture setup, stopped and repeated outside sandbox successfully. No install required.
- Ran existing trace_restart_analysis_20260911.py on each classic/previous/latest full folder;100artifacts validated per run. Migration repeat reports all25already-correct.

## Explicit scope/limitations

- This metric is full **reference** distance between restart events, including forced reset advance, not a claim of successful traced physical length. End/repeated failures can produce zero segments; final partition included even when no physical terminal attempt exists.
- Old separately scoped crop/reference-replay progress plots retain legacy metrics and are not used in this comparison. Historical narrative tables are preserved as superseded records, not silently reinterpreted. Incomplete runs are not assigned completed metrics.
- No changes to reset policy, GT-heading behavior, inference, thresholds, or traced geometry. No new performance measurements needed for reporting-only correction. arm64/macOS not tested; changes use portable C++/Python standard-library math/filesystem primitives.
