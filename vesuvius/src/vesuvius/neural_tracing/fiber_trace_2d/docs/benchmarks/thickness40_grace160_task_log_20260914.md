# Eligible JSON EMA99 evaluation and visualization

Mapping fix validation passed19native+136Python tests (49.54s Python).
Restart submitted through same approved full runner with
configs/eval_eligible_json_ema99_reference_vis_20260914_v2.json.

V1 failed publication: _read_mmap_band_chunk np.memmap Errno12 despite1.8TiB
available RAM. Each128flush thread opened fresh whole-volume sparse mappings.
Fix _publish seeds shared helper cache with existing store-owned file-backed
arrays; publication never closes borrowed maps. Independent lifecycle review
approved (writers complete before publication; publisher pool joins before store
close; helper copies output region). Regression tests8concurrent publications,
flat/chunk-major, forbid np.memmap construction and verify output/map lifetime.
Initial test fixture used unnormalized42, then42/255 rounded down byfloat16;
changed to exactly representable0.5 with the adapter's expected byte127.
Retryv2 preserves snapshot andv1inference namespace/published chunks; no cache
identity or numerical change. V1 failed artifacts retained.

Native reference audit matches Classic:31/3/5/2/41/8 eligible runs in the six
collections (90total,180directions). Managed full run started task
e8d7bd13-4eae-4000-a297-196573d90511,all8GPUs/64logicalCPU,priority100.
Output /ephemeral/hendrik/fiber3d_train2/evaluations/eligible_json_ema99_reference_vis_20260914_v1.
Model inference cache key c37c4fbf8a42c0412930c1d0d702a20572c1438d7ba0baae52de16193aa94c5c.
Command: /home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_eligible_json_ema99_reference_vis_20260914_v1.json

Frozen source current_ema_0p99.pt: SHA256
c348f867a6e922c0f7e19928c7982b9325475fd63505e42de325065e3e15a2e8,
update1800,samples230400,test_ema_0p99/loss_total0.3443518824.
Model output_stride4,one7-channel branch,embedding disabled; existing adapter
and demand publication account for native stride. No manual scale conversion.
Approved runner gained freeze-snapshot mode calling existing shared freezer,
snapshot.json records config and checkpoint provenance. Prepared full50 bindings.
Independent plan review approved.19 native+77 Python tests passed15.16s.
Latest explicit prior user instruction authorizes allGPUs/highpriority; reuse
same exclusive managed allocation as completed Classic visualization.

Use current EMA0.99 from5scroll_json_eligible_128_sd1_16x8_ema_20260914_105436; freeze via shared SHA helper. Full50 fibers/180 eligible directed spans, Classic tracing embedding off,80-voxel restart episodes. All8GPUs managed, new model namespace. Prior log archived in docs/benchmarks/classic_vis_restart80_task_log_20260914.md.
## Workspace build follow-up

Completed both full runs,180/180 each. Classic loose292 restarts/14.550058mm
mean (strict510/9.953083); EMA99u1800 loose174/19.400077 (strict334/13.361143).
Full per-scroll mean/median/P05/P95/restarts and zero-length counts generated
from saved summaries using shared complete_report and persisted in
docs/benchmarks/thickness40_grace160_20260914.md. Coverage checked identical:
6867.627252833mm directed. Classic wall84.0103s; EMA99 wall81.5072s, single
cached-inference runs with on-demand fills, not cold throughput benchmarks.
Final approved regression runner:21native+136Python passed (42.25s Python).
All requested evaluation work complete; no training altered. New-policy
rendering remains explicitly unsupported (not requested for these runs).

Classic loose-policy full run completed180/180:292 episode restarts, mean
14.550057739mm, median5.822665675mm, P05.176020696mm,P9562.408360311mm.
Strict Classic510,mean9.953082975mm.475 newly inferred tiles/55 published chunks;
all existing cache namespaces reused. EMA99 managed run started next.
Additional integration regression passed (21native total): unresolved terminal
excursion fails even below grace; delayed failure occurs later than immediate;
factor2 geometry survives JSON roundtrip. Only test executable rebuilt during
evaluation; tracer/library binaries unchanged.
`report-metrics` rejected loose workload with missing cached_evaluation key
(it is a legacy saved-metric conversion mode); no files modified by that failed
report call. Read completed summary using shared complete_report instead.
Known limitation: nondefault ellipse rendering is explicitly rejected; these
requested comparison runs disable rendering. Existing strict galleries retained.

Validation: workspace build completed; unchanged repeat reports `ninja: no work
to do`. Approved no-argument replay test runner passed20 native and136 Python
tests (50.79s Python). Build dependency preflight added to prevent silently
falling back to installation when local artifacts are absent. Independent
review approved. Config filenames remain v1; their output roots are v2 to
preserve the original prepared audit. Both full comparisons submitted using
the existing run_classic_fiber_evaluation.py full --config command, with the
two eval_*thickness40_grace160_20260914_v1.json configs. No rendering.

User requested reusable builds inside the workspace. Added no-argument
run_fiber3d_native_build, targeting volume-cartographer/build/fiber-evaluation.
The machine-local GCC13/RelWithDebInfo runner reuses existing delta3d source and
libbacktrace installation from villa read-only; no dependencies installed.
Initial libbacktrace discovery missed absent pkg-config metadata and CMake
attempted a download (failed DNS); added local metadata for the existing library,
then configuration confirmed `Using system libbacktrace`. Tests/new evaluations
now reference workspace binaries. Prepared-only loose v1 outputs preserved;
v2 outputs will hold the changed executable-path audit. Historical run configs
are intentionally unchanged.
