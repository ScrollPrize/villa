# Winding-step search: filesystem-blocked status

Checkpoint before tuning: `dd8c1331e`. Recorded 2026-09-27 23:23 UTC; this is an incomplete experiment, not a successful replacement of the legacy baseline. The filesystem wait subsequently cleared enough for work to resume.

Update23:27UTC: recovery was intermittent; full evaluations and refinement startup are again blocked in filesystem transaction waits. Four nearby cold-cost refinement points are now queued/running in addition to the incomplete runs below. Python26tests pass with explicit villa2pythonpath (installed editable otherwise points at villa).

Hard continuation means equal winding labels are enforced. New candidates instead use finite `lambda * abs(q_b - q_a)`, with physical winding q including the H/V half step. Hard legacy rows below are comparison controls only.

48 completed trial rows on the same frozen Paris4 2048 trace data and 26 references; 500-message budget unchanged. Best new initial accuracy is 26.9231%, versus legacy 38.4615%. Best new shift score is 46.1538%, versus legacy 73.0769%. These maxima need not be the same candidate. No default promotion.

One cold trial and three full evaluations (one queued) remain incomplete. Earlier five native processes were blocked in kernel `wait_transaction_locked`; `/ephemeral` was 99% full and its flush worker was active. This identified a filesystem wait, not its precise root cause. No unrelated files were deleted or data moved to another filesystem. Large-log report regeneration also blocked, then resumed. This table is generated directly from already-persisted summaries, without rerunning or reparsing native results.

## Completed results

Initial accuracy is the independent reference readout of the unconditioned solve. Shift score is `100*(1-K/26)`, where K counts adjacent changes in predicted-minus-reference offset; a global offset is ignored. Both are higher-is-better. These are screening results: reference-assisted/removal scores were not measured for these rows.

| Trial | Initial accuracy | Shift score | Defect pieces |
|---|---:|---:|---:|
| orthogonal-d10-c10-distance-screen | 11.5385% | 15.3846% | 5.4775% |
| orthogonal-d100-c0p1-fixed-distance-screen | 15.3846% | 26.9231% | 0.0000% |
| orthogonal-d100-c1-fixed-distance-screen | 11.5385% | 23.0769% | 0.0000% |
| orthogonal-d100-c10-fixed-distance-screen | 11.5385% | 34.6154% | 0.0382% |
| orthogonal-d100-c100-fixed-distance-screen | 19.2308% | 46.1538% | 4.5252% |
| parallel-d100-c0p1-fixed-distance-screen | 19.2308% | 30.7692% | 4.6103% |
| parallel-d100-c1-fixed-distance-screen | 23.0769% | 38.4615% | 4.6206% |
| parallel-d100-c10-fixed-distance-screen | 19.2308% | 30.7692% | 4.7453% |
| parallel-d100-c100-fixed-distance-screen | 19.2308% | 46.1538% | 8.3343% |
| parallel-d100-chard-fixed-legacy-screen | 38.4615% | 73.0769% | 15.5965% |
| parallel-d1-c0p1-fixed-distance-screen | 19.2308% | 23.0769% | 4.7306% |
| parallel-d1-c1-fixed-distance-screen | 15.3846% | 23.0769% | 7.6212% |
| parallel-d1-c10-fixed-distance-screen | 19.2308% | 30.7692% | 7.8281% |
| parallel-d10-c0p1-fixed-distance-screen | 19.2308% | 30.7692% | 4.6088% |
| parallel-d10-c1-distance-screen | 23.0769% | 7.6923% | 12.2594% |
| parallel-d10-c1-fixed-distance-screen | 19.2308% | 38.4615% | 4.6367% |
| parallel-d10-c10-fixed-distance-screen | 23.0769% | 42.3077% | 8.1406% |
| parallel-d100-c1-distance-screen | 11.5385% | 15.3846% | 3.9427% |
| parallel-d100-c1-fixed-distance-screen-finite-prepass | 23.0769% | 38.4615% | 4.5560% |
| parallel-d100-c1-fixed-distance-screen-mass-ordinary | 19.2308% | 11.5385% | 49.5795% |
| parallel-d100-c1-fixed-distance-screen-mass-prepass-ordinary | 19.2308% | 7.6923% | 47.8812% |
| parallel-d100-c1-fixed-distance-screen-mass-winding-ordinary | 19.2308% | 34.6154% | 13.6577% |
| parallel-d100-c1-fixed-distance-screen-mixed10 | 23.0769% | 38.4615% | 4.5384% |
| parallel-d100-c1-fixed-distance-screen-p0 | 19.2308% | 46.1538% | 7.2445% |
| parallel-d100-c1-fixed-distance-screen-p1x10 | 23.0769% | 30.7692% | 4.4283% |
| parallel-d100-c1-fixed-distance-screen-temp005 | 19.2308% | 26.9231% | 4.6235% |
| parallel-d10-c1-distance-screen-repair5 | 11.5385% | 7.6923% | 6.3065% |
| parallel-d10-c1-fixed-distance-screen-repair5 | 19.2308% | 34.6154% | 4.6367% |
| parallel-d10-c10-distance-screen-repair5 | 23.0769% | 15.3846% | 8.9374% |
| parallel-d10-c10-fixed-distance-screen-repair5 | 23.0769% | 42.3077% | 9.4495% |
| parallel-d100-c1-distance-screen-repair5 | 11.5385% | 15.3846% | 3.9250% |
| parallel-d100-c1-fixed-distance-screen | 23.0769% | 38.4615% | 4.6206% |
| parallel-d100-c1-fixed-distance-screen-repair5 | 23.0769% | 38.4615% | 4.6206% |
| parallel-d100-c10-distance-screen-repair5 | 19.2308% | 15.3846% | 6.0659% |
| parallel-d100-c10-fixed-distance-screen-repair5 | 19.2308% | 26.9231% | 4.6602% |
| parallel-d100-c1-fixed-distance-screen-slack0p1 | 19.2308% | 30.7692% | 4.6103% |
| parallel-d100-c1-fixed-distance-screen-slack1 | 19.2308% | 23.0769% | 4.6088% |
| parallel-d100-c1-fixed-distance-screen-slack10 | 23.0769% | 26.9231% | 5.0681% |
| parallel-d100-c1-fixed-distance-screen-span0p01 | 23.0769% | 38.4615% | 4.6206% |
| parallel-d100-c1-fixed-distance-screen-span0p1 | 19.2308% | 30.7692% | 4.6118% |
| parallel-d100-c1-fixed-distance-screen-span1 | 19.2308% | 23.0769% | 4.6073% |
| parallel-d10-c1000-fixed-distance-screen | 23.0769% | 42.3077% | 8.2199% |
| parallel-d10-c10000-fixed-distance-screen | 23.0769% | 42.3077% | 8.2199% |
| parallel-d100-c1000-fixed-distance-screen | 19.2308% | 46.1538% | 8.3328% |
| parallel-d100-c10000-fixed-distance-screen | 19.2308% | 46.1538% | 8.3328% |
| parallel-d100-c10-fixed-distance-screen-cool0p1-baseD10 | 26.9231% | 42.3077% | 8.3006% |
| parallel-d1000-c1-fixed-distance-screen-cool0p1-baseD100 | 19.2308% | 30.7692% | 4.6704% |
| parallel-d1000-c10-fixed-distance-screen-cool0p01-baseD10 | 26.9231% | 42.3077% | 9.3658% |

## Changes and validation

- Corrected finite-step diagnostics: positive steps are charged, not counted as removed connections; zero-cost edges remain absent.
- Tested optional synthetic-continuation defect-mass exclusion and local normalized-MAP decoder repair. Neither improved accuracy; defaults remain unchanged.
- Decoder repair respects fixed states/gauges, hard signs and existing support; no reference labels enter its decisions. Lower objective did not imply better accuracy.
- Native 122 tests passed; 15 targeted Python tests passed. Earlier broader Python validation is in task_log.
- Coarse/refined costs, joint/fixed H/V, repair, spacing penalties, very strong finite costs and matched-objective cooling were tested. Cooling keeps normal alignment/prepass unchanged by scaling winding temperature and defect cost inversely.

## Reproduction and artifacts

Build: `bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build`.

Run/resume a batch with the approved entrypoint:

```sh
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_step_tuning_full_20260927.json
```

Other batch configs use the same prefix and `screen`, `refine`, `repair`, `compact`, `strong`, `cold`, or `report` in place of `full`. Do not launch duplicate work while the existing blocked runners remain alive. Each artifact root retains commands, native hashes/patches, input/reference hashes, logs, ladders and CSVs. Timings were contended (32–48 threads per process, max208 concurrent requested threads), not standalone throughput measurements.

- `/ephemeral/hendrik/fiber_crop_bench/20260927_step_tuning_screen_v1/summary.json` — batch complete: true.
- `/ephemeral/hendrik/fiber_crop_bench/20260927_step_tuning_refine_v1/summary.json` — batch complete: true.
- `/ephemeral/hendrik/fiber_crop_bench/20260927_step_tuning_repair_v1/summary.json` — batch complete: true.
- `/ephemeral/hendrik/fiber_crop_bench/20260927_step_tuning_compact_v1/summary.json` — batch complete: true.
- `/ephemeral/hendrik/fiber_crop_bench/20260927_step_tuning_strong_v1/summary.json` — batch complete: true.
- `/ephemeral/hendrik/fiber_crop_bench/20260927_step_tuning_cold_v1/summary.json` — batch complete: false.

Pending full root: `/ephemeral/hendrik/fiber_crop_bench/20260927_step_tuning_full_v1`.
