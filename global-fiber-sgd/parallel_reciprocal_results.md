# Parallel reciprocal removal and perpendicular iteration comparison

Full Paris4_v1 AFV, 512,698 fibers; unsigned local events, whole fibers retained.
Only parallel reciprocal filtering and perpendicular projection iteration count
change. The seed-fixed projection windows, radius, support checks, normal data
and winding checks stay unchanged.

Both parallel reciprocal checks (central and probe) are now disabled by default.
The explicit `parallel_reciprocal=true` option reproduces historical results.
`perpendicular_iterations` defaults to 4; the experiment compares 4, 16 and 64.

## Reproduction and validation

```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_parallel_reciprocal.json
/home/hendrik/.venv_las/bin/python global-fiber-sgd/summarize_reciprocal_sweep.py
```

RelWithDebInfo, GCC13, 128 extraction threads, one full-volume run per setting.
These are single-run timings, not statistical speedup estimates. Raw outputs,
logs, command/input/binary hashes and resource usage are in
`data/parallel_reciprocal_v1/`. The compact aggregate is
`reciprocal_sweep_results.json`.

Native regression suite: 139 passed. Legacy-on/4 real-data pilot matches all
19,889 saved serialized measurements exactly. The summarizer checks the complete
attempt/rejection/deduplication ledger and absence of parallel probe rejections.
No inference, normal alignment, winding solve or database overwrite was run.

## Interpretation

| Parallel reciprocal | Perpendicular iterations | Accepted parallel | Accepted perpendicular | Accepted total |
|---|---:|---:|---:|---:|
| On (historical) | 4 | 1,180,410 | 1,153,778 | 2,334,188 |
| Off | 4 | 3,061,068 | 1,153,778 | 4,214,846 |
| Off | 16 | 3,061,053 | 1,153,736 | 4,214,789 |
| Off | 64 | 3,061,040 | 1,153,736 | 4,214,776 |

| Iterations | Perpendicular attempts reaching reciprocal check | Reciprocal failures | % of checked | % of all attempts | Failures at seed-window boundary | Extraction seconds | Native wall seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 11,860,359 | 2,670,276 | 22.514293% | 8.212683% | 2,639,404 | 271.08 | 301.60 |
| 16 | 11,860,436 | 2,670,297 | 22.514324% | 8.212709% | 2,639,438 | 247.74 | 278.91 |
| 64 | 11,860,446 | 2,670,299 | 22.514322% | 8.212711% | 2,639,440 | 276.77 | 306.97 |

All-attempt denominators are 32,514,051 / 32,514,205 / 32,514,221.
Runner-inclusive wall times were 303.78 / 281.36 / 309.59 seconds.
Physical-endpoint reciprocal failures were zero in all runs; earlier tangent
support gates already exclude endpoint-limited encounters.

Disabling the parallel checks adds 1,880,658 parallel events at four iterations
and leaves the accepted perpendicular count exactly unchanged. Total accepted
events increase 80.57%. More iterations do not help: 16 and 64 accept 42 fewer
perpendicular events than four (0.00364%). Keep the default at four.

At four iterations, 98.8439% of reciprocal failures touch a seed-window boundary;
only 30,872 do not. At 16 and 64, the non-boundary counts are both 30,859.
Small changes in parallel counts despite only changing perpendicular iterations
can occur through adaptive refinement of the shared candidate schedule.

Perpendicular projection alternates B and A within fixed windows centered on
the original seeds. The reciprocal check projects the same B onto A again, but
within a window centered on the **refined** A point. Thus this is not a pure
convergence test: its search domain changes. More iterations cannot in general
resolve that mismatch. This experiment deliberately leaves the windows intact.

Counters are candidate attempts before deduplication, including refinement
attempts. A failed attempt is not necessarily a lost unique constraint. Adaptive
refinement means candidate totals can change with iteration count. Boundary
coincidence is a diagnostic association, not proof of the cause of each failure.
More accepted events alone do not demonstrate anatomical correctness.
