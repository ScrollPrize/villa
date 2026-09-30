# Independent reference windings:40/80,45° traces

Completed2026-09-23 on2048³ new beam8 traces. Initial reference-free BP only; no reference-driven pruning, conditioned solve, or fresh pruned solve. Per-reference assignments use configured weights/hard signs against frozen initial states. Existing defect states still permitted; incident inactive evidence excluded. GT calibrates only global sign/gauge for comparison (not a held-out gauge). Half-winding grid, no parity filter.

|Metric|Original2048 traces|40/80,45° traces|
|---|---:|---:|
|Correct top-ranked reference winding|8/26 (30.77%)|7/26 (26.92%)|
|Top candidate with no hard-sign violations|4/26 (15.38%)|11/26 (42.31%)|
|Correct and no hard-sign violations|2/26 (7.69%)|4/26 (15.38%)|

All26 have assignments. This is distinct from26/26 in reference-conditioned removal run. Correct references:3,3.5,4,4.5,5,5.5,6. Lower removal score does not imply improved unassisted winding reconstruction.

## Full ladder

|GT|Induced|Correct|Hard-sign errors|
|---:|---:|---|---:|
|0|1|no|19|
|0.5|1.5|no|0|
|1|2|no|1|
|1.5|2.5|no|2|
|2|3|no|0|
|2.5|3.5|no|16|
|3|3|yes|24|
|3.5|3.5|yes|0|
|4|4|yes|0|
|4.5|4.5|yes|0|
|5|5|yes|5|
|5.5|5.5|yes|0|
|6|6|yes|21|
|6.5|5.5|no|15|
|7|6|no|0|
|7.5|6.5|no|30|
|8|7|no|0|
|8.5|7.5|no|18|
|9|7|no|31|
|9.5|7.5|no|0|
|10|8|no|3|
|10.5|7.5|no|12|
|11|9|no|0|
|11.5|9.5|no|0|
|12|10|no|30|
|12.5|9.5|no|4|

## Reproduce/validation

```sh
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_independent_reference_2048_coverage40x80_angle45_20260923.json
```

Output prefix:`/ephemeral/hendrik/fiber_crop_bench/20260923_beam2048_v1/bp-beam-new2048-coverage40x80-angle45-independent`. Log includes top5 candidates and costs. Completed return0,103.62s. Initial BP500message cap,not converged. All6280 initial piece validity/winding/class states identical to initial states in preceding removal run. Config reviewed independently;no native changes/build for this follow-up. Prior tests cover independent ranker/CLI. Geometric coverage is not used to infer these assignments; existing reference-to-generated constraints are used.
