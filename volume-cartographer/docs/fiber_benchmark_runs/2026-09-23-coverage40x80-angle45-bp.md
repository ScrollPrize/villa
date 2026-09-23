# BP removal score: 40/80,45° crop traces

Completed 2026-09-23, 2048³ Paris4, new inference, beam8,1484 source traces. Same reference-conditioned BP settings as corrected original2048 baseline: hard perpendicular signs,parallel-one magnitude0.1,oracle-inliers,one round,128threads. This is not the independent-reference diagnostic.

|Metric|Original2048 (20/80,25°)|New2048 (40/80,45°)|
|---|---:|---:|
|Source traces|3697|1484|
|Initial constraints|1400113|163183|
|Removed incident|625753|59560|
|Retained infringed|323344|43105|
|Retained defect|0|0|
|Problematic|949097|102665|
|Retained fulfilled|451016|60518|
|Benchmark score|-210.4353%|-169.6437%|
|Removed incident / initial|44.6930%|36.4989%|
|Problematic / initial|67.7872%|62.9140%|
|Conditioned reference exact|21/26|26/26|
|Wall time|1492.30s|339.22s|

Score = −100 × problematic / retained fulfilled; closer to0 is better. Problematic includes removed incident plus retained infringed/defect constraints, not just removals. No BP result for intermediate40/80,25° run: comparison changes both axes and angle. Graphs/counts differ, so this is end-to-end comparison. Conditioned reference correctness is not unassisted BP accuracy.

Returncode0;terminal zero_required_errors. Initial/final stages reached500message cap;conditioned reports message_limit(95.06s),accepted by existing protocol. Heuristic pruning,not minimum-removal guarantee.

## Reproduce

```sh
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_bp2048_coverage40x80_angle45_20260923.json
```

Output prefix: `/ephemeral/hendrik/fiber_crop_bench/20260923_beam2048_v1/bp-beam-new2048-coverage40x80-angle45` (.log,.run.json,CSVs,OBJs). RelWithDebInfo from previous83-test validated build. Config-only change;independent review confirmed unchanged BP settings. No inference or retracing.
