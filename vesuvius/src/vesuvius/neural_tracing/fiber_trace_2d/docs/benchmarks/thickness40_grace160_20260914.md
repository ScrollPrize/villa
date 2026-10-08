# Thickness-only relaxation and return window — 2026-09-14

Historical mean definition: superseded by [combined distance / (restarts+1)](combined_restart_mean_20260914.md). Original table retained as evidence; restart counts and percentiles remain valid.

Both models completed all 50 fibers, 90 CP-bounded/non-Lasagna eligible runs, 180 directions. Embedding disabled. Same snapshots, annotations, inference namespaces, and 80-base-voxel restart-episode counting as strict baselines.

Strict: normal/thickness radius20, tangent/width radius80; immediate failure. Loose: normal40, tangent80 (factor2), 160 base voxels traced travel after first outside committed sample to return inside. Reentry clears excursion; unresolved outside at termination/reference end counts as failure. Width was **not doubled**. Distances below are mm; mean includes terminal segments and equals directed length/(reported restarts + directed runs). Pooled global directed length6867.627252833mm. This changes the evaluation criterion, not model quality.

## All

| Model / policy | Restarts | Mean mm | Median mm | P05 mm | P95 mm | Zero-length segments |
|---|---:|---:|---:|---:|---:|---:|
| Classic strict | 510 | 9.953083 | 3.454257 | 0.179300 | 42.754248 | 1 |
| Classic loose | 292 | 14.550058 | 5.822666 | 0.176021 | 62.408360 | 4 |
| EMA99u1800 strict | 334 | 13.361143 | 4.237513 | 0.208517 | 56.360545 | 0 |
| EMA99u1800 loose | 174 | 19.400077 | 7.625640 | 0.197488 | 87.637392 | 4 |

## PHerc0139

| Model / policy | Restarts | Mean mm | Median mm | P05 mm | P95 mm | Zero-length segments |
|---|---:|---:|---:|---:|---:|---:|
| Classic strict | 166 | 5.151553 | 1.889745 | 0.111900 | 23.260862 | 0 |
| Classic loose | 108 | 6.909141 | 2.235374 | 0.079368 | 25.656991 | 1 |
| EMA99u1800 strict | 109 | 6.868737 | 1.549167 | 0.107319 | 25.674974 | 0 |
| EMA99u1800 loose | 73 | 8.700400 | 2.534936 | 0.090248 | 31.510496 | 1 |

## PHerc0332

| Model / policy | Restarts | Mean mm | Median mm | P05 mm | P95 mm | Zero-length segments |
|---|---:|---:|---:|---:|---:|---:|
| Classic strict | 12 | 23.620050 | 17.667043 | 0.910018 | 57.131745 | 0 |
| Classic loose | 9 | 28.344060 | 16.777453 | 0.923118 | 84.675506 | 0 |
| EMA99u1800 strict | 5 | 38.650991 | 27.750803 | 7.744408 | 120.295452 | 0 |
| EMA99u1800 loose | 2 | 53.145113 | 22.319331 | 15.339546 | 152.532149 | 0 |

## PHerc1299

| Model / policy | Restarts | Mean mm | Median mm | P05 mm | P95 mm | Zero-length segments |
|---|---:|---:|---:|---:|---:|---:|
| Classic strict | 44 | 7.538991 | 5.385874 | 0.504970 | 18.912017 | 0 |
| Classic loose | 26 | 11.308487 | 6.361196 | 0.317741 | 35.558385 | 0 |
| EMA99u1800 strict | 30 | 10.177638 | 3.996861 | 0.361235 | 36.338082 | 0 |
| EMA99u1800 loose | 8 | 22.616974 | 15.613233 | 0.662523 | 68.909987 | 0 |

## PHerc1451

| Model / policy | Restarts | Mean mm | Median mm | P05 mm | P95 mm | Zero-length segments |
|---|---:|---:|---:|---:|---:|---:|
| Classic strict | 3 | 6.909904 | 9.547586 | 0.770448 | 11.395592 | 0 |
| Classic loose | 2 | 8.061555 | 11.053252 | 1.746046 | 11.347567 | 0 |
| EMA99u1800 strict | 6 | 4.836933 | 2.395577 | 0.701461 | 11.128854 | 0 |
| EMA99u1800 loose | 0 | 12.092332 | 12.092332 | 11.128854 | 13.055811 | 0 |

## PHercParis4

| Model / policy | Restarts | Mean mm | Median mm | P05 mm | P95 mm | Zero-length segments |
|---|---:|---:|---:|---:|---:|---:|
| Classic strict | 285 | 12.565111 | 4.458509 | 0.197488 | 51.581074 | 1 |
| Classic loose | 147 | 19.642602 | 8.893962 | 0.195724 | 85.486571 | 3 |
| EMA99u1800 strict | 184 | 17.065381 | 6.048898 | 0.243806 | 80.916012 | 0 |
| EMA99u1800 loose | 91 | 25.462632 | 11.289367 | 0.271443 | 89.582314 | 3 |

## Commands and validation

```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_replay_tests
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_thickness40_grace160_20260914_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_eligible_ema99_thickness40_grace160_20260914_v1.json
```

Workspace RelWithDebInfo GCC13 build in volume-cartographer/build/fiber-evaluation. No native CPU tuning; existing local dependencies reused read-only (machine-specific runner, not portable setup). Incremental repeat reports no work. 21 native regressions passed, including terminal excursion failure and factor2 JSON roundtrip; Python inspection suite also run. Both full runs exclusive/all8GPUs,64 logical CPU via existing manager. Rendering disabled; nondefault ellipse rendering is explicitly rejected pending porting. Historical strict galleries unchanged.

Classic loose wall84.0103s (475 new inference tiles/55 published chunks); EMA99 loose wall81.5072s. Times are coordinator-recorded single runs with cached inference and on-demand fills, **not** cold-cache throughput benchmarks; strict timings include different cache/rendering work and are not comparable.

EMA99 frozen update1800,230400 samples from5scroll_json_eligible_128_sd1_16x8_ema_20260914_105436; SHA256 c348f867a6e922c0f7e19928c7982b9325475fd63505e42de325065e3e15a2e8. Classic best91_5k SHA256 f389da7914a6da34506f92204bf5441964e96599339dfe79dfc9c48b67165e17. Native stride4 model output handled by existing adapter.

Percentiles need not all improve when the number and locations of restart segments change. Four zero-length terminal segments occur in each loose result; retained by the common restarts+directions convention, not silently excluded.

## Artifacts

- Classic strict: `/ephemeral/hendrik/fiber3d_train2/evaluations/classic_restart80_cached_20260914/full/summary.json`
- Classic loose: `/ephemeral/hendrik/fiber3d_train2/evaluations/classic_thickness40_grace160_20260914_v2/full/summary.json`
- EMA99u1800 strict: `/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_json_ema99_reference_vis_20260914_v2/full/summary.json`
- EMA99u1800 loose: `/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema99_thickness40_grace160_20260914_v2/full/summary.json`

Config filenames retain v1 but output roots are v2 because initial v1 preparations referenced the old external build; original audits were preserved, not rewritten. Inference cache namespaces are unchanged.
