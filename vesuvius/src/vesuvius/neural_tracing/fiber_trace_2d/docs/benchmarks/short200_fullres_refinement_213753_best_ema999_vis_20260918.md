# Short200 — full-resolution refinement best EMA999

## Binding

- Benchmark: `/ephemeral/hendrik/fiber3d_train2/evaluations/short_failures_200_20260916_v3/manifest.json`
- Checkpoint SHA256: `5ec278f71b15b27187b55805310176fe2fa3b00c5264fc528b95bc6d0690206e`
- Prediction cache namespace: `fullres_refinement_213753_best_ema999_20260917_v2_chunk64`
- Inference geometry: tile 256, chunk 64, scaledown 0, bf16
- Output: `/ephemeral/hendrik/fiber3d_train2/evaluations/short200_fullres_refinement_213753_best_ema999_vis_20260918_v1`

Command:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_short200_fullres_refinement_213753_best_ema999_vis_20260918_v1.json
```

## Result

| Directed pieces | Directed mm | Filtered restarts | Mean mm | Median mm | P05 mm | P95 mm |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 662.089848 | 56 | 11.615611 | 1.724047 | 0.137536 | 8.030887 |

Native replay/inline rendering completed in 193.443 s; total benchmark reporting
and review export took 237.084 s. The repaired scheduling path reached 129
simultaneous trace waiters, demonstrating same-source direction concurrency.

Validation: 200 trace artifacts, 200 valid `render.json` galleries, 200 declared
and present JPEG pages, and 200 review-index rows. No incomplete marker remains.
The sorted inspection index and VC3D exports are in the output `README.md` and
`review_index.json`.
