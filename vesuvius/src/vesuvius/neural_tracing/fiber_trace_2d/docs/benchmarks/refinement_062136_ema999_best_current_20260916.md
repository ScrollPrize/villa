# Refinement run062136 — best and current EMA999

Both snapshots from
`5scroll_json_eligible_refinement_128_sd1_16x8_ema_20260916_062136` were frozen
before evaluation. Best is EMA999 update21300/2726400samples, selected by dense
validation loss0.2257523393. Current is EMA999 update32700/4185600samples with
dense validation loss0.2574588297.

The benchmark is the established original-GT full set:50 fibers,90 eligible
CP-bounded/non-Lasagna spans,180 directed traces and6867.627253 directedmm.
Both use ordinary tracing with embeddings disabled, no rendering, and identical
40/80 acceptance ellipse,160base-voxel excursion grace and80base-voxel restart
episode grouping. Mean is combined directed reference distance divided by
`restarts+1`; terminal spans are included in quantiles.

## Results

Distances are mm. Lower restarts is better; higher distance statistics are
better.

| Model | Restarts | Failures/mm | Mean | Median | P05 | P95 |
|---|---:|---:|---:|---:|---:|---:|
| Previous leader: refinement-212757 EMA999 u22000 | **99** | **0.014415** | **68.676** | **11.129** | 0.227 | 100.802 |
| Refinement-062136 best EMA999 u21300 | 114 | 0.016600 | 59.718 | 10.113 | 0.204 | 90.134 |
| Refinement-062136 current EMA999 u32700 | 109 | 0.015872 | 62.433 | 10.282 | **0.241** | **101.641** |

Currentu32700 beats this run's dense-validation-selected best by5 restarts and
2.714mm (+4.55%) mean distance, despite its worse dense validation loss. It does
not displace the prior u22000 tracing leader.

## Per scroll

| Scroll | Snapshot | Restarts | Mean | Median | P05 | P95 |
|---|---|---:|---:|---:|---:|---:|
| PHerc0139 | best u21300 | 50 | 23.030 | **4.086** | 0.122 | 47.301 |
|  | current u32700 | **48** | **23.970** | 3.617 | **0.217** | **47.601** |
| PHerc0332 | best u21300 | **2** | **141.720** | **22.442** | **15.340** | **152.350** |
|  | current u32700 | 3 | 106.290 | 16.777 | 6.436 | 152.190 |
| PHerc1299 | best u21300 | 9 | 40.711 | 17.793 | **1.861** | 40.544 |
|  | current u32700 | 9 | 40.711 | **17.802** | 1.392 | **40.546** |
| PHerc1451 | best u21300 | 0 | 48.369 | 12.092 | 11.129 | 13.056 |
|  | current u32700 | 0 | 48.369 | 12.092 | 11.129 | 13.056 |
| PHercParis4 | best u21300 | 53 | 89.119 | 17.308 | 0.271 | 111.486 |
|  | current u32700 | **49** | **96.249** | **17.441** | 0.271 | **114.200** |

The two Paris4 annotation collections are aggregated into one physical-scroll
row using the same complete reference-distance partition as the global metric.

## Provenance and timing

| Snapshot | SHA256 | Inference/coordinator | Native replay | Output |
|---|---|---:|---:|---|
| best EMA999 u21300 | `6e2d7ff6d85db3820d4b2500e42c3e1670cf9654b0459dcd798ff081d6948a8e` | 341.550s | 297.995s | `/ephemeral/hendrik/fiber3d_train2/evaluations/refinement_062136_best_ema999_off_20260916_v1` |
| current EMA999 u32700 | `3c13616a8debc61c1e838613ae561adb0631b00db7a9e1b8e947897dde34a597` | 344.026s | 300.340s | `/ephemeral/hendrik/fiber3d_train2/evaluations/refinement_062136_current_ema999_off_20260916_v1` |

The phases overlap and are not additive. Both runs used fresh model-specific
inference, cached source volumes, bf16 inference, and all8 GPUs under exclusive
manager ownership. A separate training job delayed the second manager grant;
that queue wait is excluded from saved timings.

Commands (each preceded by `freeze-snapshot` and `prepare` using the same
runner/config):

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_refinement_062136_best_ema999_off_20260916_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_refinement_062136_current_ema999_off_20260916_v1.json
```

Both summaries completed all180 directions and passed runner provenance/hash
validation. This is one deterministic evaluation per checkpoint, not a
significance estimate.
