# Current EMA999 — cosine40 width and FW/BW

Same 91 reviewed fibers / 182 directions, beam8/lookahead2. Forward ellipse40/80, grace160. FW/BW uses full prefix every512 and endpoints, ellipse20/40, frozen forward width mean. Cosine40 multiplies presence by cos(pi*d/2) for d<1, otherwise0; d=abs(width/mean-1)/0.4. Cumulative committed arithmetic mean; reset on restart. No numerical/parallelism changes; GPUs0–5.

Errors are80-base-voxel grouped episodes; filtered stops ungrouped. Mean=retained GT mm/(all actual stops+1). Median/P05/P95 include terminal and zero-length retained spans. GT controls failures/reseeding; terminal reverse conversions are oracle-timed.

Snapshot: `{"source": "/ephemeral/hendrik/fiber3d_train2/runs/reviewed_normal_width_hv_fullres_128_sd1_bs128_ema_20261006_122417/snapshots/current_ema_0p999.pt", "sha256": "fd9dc4bac7a04bbdbf98d773629b264b902c9c25228482a9adbdbc08e78ef05a", "path": "/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/normalwidth_122417_current_ema999_20261006_v1/frozen/live_inference/_models/fd9dc4bac7a04bbdbf98d773629b264b902c9c25228482a9adbdbc08e78ef05a/snapshot.pt", "step": 12100, "samples_seen": 1548800, "snapshot_kind": "ema", "ema_metadata": {"decay": 0.999, "updates": 12100, "start_step": 0, "best_metric": 1.0072130816323417}, "metric_name": "test_ema_0p999/loss_total", "metric": 1.0072130816323417}`.

| Variant | Errors | Filtered | Actual stops | Kept % | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cosine40_no_reverse | 44 | 0 | 46 | 100.0000 | 29.4931 | 4.7778 | 0.9171 | 18.3442 |
| cosine40_fw512 | 8 | 142 | 150 | 88.7011 | 8.1427 | 2.4983 | 0.0000 | 12.6379 |
| off_no_reverse | 48 | 0 | 50 | 100.0000 | 27.1799 | 4.7803 | 0.6329 | 18.5253 |

## PHerc0139

| Variant | Errors | Filtered | Actual stops | Kept % | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cosine40_no_reverse | 19 | 0 | 19 | 100.0000 | 14.2611 | 3.5879 | 0.7517 | 9.7710 |
| cosine40_fw512 | 3 | 33 | 36 | 88.3898 | 6.8137 | 2.4597 | 0.0000 | 8.9195 |
| off_no_reverse | 18 | 0 | 19 | 100.0000 | 14.2611 | 3.5977 | 0.5568 | 9.0643 |

## PHerc0332

| Variant | Errors | Filtered | Actual stops | Kept % | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cosine40_no_reverse | 4 | 0 | 6 | 100.0000 | 1.5898 | 0.9756 | 0.1270 | 3.0673 |
| cosine40_fw512 | 0 | 6 | 6 | 43.2206 | 0.6871 | 0.0000 | 0.0000 | 2.0034 |
| off_no_reverse | 5 | 0 | 6 | 100.0000 | 1.5898 | 0.9749 | 0.1228 | 3.0737 |

## PHerc1299

| Variant | Errors | Filtered | Actual stops | Kept % | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cosine40_no_reverse | 2 | 0 | 2 | 100.0000 | 49.1424 | 5.2572 | 2.0812 | 16.3750 |
| cosine40_fw512 | 2 | 11 | 13 | 92.1772 | 9.7067 | 4.1518 | 0.4925 | 10.6804 |
| off_no_reverse | 2 | 0 | 2 | 100.0000 | 49.1424 | 5.2572 | 2.0615 | 16.3750 |

## PHerc1451

| Variant | Errors | Filtered | Actual stops | Kept % | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cosine40_no_reverse | 0 | 0 | 0 | 100.0000 | 10.4958 | 5.2479 | 5.2479 | 5.2479 |
| cosine40_fw512 | 0 | 0 | 0 | 100.0000 | 10.4958 | 5.2479 | 5.2479 | 5.2479 |
| off_no_reverse | 0 | 0 | 0 | 100.0000 | 10.4958 | 5.2479 | 5.2479 | 5.2479 |

## PHercParis4

| Variant | Errors | Filtered | Actual stops | Kept % | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cosine40_no_reverse | 19 | 0 | 19 | 100.0000 | 46.5951 | 5.3548 | 1.4629 | 24.1890 |
| cosine40_fw512 | 3 | 92 | 95 | 88.6624 | 8.6067 | 2.6553 | 0.0000 | 13.7676 |
| off_no_reverse | 23 | 0 | 23 | 100.0000 | 38.8293 | 5.1868 | 1.1892 | 24.7379 |

## Matched previous comparisons

| Variant | Errors | Filtered | Actual stops | Kept % | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw u11000 / cosine40_no_reverse | 72 | 0 | 72 | 100.0000 | 18.9887 | 4.3543 | 0.7815 | 18.0994 |
| EMA999 / cosine40_no_reverse | 44 | 0 | 46 | 100.0000 | 29.4931 | 4.7778 | 0.9171 | 18.3442 |
| raw u11000 / cosine40_fw512 | 7 | 195 | 202 | 84.2199 | 5.7509 | 2.1963 | 0.0000 | 9.2657 |
| EMA999 / cosine40_fw512 | 8 | 142 | 150 | 88.7011 | 8.1427 | 2.4983 | 0.0000 | 12.6379 |
| raw u11000 / off_no_reverse | 73 | 0 | 73 | 100.0000 | 18.7321 | 4.3107 | 0.7960 | 18.0030 |
| EMA999 / off_no_reverse | 48 | 0 | 50 | 100.0000 | 27.1799 | 4.7803 | 0.6329 | 18.5253 |

## Timing

| Variant | Wall s | Additional inference tiles |
|---|---:|---:|
| cosine40_no_reverse | 304.25 | 5073 |
| cosine40_fw512 | 205.83 | 1482 |
| off_no_reverse | 142.15 | 528 |

## Reproduce

Control action `reviewed_ema999_width`; same frozen selection reused on resume.

```bash
bash /home/hendrik/villa3/scripts/run_training_failure_mining
```

Artifacts under `/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/normalwidth_122417_current_ema999_20261006_v1*`; timings include startup, resource waits and missing-cache inference.
