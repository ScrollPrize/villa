# Width presence modulation without FW/BW — raw u11000

Full91-fiber/182-direction cohort. Same published width/presence/direction predictions, checkpoint 2408ece6ffa2251655709cf3987357f1892a573828929934266145f09f9d2eba. FW/BW disabled: no periodic or endpoint reverse checks. All other tracing, inference precision and worker settings unchanged. Missing cache data uses existing GPU0–5 demand inference.

Cosine multiplies sampled presence by cos(pi*d/2) for d<1, else0, where d=abs(width/committed_mean-1)/tolerance (0.10 or0.40). This is additional to direction alignment scoring and smoothness, not a separate stop criterion. Mean is cumulative over seed+committed points and resets at ordinary restart.

Errors are80-base-voxel grouped restart episodes. Actual stops includes all individual failures; mean=retained GT mm/(actual stops+1), virtually concatenating all references. Median/quantiles include terminal and zero-length spans. GT drives error detection/restarts.

| Variant | Errors | Filtered | Actual stops | Kept % | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 73 | 0 | 73 | 100.0000 | 18.7321 | 4.3107 | 0.7960 | 18.0030 |
| cosine10 | 126 | 0 | 128 | 99.9988 | 10.7454 | 3.5894 | 0.6333 | 11.4314 |
| cosine40 | 72 | 0 | 72 | 100.0000 | 18.9887 | 4.3543 | 0.7815 | 18.0994 |

## PHerc0139

| Variant | Errors | Filtered | Actual stops | Kept % | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 26 | 0 | 26 | 100.0000 | 10.5638 | 3.1935 | 0.6913 | 9.2665 |
| cosine10 | 40 | 0 | 40 | 100.0000 | 6.9566 | 2.4536 | 0.5767 | 8.8833 |
| cosine40 | 24 | 0 | 24 | 100.0000 | 11.4089 | 3.3148 | 0.5715 | 9.4988 |

## PHerc0332

| Variant | Errors | Filtered | Actual stops | Kept % | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 6 | 0 | 6 | 100.0000 | 1.5898 | 1.2400 | 0.3071 | 2.4535 |
| cosine10 | 5 | 0 | 5 | 100.0000 | 1.8548 | 2.0160 | 0.4060 | 2.6518 |
| cosine40 | 6 | 0 | 6 | 100.0000 | 1.5898 | 1.2212 | 0.4377 | 2.4197 |

## PHerc1299

| Variant | Errors | Filtered | Actual stops | Kept % | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 0 | 0 | 0 | 100.0000 | 147.4272 | 5.4810 | 4.2442 | 17.2114 |
| cosine10 | 2 | 0 | 2 | 100.0000 | 49.1424 | 5.2572 | 3.6043 | 10.7213 |
| cosine40 | 1 | 0 | 1 | 100.0000 | 73.7136 | 5.4810 | 4.2485 | 10.5864 |

## PHerc1451

| Variant | Errors | Filtered | Actual stops | Kept % | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 2 | 0 | 2 | 100.0000 | 3.4986 | 3.5120 | 1.9135 | 5.0743 |
| cosine10 | 1 | 0 | 1 | 100.0000 | 5.2479 | 3.4875 | 1.9331 | 5.0718 |
| cosine40 | 2 | 0 | 2 | 100.0000 | 3.4986 | 3.5117 | 1.9137 | 5.0743 |

## PHercParis4

| Variant | Errors | Filtered | Actual stops | Kept % | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 39 | 0 | 39 | 100.0000 | 23.2976 | 4.8688 | 1.2974 | 19.3932 |
| cosine10 | 78 | 0 | 80 | 99.9982 | 11.5048 | 3.8430 | 0.6991 | 12.2667 |
| cosine40 | 39 | 0 | 39 | 100.0000 | 23.2976 | 4.8688 | 1.1721 | 20.0673 |

## Prior FW/BW512 context

| Variant | Errors | Filtered | Actual stops | Kept % | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 3 | 230 | 233 | 81.4628 | 4.8257 | 1.2730 | 0.0000 | 9.0643 |
| cosine10 | 3 | 304 | 307 | 75.4994 | 3.3979 | 1.2389 | 0.0000 | 7.1007 |
| cosine40 | 7 | 195 | 202 | 84.2199 | 5.7509 | 2.1963 | 0.0000 | 9.2657 |

## Runtime and reproduction

| Variant | Wall seconds | Additional tiles |
|---|---:|---:|
| off | 117.14 | 194 |
| cosine10 | 128.56 | 521 |
| cosine40 | 100.38 | 117 |

One run each; times include startup/cache checks and any inference. Control: `{"action":"reviewed_normal_width","width_no_reverse":true}`.

```bash
bash /home/hendrik/villa3/scripts/run_training_failure_mining
```

Companion JSON records exact summary paths, checkpoint and trace flags. Add report_only:true to regenerate without rerunning. Completed runs are skipped on normal resume.
