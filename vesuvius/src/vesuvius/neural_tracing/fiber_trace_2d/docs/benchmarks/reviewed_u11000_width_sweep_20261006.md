# Width limit and cosine sweep — raw u11000

Same frozen checkpoint as width20 benchmark; same91fibers/182directions, full-prefix512 FW/BW20/40. Same published width/presence/direction cache for ALL rows, including off. Existing on-demand runner fills only missing chunks on GPUs0–5. Cumulative committed arithmetic mean; reverse mean frozen; restart resets. d=abs(width/mean-1)/limit. Linear=max(0,1-d); cosine=cos(pi*d/2) for d<1, else0.

Mean=retained GT mm/(all actual stops+1). Median/P05/P95 are retained spans including terminal/zero-length spans. Errors are80-base-voxel grouped episodes; filtered stops ungrouped. GT failure endpoints/reseeding remain part of this benchmark, not fully GT-free filtering.

| Gate | Errors | Filtered | Actual stops | Kept % | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| linear20 | 10 | 229 | 239 | 81.5878 | 4.7123 | 1.4139 | 0.0000 | 9.0643 |
| off | 3 | 230 | 233 | 81.4628 | 4.8257 | 1.2730 | 0.0000 | 9.0643 |
| linear10 | 3 | 325 | 328 | 73.6043 | 3.1012 | 1.2237 | 0.0000 | 6.6963 |
| linear40 | 4 | 191 | 195 | 84.6966 | 5.9900 | 2.2061 | 0.0000 | 9.8782 |
| linear80 | 4 | 187 | 191 | 85.0249 | 6.1385 | 2.4335 | 0.0000 | 9.8060 |
| cosine10 | 3 | 304 | 307 | 75.4994 | 3.3979 | 1.2389 | 0.0000 | 7.1007 |
| cosine20 | 4 | 213 | 217 | 82.5556 | 5.2494 | 1.8166 | 0.0000 | 9.2657 |
| cosine40 | 7 | 195 | 202 | 84.2199 | 5.7509 | 2.1963 | 0.0000 | 9.2657 |
| cosine80 | 9 | 189 | 200 | 84.6426 | 5.8373 | 1.9990 | 0.0000 | 9.6734 |

## PHerc0139

| Gate | Errors | Filtered | Actual stops | Kept % | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| linear20 | 3 | 53 | 56 | 80.2532 | 4.0158 | 1.2336 | 0.0000 | 8.8833 |
| off | 0 | 48 | 48 | 83.5191 | 4.8615 | 1.5223 | 0.0000 | 8.8833 |
| linear10 | 0 | 83 | 83 | 67.9666 | 2.3078 | 0.6121 | 0.0000 | 4.9615 |
| linear40 | 2 | 39 | 41 | 86.2737 | 5.8588 | 2.1802 | 0.0000 | 8.8833 |
| linear80 | 2 | 36 | 38 | 87.0826 | 6.3687 | 2.4009 | 0.0000 | 8.8833 |
| cosine10 | 1 | 76 | 77 | 70.8377 | 2.5903 | 1.2185 | 0.0000 | 5.6669 |
| cosine20 | 2 | 50 | 52 | 80.7486 | 4.3455 | 1.2441 | 0.0000 | 8.8833 |
| cosine40 | 2 | 34 | 36 | 89.0518 | 6.8647 | 2.4517 | 0.0000 | 9.0643 |
| cosine80 | 4 | 40 | 44 | 86.0845 | 5.4563 | 1.7661 | 0.0000 | 8.8833 |

## PHerc0332

| Gate | Errors | Filtered | Actual stops | Kept % | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| linear20 | 0 | 7 | 7 | 29.5697 | 0.4113 | 0.0000 | 0.0000 | 1.7656 |
| off | 0 | 7 | 7 | 36.5108 | 0.5079 | 0.0000 | 0.0000 | 1.6848 |
| linear10 | 0 | 6 | 6 | 37.0029 | 0.5883 | 0.0000 | 0.0000 | 1.7752 |
| linear40 | 0 | 7 | 7 | 50.2664 | 0.6992 | 0.5964 | 0.0000 | 1.7003 |
| linear80 | 0 | 7 | 7 | 30.7421 | 0.4276 | 0.0000 | 0.0000 | 1.8568 |
| cosine10 | 0 | 9 | 9 | 18.3459 | 0.2042 | 0.0000 | 0.0000 | 1.0427 |
| cosine20 | 0 | 7 | 7 | 29.5413 | 0.4109 | 0.0000 | 0.0000 | 1.7637 |
| cosine40 | 0 | 7 | 7 | 39.2746 | 0.5463 | 0.0000 | 0.0000 | 1.6983 |
| cosine80 | 0 | 7 | 7 | 39.9821 | 0.5562 | 0.0000 | 0.0000 | 1.7320 |

## PHerc1299

| Gate | Errors | Filtered | Actual stops | Kept % | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| linear20 | 1 | 7 | 8 | 94.8668 | 15.5399 | 4.4802 | 1.2443 | 9.2657 |
| off | 0 | 4 | 4 | 96.5925 | 28.4807 | 4.5889 | 1.5653 | 9.2657 |
| linear10 | 0 | 13 | 13 | 90.3775 | 9.5172 | 4.2939 | 0.1265 | 9.5913 |
| linear40 | 0 | 5 | 5 | 95.7249 | 23.5208 | 4.5889 | 0.9187 | 9.2657 |
| linear80 | 0 | 6 | 6 | 95.2931 | 20.0697 | 4.5889 | 0.9464 | 9.2657 |
| cosine10 | 0 | 12 | 12 | 90.9963 | 10.3195 | 4.3259 | 1.2446 | 9.2657 |
| cosine20 | 1 | 6 | 7 | 95.3201 | 17.5660 | 4.5889 | 0.9585 | 9.2657 |
| cosine40 | 1 | 6 | 7 | 95.2976 | 17.5618 | 4.5889 | 0.9586 | 9.2657 |
| cosine80 | 0 | 4 | 4 | 96.5898 | 28.4799 | 4.5889 | 1.5650 | 9.7251 |

## PHerc1451

| Gate | Errors | Filtered | Actual stops | Kept % | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| linear20 | 0 | 3 | 3 | 64.0593 | 1.6809 | 0.2783 | 0.0000 | 3.6857 |
| off | 0 | 4 | 4 | 61.3923 | 1.2887 | 1.2421 | 0.0505 | 3.2120 |
| linear10 | 0 | 3 | 3 | 64.5018 | 1.6925 | 1.2310 | 0.0584 | 2.7121 |
| linear40 | 0 | 3 | 3 | 64.6399 | 1.6961 | 0.3110 | 0.0000 | 3.7056 |
| linear80 | 0 | 3 | 3 | 64.1661 | 1.6837 | 0.2857 | 0.0000 | 3.6861 |
| cosine10 | 0 | 4 | 4 | 61.2020 | 1.2847 | 1.2087 | 0.4817 | 2.2209 |
| cosine20 | 0 | 3 | 3 | 64.1758 | 1.6839 | 0.2589 | 0.0000 | 3.6931 |
| cosine40 | 0 | 3 | 3 | 64.1864 | 1.6842 | 1.2273 | 0.0541 | 3.4456 |
| cosine80 | 0 | 3 | 3 | 64.1676 | 1.6837 | 1.2270 | 0.0540 | 3.4453 |

## PHercParis4

| Gate | Errors | Filtered | Actual stops | Kept % | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| linear20 | 6 | 159 | 165 | 80.7142 | 4.5312 | 1.5806 | 0.0000 | 9.8734 |
| off | 3 | 167 | 170 | 79.2027 | 4.3163 | 1.2491 | 0.0000 | 9.3245 |
| linear10 | 3 | 220 | 223 | 73.2159 | 3.0460 | 1.2256 | 0.0000 | 6.8843 |
| linear40 | 2 | 137 | 139 | 83.1063 | 5.5319 | 1.8779 | 0.0000 | 11.1812 |
| linear80 | 2 | 135 | 137 | 83.6539 | 5.6491 | 1.9367 | 0.0000 | 10.1551 |
| cosine10 | 2 | 203 | 205 | 75.3181 | 3.4072 | 1.2408 | 0.0000 | 7.3405 |
| cosine20 | 1 | 147 | 148 | 81.9294 | 5.1242 | 2.1181 | 0.0000 | 10.8033 |
| cosine40 | 4 | 145 | 149 | 81.7508 | 5.0789 | 1.6565 | 0.0000 | 9.8060 |
| cosine80 | 5 | 135 | 142 | 83.0751 | 5.4138 | 1.8279 | 0.0000 | 10.9906 |

## Runtime and cache extension

| Gate | Wall seconds | Additional tiles | Published chunks |
|---|---:|---:|---:|
| linear20 | 152.28 | 0 | 0 |
| off | 179.05 | 1529 | 504 |
| linear10 | 168.22 | 1539 | 500 |
| linear40 | 172.71 | 443 | 124 |
| linear80 | 156.07 | 387 | 117 |
| cosine10 | 154.21 | 708 | 197 |
| cosine20 | 157.05 | 354 | 84 |
| cosine40 | 156.19 | 306 | 75 |
| cosine80 | 147.20 | 301 | 94 |

One run per configuration; times include cache validation, setup and any demand GPU inference. All published predictions are reused; new paths can require additional chunks. Earlier strict cache-only attempts are preserved separately, not counted as trace failures.

## Reproduce

Control: `{"action": "reviewed_normal_width", "width_sweep": true, "fill_missing": true}`.

```bash
bash /home/hendrik/villa3/scripts/run_training_failure_mining
```

Artifacts and audited timing/provenance: `/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/normalwidth_u11000_gate_sweep_demand_20261006_v1`. Failed missing-cache trials are not scored. Add report_only:true to regenerate this report. Native linear20 repeat is the unchanged-code control.
