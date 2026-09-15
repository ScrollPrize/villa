# EMA999 u11800: fibers ranked by failures per distance

Source: /ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1

Generated from the saved per_fiber.json; no tracing, inference, rendering or recounting was performed. Checkpoint SHA256: 4d7c3b7ed06a29a344f8f21175158bf40c42ade440943ff68aa2775ee0cc1d82. Embeddings disabled.

## Metric

Failures are the evaluation's **counted/filtered restarts**, not every visible restart marker. Rank = counted restarts / combined forward-and-reverse eligible reference distance (mm), descending. Both directions and every eligible run of an original fiber are combined. Extrapolated and Lasagna-interpolated spans contribute neither distance nor failures. One-way eligible length is also shown to avoid a factor-of-two ambiguity. Mean = directed distance / (restarts + 1), matching the current reporting convention; the ranking itself does not use this mean. Zero-failure fibers are tied at zero, ordered by path.

Acceptance: normal thickness 40 base voxels, tangential width 80, excursion grace 160, restart filtering 80 along reference arc. Collection label PHerc1229 maps to scroll PHerc1299.

**50 fibers; 180 eligible run-directions; 113 restarts; 6867.627253 directed mm; 0.016454 restarts/mm.**

## Ranked fibers

| Rank | Scroll / collection | Fiber | Restarts | One-way eligible mm | Directed mm | Restarts/mm ↓ | Mean mm | Cases |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | PHerc0139 / PHerc0139 | [lt_20260831T061702750_000026.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHerc0139/lt_20260831T061702750_000026.json) | 8 | 34.638 | 69.276 | 0.115481 | 7.697 | F [42](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/42/trace.json) (3); F [44](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/44/trace.json) (1); R [43](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/43/trace.json) (3); R [45](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/45/trace.json) (1) |
| 2 | PHercParis4 / PHercParis4_eval_fibers | [anon_20260827T134102921_000271.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_eval_fibers/anon_20260827T134102921_000271.json) | 2 | 9.907 | 19.813 | 0.100942 | 6.604 | F [104](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/104/trace.json) (1); F [106](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/106/trace.json) (1); R [105](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/105/trace.json) (0); R [107](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/107/trace.json) (0) |
| 3 | PHercParis4 / PHercParis4_fibers | [kb_20260729T182039882_000211.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_fibers/kb_20260729T182039882_000211.json) | 1 | 5.578 | 11.156 | 0.089640 | 5.578 | F [172](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/172/trace.json) (0); R [173](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/173/trace.json) (1) |
| 4 | PHercParis4 / PHercParis4_fibers | [kb_20260729T173847941_000210.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_fibers/kb_20260729T173847941_000210.json) | 20 | 112.446 | 224.892 | 0.088932 | 10.709 | F [166](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/166/trace.json) (8); F [168](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/168/trace.json) (1); F [170](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/170/trace.json) (1); R [167](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/167/trace.json) (8); R [169](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/169/trace.json) (1); R [171](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/171/trace.json) (1) |
| 5 | PHerc0139 / PHerc0139 | [kb_20260905T035141671_000093.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHerc0139/kb_20260905T035141671_000093.json) | 16 | 90.496 | 180.992 | 0.088402 | 10.647 | F [16](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/16/trace.json) (0); F [18](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/18/trace.json) (0); F [20](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/20/trace.json) (0); F [22](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/22/trace.json) (1); F [24](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/24/trace.json) (1); F [26](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/26/trace.json) (1); F [28](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/28/trace.json) (5); F [30](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/30/trace.json) (0); R [17](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/17/trace.json) (0); R [19](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/19/trace.json) (0); R [21](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/21/trace.json) (0); R [23](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/23/trace.json) (1); R [25](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/25/trace.json) (1); R [27](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/27/trace.json) (2); R [29](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/29/trace.json) (4); R [31](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/31/trace.json) (0) |
| 6 | PHercParis4 / PHercParis4_eval_fibers | [anon_20260815T025206849_000041.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_eval_fibers/anon_20260815T025206849_000041.json) | 4 | 30.617 | 61.235 | 0.065323 | 12.247 | F [84](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/84/trace.json) (2); R [85](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/85/trace.json) (2) |
| 7 | PHerc1299 / PHerc1229 | [lt_20260825T075316065_000069.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHerc1229/lt_20260825T075316065_000069.json) | 2 | 16.172 | 32.344 | 0.061835 | 10.781 | F [76](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/76/trace.json) (1); R [77](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/77/trace.json) (1) |
| 8 | PHerc0139 / PHerc0139 | [kb_20260901T224443942_000068.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHerc0139/kb_20260901T224443942_000068.json) | 12 | 120.770 | 241.540 | 0.049681 | 18.580 | F [8](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/8/trace.json) (5); F [10](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/10/trace.json) (0); F [12](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/12/trace.json) (0); F [14](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/14/trace.json) (0); R [9](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/9/trace.json) (6); R [11](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/11/trace.json) (1); R [13](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/13/trace.json) (0); R [15](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/15/trace.json) (0) |
| 9 | PHerc0139 / PHerc0139 | [kb_20260827T204421830_000022.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHerc0139/kb_20260827T204421830_000022.json) | 2 | 22.417 | 44.835 | 0.044608 | 14.945 | F [0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/0/trace.json) (1); F [2](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/2/trace.json) (0); R [1](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/1/trace.json) (1); R [3](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/3/trace.json) (0) |
| 10 | PHerc0139 / PHerc0139 | [lt_20260909T084022344_000109.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHerc0139/lt_20260909T084022344_000109.json) | 1 | 12.746 | 25.492 | 0.039227 | 12.746 | F [58](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/58/trace.json) (0); F [60](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/60/trace.json) (1); R [59](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/59/trace.json) (0); R [61](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/61/trace.json) (0) |
| 11 | PHercParis4 / PHercParis4_eval_fibers | [anon_20260905T163237868_000469.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_eval_fibers/anon_20260905T163237868_000469.json) | 7 | 92.068 | 184.137 | 0.038015 | 23.017 | F [146](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/146/trace.json) (0); F [148](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/148/trace.json) (0); F [150](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/150/trace.json) (1); F [152](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/152/trace.json) (0); F [154](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/154/trace.json) (0); F [156](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/156/trace.json) (3); R [147](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/147/trace.json) (1); R [149](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/149/trace.json) (0); R [151](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/151/trace.json) (0); R [153](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/153/trace.json) (0); R [155](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/155/trace.json) (0); R [157](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/157/trace.json) (2) |
| 12 | PHercParis4 / PHercParis4_eval_fibers | [anon_20260902T214326941_000399.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_eval_fibers/anon_20260902T214326941_000399.json) | 4 | 54.005 | 108.010 | 0.037034 | 21.602 | F [130](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/130/trace.json) (2); R [131](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/131/trace.json) (2) |
| 13 | PHerc1299 / PHerc1229 | [lt_20260824T091041867_000056.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHerc1229/lt_20260824T091041867_000056.json) | 5 | 71.464 | 142.928 | 0.034983 | 23.821 | F [74](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/74/trace.json) (3); R [75](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/75/trace.json) (2) |
| 14 | PHerc0139 / PHerc0139 | [lt_20260909T082646148_000107.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHerc0139/lt_20260909T082646148_000107.json) | 1 | 16.568 | 33.137 | 0.030178 | 16.568 | F [56](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/56/trace.json) (1); R [57](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/57/trace.json) (0) |
| 15 | PHerc0139 / PHerc0139 | [kb_20260908T172232906_000117.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHerc0139/kb_20260908T172232906_000117.json) | 1 | 18.885 | 37.771 | 0.026476 | 18.885 | F [32](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/32/trace.json) (0); F [34](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/34/trace.json) (0); F [36](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/36/trace.json) (0); F [38](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/38/trace.json) (0); F [40](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/40/trace.json) (0); R [33](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/33/trace.json) (0); R [35](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/35/trace.json) (0); R [37](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/37/trace.json) (1); R [39](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/39/trace.json) (0); R [41](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/41/trace.json) (0) |
| 16 | PHerc0139 / PHerc0139 | [lt_20260904T062829219_000052.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHerc0139/lt_20260904T062829219_000052.json) | 5 | 136.802 | 273.603 | 0.018275 | 45.601 | F [46](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/46/trace.json) (0); F [48](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/48/trace.json) (2); F [50](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/50/trace.json) (0); F [52](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/52/trace.json) (1); R [47](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/47/trace.json) (0); R [49](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/49/trace.json) (1); R [51](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/51/trace.json) (1); R [53](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/53/trace.json) (0) |
| 17 | PHerc0139 / PHerc0139 | [kb_20260828T155910844_000043.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHerc0139/kb_20260828T155910844_000043.json) | 4 | 131.903 | 263.807 | 0.015163 | 52.761 | F [4](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/4/trace.json) (2); F [6](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/6/trace.json) (0); R [5](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/5/trace.json) (2); R [7](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/7/trace.json) (0) |
| 18 | PHercParis4 / PHercParis4_eval_fibers | [anon_20260821T221016194_000135.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_eval_fibers/anon_20260821T221016194_000135.json) | 3 | 105.874 | 211.748 | 0.014168 | 52.937 | F [90](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/90/trace.json) (2); R [91](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/91/trace.json) (1) |
| 19 | PHercParis4 / PHercParis4_fibers | [kb_20260711T043119415_000150.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_fibers/kb_20260711T043119415_000150.json) | 3 | 110.123 | 220.246 | 0.013621 | 55.061 | F [164](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/164/trace.json) (2); R [165](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/165/trace.json) (1) |
| 20 | PHercParis4 / PHercParis4_eval_fibers | [anon_20260907T141106877_000497.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_eval_fibers/anon_20260907T141106877_000497.json) | 2 | 76.689 | 153.378 | 0.013040 | 51.126 | F [158](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/158/trace.json) (0); F [160](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/160/trace.json) (2); F [162](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/162/trace.json) (0); R [159](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/159/trace.json) (0); R [161](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/161/trace.json) (0); R [163](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/163/trace.json) (0) |
| 21 | PHercParis4 / PHercParis4_eval_fibers | [anon_20260902T181618881_000388.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_eval_fibers/anon_20260902T181618881_000388.json) | 3 | 120.047 | 240.093 | 0.012495 | 60.023 | F [128](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/128/trace.json) (2); R [129](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/129/trace.json) (1) |
| 22 | PHerc0332 / PHerc0332 | [dj_20260810T113807458_000001.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHerc0332/dj_20260810T113807458_000001.json) | 4 | 180.463 | 360.927 | 0.011083 | 72.185 | F [62](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/62/trace.json) (3); R [63](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/63/trace.json) (1) |
| 23 | PHercParis4 / PHercParis4_eval_fibers | [anon_20260902T125309620_000368.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_eval_fibers/anon_20260902T125309620_000368.json) | 1 | 48.058 | 96.116 | 0.010404 | 48.058 | F [120](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/120/trace.json) (0); F [122](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/122/trace.json) (1); R [121](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/121/trace.json) (0); R [123](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/123/trace.json) (0) |
| 24 | PHercParis4 / PHercParis4_eval_fibers | [anon_20260826T172849645_000255.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_eval_fibers/anon_20260826T172849645_000255.json) | 1 | 89.039 | 178.078 | 0.005616 | 89.039 | F [102](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/102/trace.json) (1); R [103](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/103/trace.json) (0) |
| 25 | PHercParis4 / PHercParis4_fibers | [kb_20260803T040828806_000224.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_fibers/kb_20260803T040828806_000224.json) | 1 | 117.687 | 235.373 | 0.004249 | 117.687 | F [174](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/174/trace.json) (1); R [175](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/175/trace.json) (0) |
| 26 | PHerc0139 / PHerc0139 | [lt_20260909T071741261_000094.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHerc0139/lt_20260909T071741261_000094.json) | 0 | 2.051 | 4.102 | 0.000000 | 4.102 | F [54](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/54/trace.json) (0); R [55](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/55/trace.json) (0) |
| 27 | PHerc0332 / PHerc0332 | [kb_20260822T033517433_000058.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHerc0332/kb_20260822T033517433_000058.json) | 0 | 15.340 | 30.679 | 0.000000 | 30.679 | F [64](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/64/trace.json) (0); R [65](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/65/trace.json) (0) |
| 28 | PHerc0332 / PHerc0332 | [kb_20260822T164832186_000062.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHerc0332/kb_20260822T164832186_000062.json) | 0 | 16.777 | 33.555 | 0.000000 | 33.555 | F [66](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/66/trace.json) (0); R [67](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/67/trace.json) (0) |
| 29 | PHerc1299 / PHerc1229 | [lt_20260819T050805459_000002.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHerc1229/lt_20260819T050805459_000002.json) | 0 | 40.413 | 80.825 | 0.000000 | 80.825 | F [68](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/68/trace.json) (0); R [69](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/69/trace.json) (0) |
| 30 | PHerc1299 / PHerc1229 | [lt_20260819T081412107_000011.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHerc1229/lt_20260819T081412107_000011.json) | 0 | 36.132 | 72.265 | 0.000000 | 72.265 | F [70](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/70/trace.json) (0); R [71](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/71/trace.json) (0) |
| 31 | PHerc1299 / PHerc1229 | [lt_20260824T083243313_000053.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHerc1229/lt_20260824T083243313_000053.json) | 0 | 39.372 | 78.743 | 0.000000 | 78.743 | F [72](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/72/trace.json) (0); R [73](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/73/trace.json) (0) |
| 32 | PHerc1451 / PHerc1451 | [dj_20260903T195009387_000005.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHerc1451/dj_20260903T195009387_000005.json) | 0 | 13.056 | 26.112 | 0.000000 | 26.112 | F [78](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/78/trace.json) (0); R [79](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/79/trace.json) (0) |
| 33 | PHerc1451 / PHerc1451 | [dj_20260903T200017539_000006.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHerc1451/dj_20260903T200017539_000006.json) | 0 | 11.129 | 22.258 | 0.000000 | 22.258 | F [80](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/80/trace.json) (0); R [81](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/81/trace.json) (0) |
| 34 | PHercParis4 / PHercParis4_eval_fibers | [anon_20260815T020414697_000008.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_eval_fibers/anon_20260815T020414697_000008.json) | 0 | 24.545 | 49.090 | 0.000000 | 49.090 | F [82](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/82/trace.json) (0); R [83](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/83/trace.json) (0) |
| 35 | PHercParis4 / PHercParis4_eval_fibers | [anon_20260815T135300158_000075.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_eval_fibers/anon_20260815T135300158_000075.json) | 0 | 21.083 | 42.166 | 0.000000 | 42.166 | F [86](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/86/trace.json) (0); R [87](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/87/trace.json) (0) |
| 36 | PHercParis4 / PHercParis4_eval_fibers | [anon_20260816T224409952_000122.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_eval_fibers/anon_20260816T224409952_000122.json) | 0 | 32.548 | 65.095 | 0.000000 | 65.095 | F [88](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/88/trace.json) (0); R [89](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/89/trace.json) (0) |
| 37 | PHercParis4 / PHercParis4_eval_fibers | [anon_20260822T103908823_000165.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_eval_fibers/anon_20260822T103908823_000165.json) | 0 | 112.370 | 224.740 | 0.000000 | 224.740 | F [92](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/92/trace.json) (0); R [93](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/93/trace.json) (0) |
| 38 | PHercParis4 / PHercParis4_eval_fibers | [anon_20260822T121633375_000173.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_eval_fibers/anon_20260822T121633375_000173.json) | 0 | 131.521 | 263.041 | 0.000000 | 263.041 | F [94](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/94/trace.json) (0); F [96](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/96/trace.json) (0); F [98](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/98/trace.json) (0); R [95](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/95/trace.json) (0); R [97](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/97/trace.json) (0); R [99](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/99/trace.json) (0) |
| 39 | PHercParis4 / PHercParis4_eval_fibers | [anon_20260824T032219305_000197.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_eval_fibers/anon_20260824T032219305_000197.json) | 0 | 40.893 | 81.787 | 0.000000 | 81.787 | F [100](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/100/trace.json) (0); R [101](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/101/trace.json) (0) |
| 40 | PHercParis4 / PHercParis4_eval_fibers | [anon_20260901T165827570_000316.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_eval_fibers/anon_20260901T165827570_000316.json) | 0 | 64.478 | 128.956 | 0.000000 | 128.956 | F [108](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/108/trace.json) (0); R [109](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/109/trace.json) (0) |
| 41 | PHercParis4 / PHercParis4_eval_fibers | [anon_20260901T170547855_000317.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_eval_fibers/anon_20260901T170547855_000317.json) | 0 | 85.785 | 171.571 | 0.000000 | 171.571 | F [110](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/110/trace.json) (0); F [112](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/112/trace.json) (0); R [111](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/111/trace.json) (0); R [113](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/113/trace.json) (0) |
| 42 | PHercParis4 / PHercParis4_eval_fibers | [anon_20260901T201915394_000335.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_eval_fibers/anon_20260901T201915394_000335.json) | 0 | 17.913 | 35.826 | 0.000000 | 35.826 | F [114](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/114/trace.json) (0); F [116](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/116/trace.json) (0); R [115](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/115/trace.json) (0); R [117](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/117/trace.json) (0) |
| 43 | PHercParis4 / PHercParis4_eval_fibers | [anon_20260902T103444019_000352.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_eval_fibers/anon_20260902T103444019_000352.json) | 0 | 74.138 | 148.276 | 0.000000 | 148.276 | F [118](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/118/trace.json) (0); R [119](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/119/trace.json) (0) |
| 44 | PHercParis4 / PHercParis4_eval_fibers | [anon_20260902T131435457_000370.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_eval_fibers/anon_20260902T131435457_000370.json) | 0 | 72.805 | 145.611 | 0.000000 | 145.611 | F [124](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/124/trace.json) (0); F [126](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/126/trace.json) (0); R [125](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/125/trace.json) (0); R [127](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/127/trace.json) (0) |
| 45 | PHercParis4 / PHercParis4_eval_fibers | [anon_20260902T215234337_000400.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_eval_fibers/anon_20260902T215234337_000400.json) | 0 | 130.920 | 261.841 | 0.000000 | 261.841 | F [132](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/132/trace.json) (0); R [133](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/133/trace.json) (0) |
| 46 | PHercParis4 / PHercParis4_eval_fibers | [anon_20260903T214715058_000404.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_eval_fibers/anon_20260903T214715058_000404.json) | 0 | 172.877 | 345.753 | 0.000000 | 345.753 | F [134](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/134/trace.json) (0); F [136](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/136/trace.json) (0); R [135](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/135/trace.json) (0); R [137](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/137/trace.json) (0) |
| 47 | PHercParis4 / PHercParis4_eval_fibers | [anon_20260904T145959056_000421.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_eval_fibers/anon_20260904T145959056_000421.json) | 0 | 131.434 | 262.868 | 0.000000 | 262.868 | F [138](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/138/trace.json) (0); F [140](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/140/trace.json) (0); F [142](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/142/trace.json) (0); R [139](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/139/trace.json) (0); R [141](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/141/trace.json) (0); R [143](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/143/trace.json) (0) |
| 48 | PHercParis4 / PHercParis4_eval_fibers | [anon_20260904T210541536_000446.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_eval_fibers/anon_20260904T210541536_000446.json) | 0 | 87.928 | 175.857 | 0.000000 | 175.857 | F [144](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/144/trace.json) (0); R [145](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/145/trace.json) (0) |
| 49 | PHercParis4 / PHercParis4_fibers | [sm_20260706T204346695_000119.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_fibers/sm_20260706T204346695_000119.json) | 0 | 145.362 | 290.724 | 0.000000 | 290.724 | F [176](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/176/trace.json) (0); R [177](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/177/trace.json) (0) |
| 50 | PHercParis4 / PHercParis4_fibers | [sm_20260707T184533000_000122.json](/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all/PHercParis4_fibers/sm_20260707T184533000_000122.json) | 0 | 87.481 | 174.961 | 0.000000 | 174.961 | F [178](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/178/trace.json) (0); R [179](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/179/trace.json) (0) |

Case parentheses contain counted restarts in that case. F = forward; R = reverse.

## Inspection links in ranked order

### 1. PHerc0139 / lt_20260831T061702750_000026.json

8 restarts / 69.276 directed mm = 0.115481 restarts/mm.

- forward case **42**: 3 restarts, 33.391 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/42/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/42/rendered/strips_0.jpg).
- forward case **44**: 1 restarts, 1.247 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/44/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/44/rendered/strips_0.jpg).
- reverse case **43**: 3 restarts, 33.391 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/43/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/43/rendered/strips_0.jpg).
- reverse case **45**: 1 restarts, 1.247 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/45/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/45/rendered/strips_0.jpg).

### 2. PHercParis4_eval_fibers / anon_20260827T134102921_000271.json

2 restarts / 19.813 directed mm = 0.100942 restarts/mm.

- forward case **104**: 1 restarts, 5.191 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/104/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/104/rendered/strips_0.jpg).
- forward case **106**: 1 restarts, 4.716 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/106/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/106/rendered/strips_0.jpg).
- reverse case **105**: 0 restarts, 5.191 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/105/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/105/rendered/strips_0.jpg).
- reverse case **107**: 0 restarts, 4.716 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/107/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/107/rendered/strips_0.jpg).

### 3. PHercParis4_fibers / kb_20260729T182039882_000211.json

1 restarts / 11.156 directed mm = 0.089640 restarts/mm.

- forward case **172**: 0 restarts, 5.578 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/172/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/172/rendered/strips_0.jpg).
- reverse case **173**: 1 restarts, 5.578 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/173/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/173/rendered/strips_0.jpg).

### 4. PHercParis4_fibers / kb_20260729T173847941_000210.json

20 restarts / 224.892 directed mm = 0.088932 restarts/mm.

- forward case **166**: 8 restarts, 35.400 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/166/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/166/rendered/strips_0.jpg).
- forward case **168**: 1 restarts, 8.922 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/168/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/168/rendered/strips_0.jpg).
- forward case **170**: 1 restarts, 68.124 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/170/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/170/rendered/strips_0.jpg).
- reverse case **167**: 8 restarts, 35.400 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/167/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/167/rendered/strips_0.jpg).
- reverse case **169**: 1 restarts, 8.922 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/169/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/169/rendered/strips_0.jpg).
- reverse case **171**: 1 restarts, 68.124 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/171/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/171/rendered/strips_0.jpg).

### 5. PHerc0139 / kb_20260905T035141671_000093.json

16 restarts / 180.992 directed mm = 0.088402 restarts/mm.

- forward case **16**: 0 restarts, 4.346 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/16/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/16/rendered/strips_0.jpg).
- forward case **18**: 0 restarts, 0.091 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/18/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/18/rendered/strips_0.jpg).
- forward case **20**: 0 restarts, 8.152 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/20/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/20/rendered/strips_0.jpg).
- forward case **22**: 1 restarts, 1.479 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/22/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/22/rendered/strips_0.jpg).
- forward case **24**: 1 restarts, 30.074 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/24/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/24/rendered/strips_0.jpg).
- forward case **26**: 1 restarts, 25.559 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/26/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/26/rendered/strips_0.jpg).
- forward case **28**: 5 restarts, 19.817 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/28/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/28/rendered/strips_0.jpg).
- forward case **30**: 0 restarts, 0.979 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/30/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/30/rendered/strips_0.jpg).
- reverse case **17**: 0 restarts, 4.346 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/17/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/17/rendered/strips_0.jpg).
- reverse case **19**: 0 restarts, 0.091 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/19/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/19/rendered/strips_0.jpg).
- reverse case **21**: 0 restarts, 8.152 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/21/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/21/rendered/strips_0.jpg).
- reverse case **23**: 1 restarts, 1.479 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/23/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/23/rendered/strips_0.jpg).
- reverse case **25**: 1 restarts, 30.074 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/25/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/25/rendered/strips_0.jpg).
- reverse case **27**: 2 restarts, 25.559 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/27/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/27/rendered/strips_0.jpg).
- reverse case **29**: 4 restarts, 19.817 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/29/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/29/rendered/strips_0.jpg).
- reverse case **31**: 0 restarts, 0.979 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/31/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/31/rendered/strips_0.jpg).

### 6. PHercParis4_eval_fibers / anon_20260815T025206849_000041.json

4 restarts / 61.235 directed mm = 0.065323 restarts/mm.

- forward case **84**: 2 restarts, 30.617 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/84/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/84/rendered/strips_0.jpg).
- reverse case **85**: 2 restarts, 30.617 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/85/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/85/rendered/strips_0.jpg).

### 7. PHerc1229 / lt_20260825T075316065_000069.json

2 restarts / 32.344 directed mm = 0.061835 restarts/mm.

- forward case **76**: 1 restarts, 16.172 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/76/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/76/rendered/strips_0.jpg).
- reverse case **77**: 1 restarts, 16.172 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/77/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/77/rendered/strips_0.jpg).

### 8. PHerc0139 / kb_20260901T224443942_000068.json

12 restarts / 241.540 directed mm = 0.049681 restarts/mm.

- forward case **8**: 5 restarts, 72.967 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/8/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/8/rendered/strips_0.jpg).
- forward case **10**: 0 restarts, 23.370 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/10/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/10/rendered/strips_0.jpg).
- forward case **12**: 0 restarts, 16.808 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/12/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/12/rendered/strips_0.jpg).
- forward case **14**: 0 restarts, 7.626 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/14/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/14/rendered/strips_0.jpg).
- reverse case **9**: 6 restarts, 72.967 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/9/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/9/rendered/strips_0.jpg).
- reverse case **11**: 1 restarts, 23.370 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/11/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/11/rendered/strips_0.jpg).
- reverse case **13**: 0 restarts, 16.808 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/13/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/13/rendered/strips_0.jpg).
- reverse case **15**: 0 restarts, 7.626 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/15/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/15/rendered/strips_0.jpg).

### 9. PHerc0139 / kb_20260827T204421830_000022.json

2 restarts / 44.835 directed mm = 0.044608 restarts/mm.

- forward case **0**: 1 restarts, 21.942 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/0/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/0/rendered/strips_0.jpg).
- forward case **2**: 0 restarts, 0.476 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/2/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/2/rendered/strips_0.jpg).
- reverse case **1**: 1 restarts, 21.942 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/1/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/1/rendered/strips_0.jpg).
- reverse case **3**: 0 restarts, 0.476 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/3/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/3/rendered/strips_0.jpg).

### 10. PHerc0139 / lt_20260909T084022344_000109.json

1 restarts / 25.492 directed mm = 0.039227 restarts/mm.

- forward case **58**: 0 restarts, 1.890 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/58/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/58/rendered/strips_0.jpg).
- forward case **60**: 1 restarts, 10.856 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/60/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/60/rendered/strips_0.jpg).
- reverse case **59**: 0 restarts, 1.890 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/59/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/59/rendered/strips_0.jpg).
- reverse case **61**: 0 restarts, 10.856 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/61/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/61/rendered/strips_0.jpg).

### 11. PHercParis4_eval_fibers / anon_20260905T163237868_000469.json

7 restarts / 184.137 directed mm = 0.038015 restarts/mm.

- forward case **146**: 0 restarts, 10.305 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/146/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/146/rendered/strips_0.jpg).
- forward case **148**: 0 restarts, 0.442 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/148/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/148/rendered/strips_0.jpg).
- forward case **150**: 1 restarts, 7.647 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/150/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/150/rendered/strips_0.jpg).
- forward case **152**: 0 restarts, 0.604 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/152/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/152/rendered/strips_0.jpg).
- forward case **154**: 0 restarts, 1.276 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/154/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/154/rendered/strips_0.jpg).
- forward case **156**: 3 restarts, 71.794 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/156/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/156/rendered/strips_0.jpg).
- reverse case **147**: 1 restarts, 10.305 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/147/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/147/rendered/strips_0.jpg).
- reverse case **149**: 0 restarts, 0.442 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/149/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/149/rendered/strips_0.jpg).
- reverse case **151**: 0 restarts, 7.647 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/151/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/151/rendered/strips_0.jpg).
- reverse case **153**: 0 restarts, 0.604 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/153/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/153/rendered/strips_0.jpg).
- reverse case **155**: 0 restarts, 1.276 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/155/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/155/rendered/strips_0.jpg).
- reverse case **157**: 2 restarts, 71.794 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/157/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/157/rendered/strips_0.jpg).

### 12. PHercParis4_eval_fibers / anon_20260902T214326941_000399.json

4 restarts / 108.010 directed mm = 0.037034 restarts/mm.

- forward case **130**: 2 restarts, 54.005 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/130/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/130/rendered/strips_0.jpg).
- reverse case **131**: 2 restarts, 54.005 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/131/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/131/rendered/strips_0.jpg).

### 13. PHerc1229 / lt_20260824T091041867_000056.json

5 restarts / 142.928 directed mm = 0.034983 restarts/mm.

- forward case **74**: 3 restarts, 71.464 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/74/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/74/rendered/strips_0.jpg).
- reverse case **75**: 2 restarts, 71.464 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/75/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/75/rendered/strips_0.jpg).

### 14. PHerc0139 / lt_20260909T082646148_000107.json

1 restarts / 33.137 directed mm = 0.030178 restarts/mm.

- forward case **56**: 1 restarts, 16.568 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/56/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/56/rendered/strips_0.jpg).
- reverse case **57**: 0 restarts, 16.568 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/57/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/57/rendered/strips_0.jpg).

### 15. PHerc0139 / kb_20260908T172232906_000117.json

1 restarts / 37.771 directed mm = 0.026476 restarts/mm.

- forward case **32**: 0 restarts, 15.419 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/32/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/32/rendered/strips_0.jpg).
- forward case **34**: 0 restarts, 1.070 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/34/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/34/rendered/strips_0.jpg).
- forward case **36**: 0 restarts, 0.208 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/36/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/36/rendered/strips_0.jpg).
- forward case **38**: 0 restarts, 0.748 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/38/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/38/rendered/strips_0.jpg).
- forward case **40**: 0 restarts, 1.441 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/40/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/40/rendered/strips_0.jpg).
- reverse case **33**: 0 restarts, 15.419 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/33/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/33/rendered/strips_0.jpg).
- reverse case **35**: 0 restarts, 1.070 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/35/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/35/rendered/strips_0.jpg).
- reverse case **37**: 1 restarts, 0.208 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/37/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/37/rendered/strips_0.jpg).
- reverse case **39**: 0 restarts, 0.748 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/39/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/39/rendered/strips_0.jpg).
- reverse case **41**: 0 restarts, 1.441 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/41/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/41/rendered/strips_0.jpg).

### 16. PHerc0139 / lt_20260904T062829219_000052.json

5 restarts / 273.603 directed mm = 0.018275 restarts/mm.

- forward case **46**: 0 restarts, 2.160 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/46/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/46/rendered/strips_0.jpg).
- forward case **48**: 2 restarts, 16.854 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/48/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/48/rendered/strips_0.jpg).
- forward case **50**: 0 restarts, 15.723 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/50/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/50/rendered/strips_0.jpg).
- forward case **52**: 1 restarts, 102.064 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/52/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/52/rendered/strips_0.jpg).
- reverse case **47**: 0 restarts, 2.160 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/47/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/47/rendered/strips_0.jpg).
- reverse case **49**: 1 restarts, 16.854 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/49/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/49/rendered/strips_0.jpg).
- reverse case **51**: 1 restarts, 15.723 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/51/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/51/rendered/strips_0.jpg).
- reverse case **53**: 0 restarts, 102.064 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/53/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/53/rendered/strips_0.jpg).

### 17. PHerc0139 / kb_20260828T155910844_000043.json

4 restarts / 263.807 directed mm = 0.015163 restarts/mm.

- forward case **4**: 2 restarts, 128.712 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/4/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/4/rendered/strips_0.jpg).
- forward case **6**: 0 restarts, 3.191 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/6/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/6/rendered/strips_0.jpg).
- reverse case **5**: 2 restarts, 128.712 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/5/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/5/rendered/strips_0.jpg).
- reverse case **7**: 0 restarts, 3.191 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/7/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/7/rendered/strips_0.jpg).

### 18. PHercParis4_eval_fibers / anon_20260821T221016194_000135.json

3 restarts / 211.748 directed mm = 0.014168 restarts/mm.

- forward case **90**: 2 restarts, 105.874 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/90/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/90/rendered/strips_0.jpg).
- reverse case **91**: 1 restarts, 105.874 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/91/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/91/rendered/strips_0.jpg).

### 19. PHercParis4_fibers / kb_20260711T043119415_000150.json

3 restarts / 220.246 directed mm = 0.013621 restarts/mm.

- forward case **164**: 2 restarts, 110.123 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/164/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/164/rendered/strips_0.jpg).
- reverse case **165**: 1 restarts, 110.123 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/165/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/165/rendered/strips_0.jpg).

### 20. PHercParis4_eval_fibers / anon_20260907T141106877_000497.json

2 restarts / 153.378 directed mm = 0.013040 restarts/mm.

- forward case **158**: 0 restarts, 0.271 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/158/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/158/rendered/strips_0.jpg).
- forward case **160**: 2 restarts, 74.120 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/160/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/160/rendered/strips_0.jpg).
- forward case **162**: 0 restarts, 2.298 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/162/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/162/rendered/strips_0.jpg).
- reverse case **159**: 0 restarts, 0.271 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/159/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/159/rendered/strips_0.jpg).
- reverse case **161**: 0 restarts, 74.120 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/161/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/161/rendered/strips_0.jpg).
- reverse case **163**: 0 restarts, 2.298 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/163/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/163/rendered/strips_0.jpg).

### 21. PHercParis4_eval_fibers / anon_20260902T181618881_000388.json

3 restarts / 240.093 directed mm = 0.012495 restarts/mm.

- forward case **128**: 2 restarts, 120.047 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/128/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/128/rendered/strips_0.jpg).
- reverse case **129**: 1 restarts, 120.047 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/129/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/129/rendered/strips_0.jpg).

### 22. PHerc0332 / dj_20260810T113807458_000001.json

4 restarts / 360.927 directed mm = 0.011083 restarts/mm.

- forward case **62**: 3 restarts, 180.463 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/62/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/62/rendered/strips_0.jpg).
- reverse case **63**: 1 restarts, 180.463 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/63/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/63/rendered/strips_0.jpg).

### 23. PHercParis4_eval_fibers / anon_20260902T125309620_000368.json

1 restarts / 96.116 directed mm = 0.010404 restarts/mm.

- forward case **120**: 0 restarts, 4.800 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/120/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/120/rendered/strips_0.jpg).
- forward case **122**: 1 restarts, 43.259 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/122/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/122/rendered/strips_0.jpg).
- reverse case **121**: 0 restarts, 4.800 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/121/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/121/rendered/strips_0.jpg).
- reverse case **123**: 0 restarts, 43.259 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/123/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/123/rendered/strips_0.jpg).

### 24. PHercParis4_eval_fibers / anon_20260826T172849645_000255.json

1 restarts / 178.078 directed mm = 0.005616 restarts/mm.

- forward case **102**: 1 restarts, 89.039 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/102/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/102/rendered/strips_0.jpg).
- reverse case **103**: 0 restarts, 89.039 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/103/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/103/rendered/strips_0.jpg).

### 25. PHercParis4_fibers / kb_20260803T040828806_000224.json

1 restarts / 235.373 directed mm = 0.004249 restarts/mm.

- forward case **174**: 1 restarts, 117.687 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/174/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/174/rendered/strips_0.jpg).
- reverse case **175**: 0 restarts, 117.687 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/175/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/175/rendered/strips_0.jpg).

### 26. PHerc0139 / lt_20260909T071741261_000094.json

0 restarts / 4.102 directed mm = 0.000000 restarts/mm.

- forward case **54**: 0 restarts, 2.051 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/54/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/54/rendered/strips_0.jpg).
- reverse case **55**: 0 restarts, 2.051 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/55/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/55/rendered/strips_0.jpg).

### 27. PHerc0332 / kb_20260822T033517433_000058.json

0 restarts / 30.679 directed mm = 0.000000 restarts/mm.

- forward case **64**: 0 restarts, 15.340 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/64/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/64/rendered/strips_0.jpg).
- reverse case **65**: 0 restarts, 15.340 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/65/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/65/rendered/strips_0.jpg).

### 28. PHerc0332 / kb_20260822T164832186_000062.json

0 restarts / 33.555 directed mm = 0.000000 restarts/mm.

- forward case **66**: 0 restarts, 16.777 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/66/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/66/rendered/strips_0.jpg).
- reverse case **67**: 0 restarts, 16.777 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/67/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/67/rendered/strips_0.jpg).

### 29. PHerc1229 / lt_20260819T050805459_000002.json

0 restarts / 80.825 directed mm = 0.000000 restarts/mm.

- forward case **68**: 0 restarts, 40.413 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/68/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/68/rendered/strips_0.jpg).
- reverse case **69**: 0 restarts, 40.413 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/69/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/69/rendered/strips_0.jpg).

### 30. PHerc1229 / lt_20260819T081412107_000011.json

0 restarts / 72.265 directed mm = 0.000000 restarts/mm.

- forward case **70**: 0 restarts, 36.132 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/70/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/70/rendered/strips_0.jpg).
- reverse case **71**: 0 restarts, 36.132 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/71/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/71/rendered/strips_0.jpg).

### 31. PHerc1229 / lt_20260824T083243313_000053.json

0 restarts / 78.743 directed mm = 0.000000 restarts/mm.

- forward case **72**: 0 restarts, 39.372 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/72/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/72/rendered/strips_0.jpg).
- reverse case **73**: 0 restarts, 39.372 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/73/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/73/rendered/strips_0.jpg).

### 32. PHerc1451 / dj_20260903T195009387_000005.json

0 restarts / 26.112 directed mm = 0.000000 restarts/mm.

- forward case **78**: 0 restarts, 13.056 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/78/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/78/rendered/strips_0.jpg).
- reverse case **79**: 0 restarts, 13.056 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/79/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/79/rendered/strips_0.jpg).

### 33. PHerc1451 / dj_20260903T200017539_000006.json

0 restarts / 22.258 directed mm = 0.000000 restarts/mm.

- forward case **80**: 0 restarts, 11.129 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/80/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/80/rendered/strips_0.jpg).
- reverse case **81**: 0 restarts, 11.129 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/81/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/81/rendered/strips_0.jpg).

### 34. PHercParis4_eval_fibers / anon_20260815T020414697_000008.json

0 restarts / 49.090 directed mm = 0.000000 restarts/mm.

- forward case **82**: 0 restarts, 24.545 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/82/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/82/rendered/strips_0.jpg).
- reverse case **83**: 0 restarts, 24.545 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/83/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/83/rendered/strips_0.jpg).

### 35. PHercParis4_eval_fibers / anon_20260815T135300158_000075.json

0 restarts / 42.166 directed mm = 0.000000 restarts/mm.

- forward case **86**: 0 restarts, 21.083 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/86/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/86/rendered/strips_0.jpg).
- reverse case **87**: 0 restarts, 21.083 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/87/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/87/rendered/strips_0.jpg).

### 36. PHercParis4_eval_fibers / anon_20260816T224409952_000122.json

0 restarts / 65.095 directed mm = 0.000000 restarts/mm.

- forward case **88**: 0 restarts, 32.548 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/88/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/88/rendered/strips_0.jpg).
- reverse case **89**: 0 restarts, 32.548 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/89/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/89/rendered/strips_0.jpg).

### 37. PHercParis4_eval_fibers / anon_20260822T103908823_000165.json

0 restarts / 224.740 directed mm = 0.000000 restarts/mm.

- forward case **92**: 0 restarts, 112.370 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/92/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/92/rendered/strips_0.jpg).
- reverse case **93**: 0 restarts, 112.370 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/93/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/93/rendered/strips_0.jpg).

### 38. PHercParis4_eval_fibers / anon_20260822T121633375_000173.json

0 restarts / 263.041 directed mm = 0.000000 restarts/mm.

- forward case **94**: 0 restarts, 9.337 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/94/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/94/rendered/strips_0.jpg).
- forward case **96**: 0 restarts, 56.776 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/96/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/96/rendered/strips_0.jpg).
- forward case **98**: 0 restarts, 65.407 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/98/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/98/rendered/strips_0.jpg).
- reverse case **95**: 0 restarts, 9.337 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/95/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/95/rendered/strips_0.jpg).
- reverse case **97**: 0 restarts, 56.776 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/97/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/97/rendered/strips_0.jpg).
- reverse case **99**: 0 restarts, 65.407 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/99/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/99/rendered/strips_0.jpg).

### 39. PHercParis4_eval_fibers / anon_20260824T032219305_000197.json

0 restarts / 81.787 directed mm = 0.000000 restarts/mm.

- forward case **100**: 0 restarts, 40.893 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/100/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/100/rendered/strips_0.jpg).
- reverse case **101**: 0 restarts, 40.893 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/101/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/101/rendered/strips_0.jpg).

### 40. PHercParis4_eval_fibers / anon_20260901T165827570_000316.json

0 restarts / 128.956 directed mm = 0.000000 restarts/mm.

- forward case **108**: 0 restarts, 64.478 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/108/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/108/rendered/strips_0.jpg).
- reverse case **109**: 0 restarts, 64.478 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/109/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/109/rendered/strips_0.jpg).

### 41. PHercParis4_eval_fibers / anon_20260901T170547855_000317.json

0 restarts / 171.571 directed mm = 0.000000 restarts/mm.

- forward case **110**: 0 restarts, 51.417 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/110/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/110/rendered/strips_0.jpg).
- forward case **112**: 0 restarts, 34.368 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/112/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/112/rendered/strips_0.jpg).
- reverse case **111**: 0 restarts, 51.417 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/111/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/111/rendered/strips_0.jpg).
- reverse case **113**: 0 restarts, 34.368 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/113/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/113/rendered/strips_0.jpg).

### 42. PHercParis4_eval_fibers / anon_20260901T201915394_000335.json

0 restarts / 35.826 directed mm = 0.000000 restarts/mm.

- forward case **114**: 0 restarts, 17.716 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/114/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/114/rendered/strips_0.jpg).
- forward case **116**: 0 restarts, 0.197 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/116/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/116/rendered/strips_0.jpg).
- reverse case **115**: 0 restarts, 17.716 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/115/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/115/rendered/strips_0.jpg).
- reverse case **117**: 0 restarts, 0.197 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/117/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/117/rendered/strips_0.jpg).

### 43. PHercParis4_eval_fibers / anon_20260902T103444019_000352.json

0 restarts / 148.276 directed mm = 0.000000 restarts/mm.

- forward case **118**: 0 restarts, 74.138 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/118/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/118/rendered/strips_0.jpg).
- reverse case **119**: 0 restarts, 74.138 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/119/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/119/rendered/strips_0.jpg).

### 44. PHercParis4_eval_fibers / anon_20260902T131435457_000370.json

0 restarts / 145.611 directed mm = 0.000000 restarts/mm.

- forward case **124**: 0 restarts, 24.515 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/124/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/124/rendered/strips_0.jpg).
- forward case **126**: 0 restarts, 48.290 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/126/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/126/rendered/strips_0.jpg).
- reverse case **125**: 0 restarts, 24.515 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/125/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/125/rendered/strips_0.jpg).
- reverse case **127**: 0 restarts, 48.290 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/127/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/127/rendered/strips_0.jpg).

### 45. PHercParis4_eval_fibers / anon_20260902T215234337_000400.json

0 restarts / 261.841 directed mm = 0.000000 restarts/mm.

- forward case **132**: 0 restarts, 130.920 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/132/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/132/rendered/strips_0.jpg).
- reverse case **133**: 0 restarts, 130.920 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/133/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/133/rendered/strips_0.jpg).

### 46. PHercParis4_eval_fibers / anon_20260903T214715058_000404.json

0 restarts / 345.753 directed mm = 0.000000 restarts/mm.

- forward case **134**: 0 restarts, 114.984 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/134/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/134/rendered/strips_0.jpg).
- forward case **136**: 0 restarts, 57.893 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/136/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/136/rendered/strips_0.jpg).
- reverse case **135**: 0 restarts, 114.984 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/135/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/135/rendered/strips_0.jpg).
- reverse case **137**: 0 restarts, 57.893 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/137/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/137/rendered/strips_0.jpg).

### 47. PHercParis4_eval_fibers / anon_20260904T145959056_000421.json

0 restarts / 262.868 directed mm = 0.000000 restarts/mm.

- forward case **138**: 0 restarts, 19.717 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/138/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/138/rendered/strips_0.jpg).
- forward case **140**: 0 restarts, 1.955 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/140/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/140/rendered/strips_0.jpg).
- forward case **142**: 0 restarts, 109.763 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/142/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/142/rendered/strips_0.jpg).
- reverse case **139**: 0 restarts, 19.717 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/139/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/139/rendered/strips_0.jpg).
- reverse case **141**: 0 restarts, 1.955 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/141/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/141/rendered/strips_0.jpg).
- reverse case **143**: 0 restarts, 109.763 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/143/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/143/rendered/strips_0.jpg).

### 48. PHercParis4_eval_fibers / anon_20260904T210541536_000446.json

0 restarts / 175.857 directed mm = 0.000000 restarts/mm.

- forward case **144**: 0 restarts, 87.928 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/144/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/144/rendered/strips_0.jpg).
- reverse case **145**: 0 restarts, 87.928 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/145/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/145/rendered/strips_0.jpg).

### 49. PHercParis4_fibers / sm_20260706T204346695_000119.json

0 restarts / 290.724 directed mm = 0.000000 restarts/mm.

- forward case **176**: 0 restarts, 145.362 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/176/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/176/rendered/strips_0.jpg).
- reverse case **177**: 0 restarts, 145.362 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/177/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/177/rendered/strips_0.jpg).

### 50. PHercParis4_fibers / sm_20260707T184533000_000122.json

0 restarts / 174.961 directed mm = 0.000000 restarts/mm.

- forward case **178**: 0 restarts, 87.481 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/178/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/178/rendered/strips_0.jpg).
- reverse case **179**: 0 restarts, 87.481 mm; [trace metadata](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/179/trace.json); [strip 0](/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/cases/179/rendered/strips_0.jpg).


