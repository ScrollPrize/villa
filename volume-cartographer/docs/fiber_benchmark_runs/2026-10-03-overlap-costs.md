# Zero-overlap cost tuning

Frozen raw40500 iso20 beam8 1024 crop, all twelve 0% overlap selections. Cost-only, at most two coordinate-search rounds per selection; not exhaustive. Rank: non-oracle accuracy, observed shift, then reference-conditioned removal. Min256, median±64, fixed scale .822/phase .5,500 messages unchanged. Four concurrent candidates,128 native threads each. No inference or retracing.

Status: complete. Trials: 576.

| Variant | Status | Before accuracy | Before shift | Before removal | Tuned accuracy | Tuned shift | Tuned removal |
|---|---|---:|---:|---:|---:|---:|---:|
| No-dedup baseline | historical matched control | 61.5385% | 87.5000% | -28.9207% | 61.5385% | 87.5000% | -28.9207% |
| length-10x20-f0 | complete | 30.7692% | 83.3333% | -78.4643% | 53.8462% | 87.5000% | -59.7769% |
| cost-10x20-f0 | complete | 30.7692% | 83.3333% | -193.9482% | 65.3846% | 91.6667% | -55.6864% |
| presence-10x20-f0 | complete | 30.7692% | 75.0000% | -202.9867% | 65.3846% | 91.6667% | -55.1216% |
| length-20x40-f0 | complete | 50.0000% | 79.1667% | -82.9448% | 53.8462% | 95.8333% | -62.4165% |
| cost-20x40-f0 | complete | 34.6154% | 50.0000% | -187.7869% | 61.5385% | 91.6667% | -58.0819% |
| presence-20x40-f0 | complete | 42.3077% | 70.8333% | -148.4243% | 61.5385% | 87.5000% | -60.1005% |
| length-10x10-f0 | complete | 26.9231% | 79.1667% | -79.3645% | 53.8462% | 87.5000% | -59.6754% |
| cost-10x10-f0 | complete | 30.7692% | 83.3333% | -197.3445% | 65.3846% | 83.3333% | -61.7333% |
| presence-10x10-f0 | complete | 30.7692% | 75.0000% | -204.2827% | 65.3846% | 91.6667% | -60.3234% |
| length-20x20-f0 | complete | 30.7692% | 83.3333% | -76.6113% | 53.8462% | 87.5000% | -59.9068% |
| cost-20x20-f0 | complete | 34.6154% | 75.0000% | -193.0189% | 53.8462% | 95.8333% | -60.6995% |
| presence-20x20-f0 | complete | 30.7692% | 75.0000% | -226.4068% | 53.8462% | 95.8333% | -58.2732% |

## Outcome

All12 selections complete, 576 unique weight configurations, 91 reference-conditioned removal solves plus 12 independent and 12 oracle winner repeats. Failed trials: 0.

Best accuracy-first selection: `presence-10x20-f0`; accuracy 65.3846%, observed shift 91.6667%, removal -55.1216%.

All repeated ladders/states and oracle counts match. This is bounded two-round tuning, not a proof of a global optimum. No held-out crop test or automatic default promotion was performed.

## Selected weights

| Variant | Disable /128 | Boundary | Perp near | Perp far | Parallel same | Parallel one | Sign cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| cost-10x10-f0 | 100 | 10 | 0 | 0.01 | 0 | 0.0375 | 11 |
| cost-10x20-f0 | 100 | 200 | 0 | 0.2 | 0 | 0.0375 | hard |
| cost-20x20-f0 | 100 | 200 | 0 | 0.01 | 0 | 0.0375 | hard |
| cost-20x40-f0 | 100 | 200 | 0 | 0.05 | 0 | 0.0375 | hard |
| length-10x10-f0 | 25 | 10 | 0 | 0.2 | 0 | 0.0375 | hard |
| length-10x20-f0 | 25 | 10 | 0 | 0.2 | 0 | 0.0375 | hard |
| length-20x20-f0 | 25 | 10 | 0 | 0.2 | 0 | 0.0375 | hard |
| length-20x40-f0 | 100 | 200 | 0 | 0.2 | 0 | 0.0375 | hard |
| presence-10x10-f0 | 100 | 50 | 0 | 0.01 | 0 | 0.0375 | hard |
| presence-10x20-f0 | 100 | 10 | 0.2 | 0.2 | 0 | 0.0375 | hard |
| presence-20x20-f0 | 400 | 10 | 0 | 0.01 | 0 | 0.0375 | hard |
| presence-20x40-f0 | 100 | 10 | 0 | 0.01 | 0.2 | 0.0375 | hard |

## Winner coverage/readout diagnostics

| Variant | Correct /26 | Missing references | Shift references | Reference-conditioned accuracy |
|---|---:|---|---:|---:|
| cost-10x10-f0 | 17 | ['4', '12'] | 24 | 84.6154% |
| cost-10x20-f0 | 17 | ['4', '12'] | 24 | 80.7692% |
| cost-20x20-f0 | 14 | ['4', '12'] | 24 | 88.4615% |
| cost-20x40-f0 | 16 | ['4', '12'] | 24 | 88.4615% |
| length-10x10-f0 | 14 | ['4', '12'] | 24 | 84.6154% |
| length-10x20-f0 | 14 | ['4', '12'] | 24 | 84.6154% |
| length-20x20-f0 | 14 | ['4', '12'] | 24 | 84.6154% |
| length-20x40-f0 | 14 | ['4', '12'] | 24 | 88.4615% |
| presence-10x10-f0 | 17 | ['4', '12'] | 24 | 88.4615% |
| presence-10x20-f0 | 17 | ['4', '12'] | 24 | 88.4615% |
| presence-20x20-f0 | 14 | ['4', '12'] | 24 | 88.4615% |
| presence-20x40-f0 | 16 | ['4', '12'] | 24 | 88.4615% |

## Protocol and reproducibility

Grids: `{"boundary": [0, 10, 50, 200], "disable": [0, 25, 50, 100, 200, 400], "far": [0, 0.01, 0.05, 0.2], "near": [0, 0.01, 0.05, 0.2], "p0": [0, 0.01, 0.05, 0.2], "p1": [0, 0.005, 0.015, 0.0375, 0.1, 0.3], "sign": ["11", "44", "176", "hard"]}`. The second round combines changes by searching around the first winner. A first-round plateau stops that selection. Inactive P2 stays0 (parallel cutoff1.5); continuity remains hard. Finite sign penalties permit sign violations; hard does not. Zero magnitude does not remove the separately configured sign term. Removal is evaluated only for tied accuracy/shift finalists plus controls/winners; other removal scores are not run, not zero. Failed trials/gauge failures stay explicit in JSON. All baselines are rerun and checked against the saved untuned ladder; winners repeated. Every valid independent/oracle pair must have identical initial states and factors.

Accuracy denominator26; observed shift excludes missing references. Removal is −100×problematic/retained fulfilled (higher is better), not an accuracy. These are tuning-set results, not held-out validation. No production default changes. Retained trace geometry is fixed per variant; artifact replay not measured.

Artifacts/configs/logs/provenance: `/ephemeral/hendrik/fiber_crop_bench/20261003_iso20_overlap_costs_v1`. Native RelWithDebInfo. Run: `bash /home/hendrik/villa3/scripts/run_training_failure_mining`, control `{"action":"iso20_overlap_costs"}`; add `"report_only":true` to refresh. One solve per configuration plus final repeats; concurrent timings are not latency comparisons.

## All trials

| Variant/trial | Weights | Accuracy | Shift | Removal | Error |
|---|---|---:|---:|---:|---|
| cost-10x10-f0-03f7c52d8f68 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "44"}` | 30.7692% | 66.6667% | unavailable |  |
| cost-10x10-f0-040de5cd74e6 | `{"boundary": 10, "disable": 400, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 83.3333% | unavailable |  |
| cost-10x10-f0-0a6b95daf499 | `{"boundary": 50, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 79.1667% | unavailable |  |
| cost-10x10-f0-0ad0de4b6497 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "176"}` | 53.8462% | 87.5000% | unavailable |  |
| cost-10x10-f0-0ee89d1c3af5 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.05, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 34.6154% | 75.0000% | unavailable |  |
| cost-10x10-f0-13f56f9f9451 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.3, "sign": "hard"}` | 34.6154% | 83.3333% | unavailable |  |
| cost-10x10-f0-22d41ac26bc8 | `{"boundary": 10, "disable": 25, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 58.3333% | unavailable |  |
| cost-10x10-f0-299b306b82c3 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | -197.3445% |  |
| cost-10x10-f0-2cb0bf26f547 | `{"boundary": 10, "disable": 50, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| cost-10x10-f0-34e128eb1a83 | `{"boundary": 10, "disable": 400, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| cost-10x10-f0-359901695caa | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.2, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| cost-10x10-f0-3b81f34c0aff | `{"boundary": 50, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 95.8333% | unavailable |  |
| cost-10x10-f0-41449ebff635 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0.2, "p1": 0.0375, "sign": "hard"}` | 34.6154% | 70.8333% | unavailable |  |
| cost-10x10-f0-41bbd5f7d429 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "11"}` | 65.3846% | 83.3333% | -61.7333% |  |
| cost-10x10-f0-4c75248e58bd | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 66.6667% | unavailable |  |
| cost-10x10-f0-4caf2402642a | `{"boundary": 10, "disable": 0, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 29.1667% | unavailable |  |
| cost-10x10-f0-50792d92f305 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0, "sign": "hard"}` | 34.6154% | 62.5000% | unavailable |  |
| cost-10x10-f0-5f361d1ab0a4 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.05, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| cost-10x10-f0-6ea251683cf9 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "11"}` | 30.7692% | 75.0000% | unavailable |  |
| cost-10x10-f0-6f3c4ea3c7dc | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.01, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| cost-10x10-f0-72e2c7f38138 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 42.3077% | 83.3333% | unavailable |  |
| cost-10x10-f0-73ed196882d2 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "176"}` | 30.7692% | 75.0000% | unavailable |  |
| cost-10x10-f0-7cd8ca626ef7 | `{"boundary": 10, "disable": 200, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| cost-10x10-f0-80f5928f563d | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.005, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| cost-10x10-f0-95fa6941c724 | `{"boundary": 0, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 70.8333% | unavailable |  |
| cost-10x10-f0-a0662bafff90 | `{"boundary": 0, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| cost-10x10-f0-a218e0647bb9 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "44"}` | 53.8462% | 87.5000% | unavailable |  |
| cost-10x10-f0-a23b9b95fc05 | `{"boundary": 200, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 95.8333% | unavailable |  |
| cost-10x10-f0-a5214c123bfd | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 87.5000% | -62.3436% |  |
| cost-10x10-f0-ac0a6ac831d8 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.005, "sign": "hard"}` | 30.7692% | 41.6667% | unavailable |  |
| cost-10x10-f0-b43effe3a7db | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0.01, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| cost-10x10-f0-b7948ffa58e5 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.1, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| cost-10x10-f0-b9052bd47e1c | `{"boundary": 10, "disable": 0, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 42.3077% | 45.8333% | unavailable |  |
| cost-10x10-f0-bd48d88a0fb1 | `{"boundary": 10, "disable": 50, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 87.5000% | unavailable |  |
| cost-10x10-f0-cadf91c60125 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.01, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 83.3333% | unavailable |  |
| cost-10x10-f0-cdb6627a0ac2 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.3, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| cost-10x10-f0-d58de4bef07f | `{"boundary": 10, "disable": 200, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 83.3333% | unavailable |  |
| cost-10x10-f0-d76114838bf5 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.1, "sign": "hard"}` | 30.7692% | 87.5000% | unavailable |  |
| cost-10x10-f0-d819a81f7f6b | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0, "sign": "hard"}` | 15.3846% | 33.3333% | unavailable |  |
| cost-10x10-f0-daeacfc423ad | `{"boundary": 200, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 79.1667% | unavailable |  |
| cost-10x10-f0-dbe5aa3e4db7 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0.05, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| cost-10x10-f0-dd3e5e02fb78 | `{"boundary": 10, "disable": 100, "far": 0.05, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 87.5000% | unavailable |  |
| cost-10x10-f0-e37ee84b7d76 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0.01, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 62.5000% | unavailable |  |
| cost-10x10-f0-e52cf3900b82 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0.05, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| cost-10x10-f0-eab4515c34ee | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.015, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| cost-10x10-f0-eb71f109217c | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.015, "sign": "hard"}` | 30.7692% | 41.6667% | unavailable |  |
| cost-10x10-f0-ec964c636b9d | `{"boundary": 10, "disable": 25, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| cost-10x10-f0-ff06f81e2408 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| cost-10x20-f0-03f7c52d8f68 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "44"}` | 30.7692% | 66.6667% | unavailable |  |
| cost-10x20-f0-0a6b95daf499 | `{"boundary": 50, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 79.1667% | unavailable |  |
| cost-10x20-f0-0bc01d547270 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0.01, "p1": 0.0375, "sign": "hard"}` | 61.5385% | 91.6667% | unavailable |  |
| cost-10x20-f0-0ee89d1c3af5 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.05, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 34.6154% | 75.0000% | unavailable |  |
| cost-10x20-f0-1f24c775c606 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "11"}` | 61.5385% | 87.5000% | unavailable |  |
| cost-10x20-f0-299b306b82c3 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | -193.9482% |  |
| cost-10x20-f0-2cb0bf26f547 | `{"boundary": 10, "disable": 50, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| cost-10x20-f0-2d33708bd653 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 38.4615% | 79.1667% | unavailable |  |
| cost-10x20-f0-335c469efb3a | `{"boundary": 10, "disable": 50, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 61.5385% | 91.6667% | unavailable |  |
| cost-10x20-f0-34e128eb1a83 | `{"boundary": 10, "disable": 400, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| cost-10x20-f0-359901695caa | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.2, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 75.0000% | unavailable |  |
| cost-10x20-f0-375610631f4a | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "176"}` | 65.3846% | 91.6667% | -56.5631% |  |
| cost-10x20-f0-456f413ec5eb | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0.01, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 61.5385% | 91.6667% | unavailable |  |
| cost-10x20-f0-4c75248e58bd | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 66.6667% | unavailable |  |
| cost-10x20-f0-4caf2402642a | `{"boundary": 10, "disable": 0, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 29.1667% | unavailable |  |
| cost-10x20-f0-5f361d1ab0a4 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.05, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| cost-10x20-f0-619ed0f70866 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.1, "sign": "hard"}` | 57.6923% | 75.0000% | unavailable |  |
| cost-10x20-f0-6ea251683cf9 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "11"}` | 30.7692% | 75.0000% | unavailable |  |
| cost-10x20-f0-6f3c4ea3c7dc | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.01, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 46.1538% | 70.8333% | unavailable |  |
| cost-10x20-f0-72e2c7f38138 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 61.5385% | 91.6667% | -57.0510% |  |
| cost-10x20-f0-73ed196882d2 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "176"}` | 30.7692% | 75.0000% | unavailable |  |
| cost-10x20-f0-77759fcba1e6 | `{"boundary": 10, "disable": 400, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 61.5385% | 91.6667% | unavailable |  |
| cost-10x20-f0-78c2dbc42acc | `{"boundary": 0, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 61.5385% | 91.6667% | unavailable |  |
| cost-10x20-f0-7b2e41cbddca | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.3, "sign": "hard"}` | 38.4615% | 41.6667% | unavailable |  |
| cost-10x20-f0-7cd8ca626ef7 | `{"boundary": 10, "disable": 200, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| cost-10x20-f0-7eaadd24b08e | `{"boundary": 200, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 65.3846% | 91.6667% | -55.6864% |  |
| cost-10x20-f0-80f5928f563d | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.005, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| cost-10x20-f0-8d1512fb61b5 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0.2, "p1": 0.0375, "sign": "hard"}` | 46.1538% | 45.8333% | unavailable |  |
| cost-10x20-f0-939a7482b782 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0.05, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 66.6667% | unavailable |  |
| cost-10x20-f0-95c965e7b102 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.015, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| cost-10x20-f0-a0662bafff90 | `{"boundary": 0, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| cost-10x20-f0-a5214c123bfd | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 95.8333% | unavailable |  |
| cost-10x20-f0-b248db304b4a | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0.05, "p1": 0.0375, "sign": "hard"}` | 61.5385% | 91.6667% | unavailable |  |
| cost-10x20-f0-b40d0e49f0c3 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "44"}` | 61.5385% | 87.5000% | unavailable |  |
| cost-10x20-f0-b7948ffa58e5 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.1, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| cost-10x20-f0-ca33a9afbf3a | `{"boundary": 10, "disable": 0, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 38.4615% | 25.0000% | unavailable |  |
| cost-10x20-f0-cadf91c60125 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.01, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 83.3333% | unavailable |  |
| cost-10x20-f0-cdb6627a0ac2 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.3, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| cost-10x20-f0-d72b7873d55a | `{"boundary": 10, "disable": 200, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 61.5385% | 91.6667% | unavailable |  |
| cost-10x20-f0-d819a81f7f6b | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0, "sign": "hard"}` | 15.3846% | 37.5000% | unavailable |  |
| cost-10x20-f0-d91fd55263a2 | `{"boundary": 10, "disable": 25, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 61.5385% | 91.6667% | unavailable |  |
| cost-10x20-f0-daeacfc423ad | `{"boundary": 200, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 79.1667% | unavailable |  |
| cost-10x20-f0-dd3e5e02fb78 | `{"boundary": 10, "disable": 100, "far": 0.05, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 87.5000% | unavailable |  |
| cost-10x20-f0-e14400259eb4 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.005, "sign": "hard"}` | 26.9231% | 62.5000% | unavailable |  |
| cost-10x20-f0-e5eb4062f846 | `{"boundary": 50, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 61.5385% | 91.6667% | unavailable |  |
| cost-10x20-f0-eab4515c34ee | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.015, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| cost-10x20-f0-ec964c636b9d | `{"boundary": 10, "disable": 25, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| cost-10x20-f0-f4a63df9dca4 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0, "sign": "hard"}` | 26.9231% | 45.8333% | unavailable |  |
| cost-20x20-f0-03f7c52d8f68 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "44"}` | 34.6154% | 62.5000% | unavailable |  |
| cost-20x20-f0-040de5cd74e6 | `{"boundary": 10, "disable": 400, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 95.8333% | unavailable |  |
| cost-20x20-f0-0a6b95daf499 | `{"boundary": 50, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 38.4615% | 75.0000% | unavailable |  |
| cost-20x20-f0-0ad0de4b6497 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "176"}` | 53.8462% | 87.5000% | unavailable |  |
| cost-20x20-f0-0ee89d1c3af5 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.05, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 34.6154% | 75.0000% | unavailable |  |
| cost-20x20-f0-13f56f9f9451 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.3, "sign": "hard"}` | 50.0000% | 70.8333% | unavailable |  |
| cost-20x20-f0-22d41ac26bc8 | `{"boundary": 10, "disable": 25, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 87.5000% | unavailable |  |
| cost-20x20-f0-299b306b82c3 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 34.6154% | 75.0000% | -193.0189% |  |
| cost-20x20-f0-2cb0bf26f547 | `{"boundary": 10, "disable": 50, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 42.3077% | 83.3333% | unavailable |  |
| cost-20x20-f0-34e128eb1a83 | `{"boundary": 10, "disable": 400, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 38.4615% | 75.0000% | unavailable |  |
| cost-20x20-f0-359901695caa | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.2, "p1": 0.0375, "sign": "hard"}` | 34.6154% | 75.0000% | unavailable |  |
| cost-20x20-f0-3b81f34c0aff | `{"boundary": 50, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 95.8333% | unavailable |  |
| cost-20x20-f0-41449ebff635 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0.2, "p1": 0.0375, "sign": "hard"}` | 46.1538% | 79.1667% | unavailable |  |
| cost-20x20-f0-41bbd5f7d429 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "11"}` | 53.8462% | 79.1667% | unavailable |  |
| cost-20x20-f0-4c75248e58bd | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 34.6154% | 58.3333% | unavailable |  |
| cost-20x20-f0-4caf2402642a | `{"boundary": 10, "disable": 0, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 19.2308% | 16.6667% | unavailable |  |
| cost-20x20-f0-50792d92f305 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0, "sign": "hard"}` | 50.0000% | 70.8333% | unavailable |  |
| cost-20x20-f0-5f361d1ab0a4 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.05, "p1": 0.0375, "sign": "hard"}` | 42.3077% | 83.3333% | unavailable |  |
| cost-20x20-f0-6ea251683cf9 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "11"}` | 30.7692% | 75.0000% | unavailable |  |
| cost-20x20-f0-6f3c4ea3c7dc | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.01, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 46.1538% | 70.8333% | unavailable |  |
| cost-20x20-f0-72e2c7f38138 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| cost-20x20-f0-73ed196882d2 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "176"}` | 34.6154% | 75.0000% | unavailable |  |
| cost-20x20-f0-7cd8ca626ef7 | `{"boundary": 10, "disable": 200, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 38.4615% | 75.0000% | unavailable |  |
| cost-20x20-f0-80f5928f563d | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.005, "sign": "hard"}` | 38.4615% | 75.0000% | unavailable |  |
| cost-20x20-f0-95fa6941c724 | `{"boundary": 0, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 95.8333% | unavailable |  |
| cost-20x20-f0-a0662bafff90 | `{"boundary": 0, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 38.4615% | 75.0000% | unavailable |  |
| cost-20x20-f0-a218e0647bb9 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "44"}` | 53.8462% | 87.5000% | unavailable |  |
| cost-20x20-f0-a23b9b95fc05 | `{"boundary": 200, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 95.8333% | -60.6995% |  |
| cost-20x20-f0-a5214c123bfd | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 95.8333% | -60.7660% |  |
| cost-20x20-f0-ac0a6ac831d8 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.005, "sign": "hard"}` | 50.0000% | 62.5000% | unavailable |  |
| cost-20x20-f0-b43effe3a7db | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0.01, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 38.4615% | 83.3333% | unavailable |  |
| cost-20x20-f0-b7948ffa58e5 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.1, "sign": "hard"}` | 38.4615% | 75.0000% | unavailable |  |
| cost-20x20-f0-b9052bd47e1c | `{"boundary": 10, "disable": 0, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 33.3333% | unavailable |  |
| cost-20x20-f0-bd48d88a0fb1 | `{"boundary": 10, "disable": 50, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 95.8333% | unavailable |  |
| cost-20x20-f0-cadf91c60125 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.01, "p1": 0.0375, "sign": "hard"}` | 42.3077% | 83.3333% | unavailable |  |
| cost-20x20-f0-cdb6627a0ac2 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.3, "sign": "hard"}` | 34.6154% | 75.0000% | unavailable |  |
| cost-20x20-f0-d58de4bef07f | `{"boundary": 10, "disable": 200, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 95.8333% | unavailable |  |
| cost-20x20-f0-d76114838bf5 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.1, "sign": "hard"}` | 42.3077% | 87.5000% | unavailable |  |
| cost-20x20-f0-d819a81f7f6b | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0, "sign": "hard"}` | 15.3846% | 37.5000% | unavailable |  |
| cost-20x20-f0-daeacfc423ad | `{"boundary": 200, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 38.4615% | 91.6667% | unavailable |  |
| cost-20x20-f0-dbe5aa3e4db7 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0.05, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 70.8333% | unavailable |  |
| cost-20x20-f0-dd3e5e02fb78 | `{"boundary": 10, "disable": 100, "far": 0.05, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 34.6154% | 70.8333% | unavailable |  |
| cost-20x20-f0-e37ee84b7d76 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0.01, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 95.8333% | unavailable |  |
| cost-20x20-f0-e52cf3900b82 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0.05, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 87.5000% | unavailable |  |
| cost-20x20-f0-eab4515c34ee | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.015, "sign": "hard"}` | 34.6154% | 75.0000% | unavailable |  |
| cost-20x20-f0-eb71f109217c | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.015, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| cost-20x20-f0-ec964c636b9d | `{"boundary": 10, "disable": 25, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 34.6154% | 75.0000% | unavailable |  |
| cost-20x20-f0-ff06f81e2408 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 34.6154% | 83.3333% | unavailable |  |
| cost-20x40-f0-03f7c52d8f68 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "44"}` | 34.6154% | 58.3333% | unavailable |  |
| cost-20x40-f0-0a6b95daf499 | `{"boundary": 50, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 38.4615% | 50.0000% | unavailable |  |
| cost-20x40-f0-0d5bc89476b2 | `{"boundary": 50, "disable": 100, "far": 0.05, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| cost-20x40-f0-0ee89d1c3af5 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.05, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 54.1667% | unavailable |  |
| cost-20x40-f0-1b873e76dde9 | `{"boundary": 10, "disable": 100, "far": 0.05, "near": 0, "p0": 0, "p1": 0.0375, "sign": "11"}` | 57.6923% | 87.5000% | unavailable |  |
| cost-20x40-f0-299b306b82c3 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 34.6154% | 50.0000% | -187.7869% |  |
| cost-20x40-f0-2cb0bf26f547 | `{"boundary": 10, "disable": 50, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 34.6154% | 50.0000% | unavailable |  |
| cost-20x40-f0-2ccae4e600eb | `{"boundary": 0, "disable": 100, "far": 0.05, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 62.5000% | unavailable |  |
| cost-20x40-f0-34e128eb1a83 | `{"boundary": 10, "disable": 400, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 34.6154% | 45.8333% | unavailable |  |
| cost-20x40-f0-359901695caa | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.2, "p1": 0.0375, "sign": "hard"}` | 34.6154% | 75.0000% | unavailable |  |
| cost-20x40-f0-42097bf8b526 | `{"boundary": 10, "disable": 100, "far": 0.05, "near": 0, "p0": 0, "p1": 0.015, "sign": "hard"}` | 50.0000% | 70.8333% | unavailable |  |
| cost-20x40-f0-44ac27b318b5 | `{"boundary": 200, "disable": 100, "far": 0.05, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 61.5385% | 91.6667% | -58.0819% |  |
| cost-20x40-f0-4526f1fdc83b | `{"boundary": 10, "disable": 100, "far": 0.05, "near": 0, "p0": 0.05, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| cost-20x40-f0-4c75248e58bd | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 66.6667% | unavailable |  |
| cost-20x40-f0-4caf2402642a | `{"boundary": 10, "disable": 0, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 19.2308% | 27.2727% | unavailable |  |
| cost-20x40-f0-4e444f500b81 | `{"boundary": 10, "disable": 100, "far": 0.05, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 70.8333% | unavailable |  |
| cost-20x40-f0-58d51de96cad | `{"boundary": 10, "disable": 100, "far": 0.05, "near": 0, "p0": 0.01, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| cost-20x40-f0-5ac326855558 | `{"boundary": 10, "disable": 0, "far": 0.05, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 38.4615% | 39.1304% | unavailable |  |
| cost-20x40-f0-5f361d1ab0a4 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.05, "p1": 0.0375, "sign": "hard"}` | 42.3077% | 62.5000% | unavailable |  |
| cost-20x40-f0-66b9bfca1d5b | `{"boundary": 10, "disable": 100, "far": 0.05, "near": 0, "p0": 0, "p1": 0.3, "sign": "hard"}` | 42.3077% | 83.3333% | unavailable |  |
| cost-20x40-f0-69bd26fe70f2 | `{"boundary": 10, "disable": 100, "far": 0.05, "near": 0.01, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 87.5000% | unavailable |  |
| cost-20x40-f0-6a1141c7d411 | `{"boundary": 10, "disable": 50, "far": 0.05, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 62.5000% | unavailable |  |
| cost-20x40-f0-6ea251683cf9 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "11"}` | 34.6154% | 37.5000% | unavailable |  |
| cost-20x40-f0-6f3c4ea3c7dc | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.01, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 46.1538% | 66.6667% | unavailable |  |
| cost-20x40-f0-72e2c7f38138 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 70.8333% | unavailable |  |
| cost-20x40-f0-73ed196882d2 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "176"}` | 34.6154% | 54.1667% | unavailable |  |
| cost-20x40-f0-7cd8ca626ef7 | `{"boundary": 10, "disable": 200, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 34.6154% | 45.8333% | unavailable |  |
| cost-20x40-f0-80f5928f563d | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.005, "sign": "hard"}` | 38.4615% | 50.0000% | unavailable |  |
| cost-20x40-f0-a0662bafff90 | `{"boundary": 0, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 34.6154% | 45.8333% | unavailable |  |
| cost-20x40-f0-a0db83688224 | `{"boundary": 10, "disable": 100, "far": 0.05, "near": 0, "p0": 0, "p1": 0.005, "sign": "hard"}` | 50.0000% | 70.8333% | unavailable |  |
| cost-20x40-f0-a5214c123bfd | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 26.9231% | 37.5000% | unavailable |  |
| cost-20x40-f0-a5ed3fe4c5d6 | `{"boundary": 10, "disable": 100, "far": 0.05, "near": 0, "p0": 0, "p1": 0.0375, "sign": "44"}` | 53.8462% | 70.8333% | unavailable |  |
| cost-20x40-f0-ab0db2cfaf4c | `{"boundary": 10, "disable": 25, "far": 0.05, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 42.3077% | 62.5000% | unavailable |  |
| cost-20x40-f0-b7948ffa58e5 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.1, "sign": "hard"}` | 34.6154% | 50.0000% | unavailable |  |
| cost-20x40-f0-c15dd4d4beab | `{"boundary": 10, "disable": 200, "far": 0.05, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 62.5000% | unavailable |  |
| cost-20x40-f0-cadf91c60125 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.01, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 75.0000% | unavailable |  |
| cost-20x40-f0-cdb6627a0ac2 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.3, "sign": "hard"}` | 34.6154% | 50.0000% | unavailable |  |
| cost-20x40-f0-d66476b2f221 | `{"boundary": 10, "disable": 100, "far": 0.05, "near": 0, "p0": 0.2, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 87.5000% | unavailable |  |
| cost-20x40-f0-d819a81f7f6b | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0, "sign": "hard"}` | 15.3846% | 33.3333% | unavailable |  |
| cost-20x40-f0-daeacfc423ad | `{"boundary": 200, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 46.1538% | 58.3333% | unavailable |  |
| cost-20x40-f0-dbc4a5ba9897 | `{"boundary": 10, "disable": 100, "far": 0.05, "near": 0, "p0": 0, "p1": 0.1, "sign": "hard"}` | 30.7692% | 50.0000% | unavailable |  |
| cost-20x40-f0-dd3e5e02fb78 | `{"boundary": 10, "disable": 100, "far": 0.05, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | -62.2833% |  |
| cost-20x40-f0-e4345b366d79 | `{"boundary": 10, "disable": 100, "far": 0.05, "near": 0, "p0": 0, "p1": 0.0375, "sign": "176"}` | 50.0000% | 62.5000% | unavailable |  |
| cost-20x40-f0-eab4515c34ee | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.015, "sign": "hard"}` | 38.4615% | 50.0000% | unavailable |  |
| cost-20x40-f0-ec964c636b9d | `{"boundary": 10, "disable": 25, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 38.4615% | 50.0000% | unavailable |  |
| cost-20x40-f0-ee15c8f7a509 | `{"boundary": 10, "disable": 100, "far": 0.05, "near": 0.05, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 87.5000% | unavailable |  |
| cost-20x40-f0-f2aa98eac5bf | `{"boundary": 10, "disable": 100, "far": 0.05, "near": 0, "p0": 0, "p1": 0, "sign": "hard"}` | 42.3077% | 41.6667% | unavailable |  |
| cost-20x40-f0-ff7a81d7dcab | `{"boundary": 10, "disable": 400, "far": 0.05, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 62.5000% | unavailable |  |
| length-10x10-f0-03f7c52d8f68 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "44"}` | 34.6154% | 83.3333% | unavailable |  |
| length-10x10-f0-0a6b95daf499 | `{"boundary": 50, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 34.6154% | 79.1667% | unavailable |  |
| length-10x10-f0-0bc01d547270 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0.01, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -60.2171% |  |
| length-10x10-f0-0ee89d1c3af5 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.05, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 70.8333% | unavailable |  |
| length-10x10-f0-1f24c775c606 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "11"}` | 53.8462% | 83.3333% | unavailable |  |
| length-10x10-f0-299b306b82c3 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 26.9231% | 79.1667% | -79.3645% |  |
| length-10x10-f0-2cb0bf26f547 | `{"boundary": 10, "disable": 50, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 26.9231% | 79.1667% | unavailable |  |
| length-10x10-f0-2d33708bd653 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| length-10x10-f0-335c469efb3a | `{"boundary": 10, "disable": 50, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -60.0181% |  |
| length-10x10-f0-34e128eb1a83 | `{"boundary": 10, "disable": 400, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 26.9231% | 79.1667% | unavailable |  |
| length-10x10-f0-359901695caa | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.2, "p1": 0.0375, "sign": "hard"}` | 26.9231% | 79.1667% | unavailable |  |
| length-10x10-f0-375610631f4a | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "176"}` | 53.8462% | 87.5000% | -60.7808% |  |
| length-10x10-f0-456f413ec5eb | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0.01, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -60.0723% |  |
| length-10x10-f0-4c75248e58bd | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 26.9231% | 66.6667% | unavailable |  |
| length-10x10-f0-4caf2402642a | `{"boundary": 10, "disable": 0, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 30.4348% | unavailable |  |
| length-10x10-f0-5f361d1ab0a4 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.05, "p1": 0.0375, "sign": "hard"}` | 38.4615% | 79.1667% | unavailable |  |
| length-10x10-f0-619ed0f70866 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.1, "sign": "hard"}` | 53.8462% | 87.5000% | -60.0361% |  |
| length-10x10-f0-6ea251683cf9 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "11"}` | 34.6154% | 83.3333% | unavailable |  |
| length-10x10-f0-6f3c4ea3c7dc | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.01, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| length-10x10-f0-72e2c7f38138 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -60.2534% |  |
| length-10x10-f0-73ed196882d2 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "176"}` | 30.7692% | 79.1667% | unavailable |  |
| length-10x10-f0-77759fcba1e6 | `{"boundary": 10, "disable": 400, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -61.8830% |  |
| length-10x10-f0-78c2dbc42acc | `{"boundary": 0, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -60.7808% |  |
| length-10x10-f0-7b2e41cbddca | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.3, "sign": "hard"}` | 50.0000% | 75.0000% | unavailable |  |
| length-10x10-f0-7cd8ca626ef7 | `{"boundary": 10, "disable": 200, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 26.9231% | 79.1667% | unavailable |  |
| length-10x10-f0-7eaadd24b08e | `{"boundary": 200, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -63.0005% |  |
| length-10x10-f0-80f5928f563d | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.005, "sign": "hard"}` | 26.9231% | 79.1667% | unavailable |  |
| length-10x10-f0-8d1512fb61b5 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0.2, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 75.0000% | unavailable |  |
| length-10x10-f0-939a7482b782 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0.05, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| length-10x10-f0-95c965e7b102 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.015, "sign": "hard"}` | 53.8462% | 87.5000% | -60.2534% |  |
| length-10x10-f0-a0662bafff90 | `{"boundary": 0, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 26.9231% | 79.1667% | unavailable |  |
| length-10x10-f0-a5214c123bfd | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| length-10x10-f0-b248db304b4a | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0.05, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 75.0000% | unavailable |  |
| length-10x10-f0-b40d0e49f0c3 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "44"}` | 53.8462% | 87.5000% | -61.1282% |  |
| length-10x10-f0-b7948ffa58e5 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.1, "sign": "hard"}` | 26.9231% | 79.1667% | unavailable |  |
| length-10x10-f0-ca33a9afbf3a | `{"boundary": 10, "disable": 0, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 34.6154% | 30.4348% | unavailable |  |
| length-10x10-f0-cadf91c60125 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.01, "p1": 0.0375, "sign": "hard"}` | 38.4615% | 79.1667% | unavailable |  |
| length-10x10-f0-cdb6627a0ac2 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.3, "sign": "hard"}` | 26.9231% | 79.1667% | unavailable |  |
| length-10x10-f0-d72b7873d55a | `{"boundary": 10, "disable": 200, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -60.2534% |  |
| length-10x10-f0-d819a81f7f6b | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0, "sign": "hard"}` | 19.2308% | 33.3333% | unavailable |  |
| length-10x10-f0-d91fd55263a2 | `{"boundary": 10, "disable": 25, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -59.6754% |  |
| length-10x10-f0-daeacfc423ad | `{"boundary": 200, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 26.9231% | 75.0000% | unavailable |  |
| length-10x10-f0-dd3e5e02fb78 | `{"boundary": 10, "disable": 100, "far": 0.05, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -61.2749% |  |
| length-10x10-f0-e14400259eb4 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.005, "sign": "hard"}` | 50.0000% | 75.0000% | unavailable |  |
| length-10x10-f0-e5eb4062f846 | `{"boundary": 50, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -60.9269% |  |
| length-10x10-f0-eab4515c34ee | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.015, "sign": "hard"}` | 26.9231% | 79.1667% | unavailable |  |
| length-10x10-f0-ec964c636b9d | `{"boundary": 10, "disable": 25, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 34.6154% | 83.3333% | unavailable |  |
| length-10x10-f0-f4a63df9dca4 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0, "sign": "hard"}` | 50.0000% | 75.0000% | unavailable |  |
| length-10x20-f0-03f7c52d8f68 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "44"}` | 34.6154% | 83.3333% | unavailable |  |
| length-10x20-f0-0a6b95daf499 | `{"boundary": 50, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 38.4615% | 87.5000% | unavailable |  |
| length-10x20-f0-0bc01d547270 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0.01, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -60.3243% |  |
| length-10x20-f0-0ee89d1c3af5 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.05, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| length-10x20-f0-1f24c775c606 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "11"}` | 53.8462% | 83.3333% | unavailable |  |
| length-10x20-f0-299b306b82c3 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | -78.4643% |  |
| length-10x20-f0-2cb0bf26f547 | `{"boundary": 10, "disable": 50, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| length-10x20-f0-2d33708bd653 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| length-10x20-f0-335c469efb3a | `{"boundary": 10, "disable": 50, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -60.1232% |  |
| length-10x20-f0-34e128eb1a83 | `{"boundary": 10, "disable": 400, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| length-10x20-f0-359901695caa | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.2, "p1": 0.0375, "sign": "hard"}` | 26.9231% | 75.0000% | unavailable |  |
| length-10x20-f0-375610631f4a | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "176"}` | 53.8462% | 87.5000% | -60.7466% |  |
| length-10x20-f0-456f413ec5eb | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0.01, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -60.1780% |  |
| length-10x20-f0-4c75248e58bd | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 26.9231% | 70.8333% | unavailable |  |
| length-10x20-f0-4caf2402642a | `{"boundary": 10, "disable": 0, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 26.9231% | 30.4348% | unavailable |  |
| length-10x20-f0-5f361d1ab0a4 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.05, "p1": 0.0375, "sign": "hard"}` | 38.4615% | 79.1667% | unavailable |  |
| length-10x20-f0-619ed0f70866 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.1, "sign": "hard"}` | 53.8462% | 87.5000% | -60.1415% |  |
| length-10x20-f0-6ea251683cf9 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "11"}` | 34.6154% | 83.3333% | unavailable |  |
| length-10x20-f0-6f3c4ea3c7dc | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.01, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| length-10x20-f0-72e2c7f38138 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -60.3610% |  |
| length-10x20-f0-73ed196882d2 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "176"}` | 30.7692% | 83.3333% | unavailable |  |
| length-10x20-f0-77759fcba1e6 | `{"boundary": 10, "disable": 400, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -61.9520% |  |
| length-10x20-f0-78c2dbc42acc | `{"boundary": 0, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -60.8387% |  |
| length-10x20-f0-7b2e41cbddca | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.3, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| length-10x20-f0-7cd8ca626ef7 | `{"boundary": 10, "disable": 200, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| length-10x20-f0-7eaadd24b08e | `{"boundary": 200, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -62.1765% |  |
| length-10x20-f0-80f5928f563d | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.005, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| length-10x20-f0-8d1512fb61b5 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0.2, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| length-10x20-f0-939a7482b782 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0.05, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| length-10x20-f0-95c965e7b102 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.015, "sign": "hard"}` | 53.8462% | 87.5000% | -60.3610% |  |
| length-10x20-f0-a0662bafff90 | `{"boundary": 0, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| length-10x20-f0-a5214c123bfd | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| length-10x20-f0-b248db304b4a | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0.05, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -60.2694% |  |
| length-10x20-f0-b40d0e49f0c3 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "44"}` | 53.8462% | 87.5000% | -61.1526% |  |
| length-10x20-f0-b7948ffa58e5 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.1, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| length-10x20-f0-ca33a9afbf3a | `{"boundary": 10, "disable": 0, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 34.6154% | 30.4348% | unavailable |  |
| length-10x20-f0-cadf91c60125 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.01, "p1": 0.0375, "sign": "hard"}` | 38.4615% | 79.1667% | unavailable |  |
| length-10x20-f0-cdb6627a0ac2 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.3, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| length-10x20-f0-d72b7873d55a | `{"boundary": 10, "disable": 200, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -60.3060% |  |
| length-10x20-f0-d819a81f7f6b | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0, "sign": "hard"}` | 19.2308% | 37.5000% | unavailable |  |
| length-10x20-f0-d91fd55263a2 | `{"boundary": 10, "disable": 25, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -59.7769% |  |
| length-10x20-f0-daeacfc423ad | `{"boundary": 200, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 26.9231% | 79.1667% | unavailable |  |
| length-10x20-f0-dd3e5e02fb78 | `{"boundary": 10, "disable": 100, "far": 0.05, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -61.3934% |  |
| length-10x20-f0-e14400259eb4 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.005, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| length-10x20-f0-e5eb4062f846 | `{"boundary": 50, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -61.0416% |  |
| length-10x20-f0-eab4515c34ee | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.015, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| length-10x20-f0-ec964c636b9d | `{"boundary": 10, "disable": 25, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 34.6154% | 83.3333% | unavailable |  |
| length-10x20-f0-f4a63df9dca4 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| length-20x20-f0-03f7c52d8f68 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "44"}` | 34.6154% | 79.1667% | unavailable |  |
| length-20x20-f0-0a6b95daf499 | `{"boundary": 50, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 79.1667% | unavailable |  |
| length-20x20-f0-0bc01d547270 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0.01, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -60.4676% |  |
| length-20x20-f0-0ee89d1c3af5 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.05, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| length-20x20-f0-1f24c775c606 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "11"}` | 50.0000% | 87.5000% | unavailable |  |
| length-20x20-f0-299b306b82c3 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | -76.6113% |  |
| length-20x20-f0-2cb0bf26f547 | `{"boundary": 10, "disable": 50, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| length-20x20-f0-2d33708bd653 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| length-20x20-f0-335c469efb3a | `{"boundary": 10, "disable": 50, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -60.2615% |  |
| length-20x20-f0-34e128eb1a83 | `{"boundary": 10, "disable": 400, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| length-20x20-f0-359901695caa | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.2, "p1": 0.0375, "sign": "hard"}` | 26.9231% | 83.3333% | unavailable |  |
| length-20x20-f0-375610631f4a | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "176"}` | 53.8462% | 87.5000% | -61.2782% |  |
| length-20x20-f0-456f413ec5eb | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0.01, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| length-20x20-f0-4c75248e58bd | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 26.9231% | 70.8333% | unavailable |  |
| length-20x20-f0-4caf2402642a | `{"boundary": 10, "disable": 0, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 23.0769% | 26.0870% | unavailable |  |
| length-20x20-f0-5f361d1ab0a4 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.05, "p1": 0.0375, "sign": "hard"}` | 34.6154% | 79.1667% | unavailable |  |
| length-20x20-f0-619ed0f70866 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.1, "sign": "hard"}` | 53.8462% | 87.5000% | -60.2802% |  |
| length-20x20-f0-6ea251683cf9 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "11"}` | 38.4615% | 83.3333% | unavailable |  |
| length-20x20-f0-6f3c4ea3c7dc | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.01, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| length-20x20-f0-72e2c7f38138 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -60.4301% |  |
| length-20x20-f0-73ed196882d2 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "176"}` | 30.7692% | 79.1667% | unavailable |  |
| length-20x20-f0-77759fcba1e6 | `{"boundary": 10, "disable": 400, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -60.8624% |  |
| length-20x20-f0-78c2dbc42acc | `{"boundary": 0, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -60.1867% |  |
| length-20x20-f0-7b2e41cbddca | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.3, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| length-20x20-f0-7cd8ca626ef7 | `{"boundary": 10, "disable": 200, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| length-20x20-f0-7eaadd24b08e | `{"boundary": 200, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -63.0017% |  |
| length-20x20-f0-80f5928f563d | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.005, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| length-20x20-f0-8d1512fb61b5 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0.2, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -60.4488% |  |
| length-20x20-f0-939a7482b782 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0.05, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| length-20x20-f0-95c965e7b102 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.015, "sign": "hard"}` | 53.8462% | 87.5000% | -60.5051% |  |
| length-20x20-f0-a0662bafff90 | `{"boundary": 0, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| length-20x20-f0-a5214c123bfd | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| length-20x20-f0-b248db304b4a | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0.05, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -60.3926% |  |
| length-20x20-f0-b40d0e49f0c3 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "44"}` | 53.8462% | 87.5000% | -61.4489% |  |
| length-20x20-f0-b7948ffa58e5 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.1, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| length-20x20-f0-ca33a9afbf3a | `{"boundary": 10, "disable": 0, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 42.3077% | 30.4348% | unavailable |  |
| length-20x20-f0-cadf91c60125 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.01, "p1": 0.0375, "sign": "hard"}` | 38.4615% | 79.1667% | unavailable |  |
| length-20x20-f0-cdb6627a0ac2 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.3, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| length-20x20-f0-d72b7873d55a | `{"boundary": 10, "disable": 200, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -60.3364% |  |
| length-20x20-f0-d819a81f7f6b | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0, "sign": "hard"}` | 15.3846% | 37.5000% | unavailable |  |
| length-20x20-f0-d91fd55263a2 | `{"boundary": 10, "disable": 25, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -59.9068% |  |
| length-20x20-f0-daeacfc423ad | `{"boundary": 200, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| length-20x20-f0-dd3e5e02fb78 | `{"boundary": 10, "disable": 100, "far": 0.05, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -61.0323% |  |
| length-20x20-f0-e14400259eb4 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.005, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| length-20x20-f0-e5eb4062f846 | `{"boundary": 50, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -61.4489% |  |
| length-20x20-f0-eab4515c34ee | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.015, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| length-20x20-f0-ec964c636b9d | `{"boundary": 10, "disable": 25, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 79.1667% | unavailable |  |
| length-20x20-f0-f4a63df9dca4 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0, "sign": "hard"}` | 53.8462% | 87.5000% | -61.4299% |  |
| length-20x40-f0-03f7c52d8f68 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "44"}` | 50.0000% | 75.0000% | unavailable |  |
| length-20x40-f0-0a6b95daf499 | `{"boundary": 50, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| length-20x40-f0-0bc01d547270 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0.01, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | unavailable |  |
| length-20x40-f0-0ee89d1c3af5 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.05, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| length-20x40-f0-1f24c775c606 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "11"}` | 50.0000% | 83.3333% | unavailable |  |
| length-20x40-f0-299b306b82c3 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | -82.9448% |  |
| length-20x40-f0-2cb0bf26f547 | `{"boundary": 10, "disable": 50, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| length-20x40-f0-2d33708bd653 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| length-20x40-f0-335c469efb3a | `{"boundary": 10, "disable": 50, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | unavailable |  |
| length-20x40-f0-34e128eb1a83 | `{"boundary": 10, "disable": 400, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 42.3077% | 79.1667% | unavailable |  |
| length-20x40-f0-359901695caa | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.2, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| length-20x40-f0-375610631f4a | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "176"}` | 53.8462% | 87.5000% | unavailable |  |
| length-20x40-f0-456f413ec5eb | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0.01, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | unavailable |  |
| length-20x40-f0-4c75248e58bd | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 26.9231% | 70.8333% | unavailable |  |
| length-20x40-f0-4caf2402642a | `{"boundary": 10, "disable": 0, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 26.9231% | 39.1304% | unavailable |  |
| length-20x40-f0-5f361d1ab0a4 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.05, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| length-20x40-f0-619ed0f70866 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.1, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| length-20x40-f0-6ea251683cf9 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "11"}` | 42.3077% | 79.1667% | unavailable |  |
| length-20x40-f0-6f3c4ea3c7dc | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.01, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| length-20x40-f0-72e2c7f38138 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -59.3845% |  |
| length-20x40-f0-73ed196882d2 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "176"}` | 50.0000% | 75.0000% | unavailable |  |
| length-20x40-f0-77759fcba1e6 | `{"boundary": 10, "disable": 400, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | unavailable |  |
| length-20x40-f0-78c2dbc42acc | `{"boundary": 0, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | unavailable |  |
| length-20x40-f0-7b2e41cbddca | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.3, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| length-20x40-f0-7cd8ca626ef7 | `{"boundary": 10, "disable": 200, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 42.3077% | 79.1667% | unavailable |  |
| length-20x40-f0-7eaadd24b08e | `{"boundary": 200, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 95.8333% | -62.4165% |  |
| length-20x40-f0-80f5928f563d | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.005, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| length-20x40-f0-8d1512fb61b5 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0.2, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | unavailable |  |
| length-20x40-f0-939a7482b782 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0.05, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| length-20x40-f0-95c965e7b102 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.015, "sign": "hard"}` | 53.8462% | 87.5000% | unavailable |  |
| length-20x40-f0-a0662bafff90 | `{"boundary": 0, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 75.0000% | unavailable |  |
| length-20x40-f0-a5214c123bfd | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 83.3333% | unavailable |  |
| length-20x40-f0-b248db304b4a | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0.05, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | unavailable |  |
| length-20x40-f0-b40d0e49f0c3 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "44"}` | 53.8462% | 87.5000% | unavailable |  |
| length-20x40-f0-b7948ffa58e5 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.1, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| length-20x40-f0-ca33a9afbf3a | `{"boundary": 10, "disable": 0, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 26.9231% | 31.8182% | unavailable |  |
| length-20x40-f0-cadf91c60125 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.01, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| length-20x40-f0-cdb6627a0ac2 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.3, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| length-20x40-f0-d72b7873d55a | `{"boundary": 10, "disable": 200, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | unavailable |  |
| length-20x40-f0-d819a81f7f6b | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0, "sign": "hard"}` | 19.2308% | 37.5000% | unavailable |  |
| length-20x40-f0-d91fd55263a2 | `{"boundary": 10, "disable": 25, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | unavailable |  |
| length-20x40-f0-daeacfc423ad | `{"boundary": 200, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 87.5000% | unavailable |  |
| length-20x40-f0-dd3e5e02fb78 | `{"boundary": 10, "disable": 100, "far": 0.05, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | -65.1765% |  |
| length-20x40-f0-e14400259eb4 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.005, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| length-20x40-f0-e5eb4062f846 | `{"boundary": 50, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 87.5000% | unavailable |  |
| length-20x40-f0-eab4515c34ee | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.015, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| length-20x40-f0-ec964c636b9d | `{"boundary": 10, "disable": 25, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| length-20x40-f0-f4a63df9dca4 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| presence-10x10-f0-03f7c52d8f68 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "44"}` | 38.4615% | 87.5000% | unavailable |  |
| presence-10x10-f0-040de5cd74e6 | `{"boundary": 10, "disable": 400, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 83.3333% | unavailable |  |
| presence-10x10-f0-0a6b95daf499 | `{"boundary": 50, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-10x10-f0-0ad0de4b6497 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "176"}` | 61.5385% | 79.1667% | unavailable |  |
| presence-10x10-f0-0ee89d1c3af5 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.05, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 46.1538% | 66.6667% | unavailable |  |
| presence-10x10-f0-13f56f9f9451 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.3, "sign": "hard"}` | 61.5385% | 79.1667% | unavailable |  |
| presence-10x10-f0-22d41ac26bc8 | `{"boundary": 10, "disable": 25, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 65.3846% | 91.6667% | -60.8983% |  |
| presence-10x10-f0-299b306b82c3 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 75.0000% | -204.2827% |  |
| presence-10x10-f0-2cb0bf26f547 | `{"boundary": 10, "disable": 50, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-10x10-f0-34e128eb1a83 | `{"boundary": 10, "disable": 400, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-10x10-f0-359901695caa | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.2, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 83.3333% | unavailable |  |
| presence-10x10-f0-3b81f34c0aff | `{"boundary": 50, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 65.3846% | 91.6667% | -60.3234% |  |
| presence-10x10-f0-41449ebff635 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0.2, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 66.6667% | unavailable |  |
| presence-10x10-f0-41bbd5f7d429 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "11"}` | 65.3846% | 91.6667% | -61.5691% |  |
| presence-10x10-f0-4c75248e58bd | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 70.8333% | unavailable |  |
| presence-10x10-f0-4caf2402642a | `{"boundary": 10, "disable": 0, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 26.9231% | 25.0000% | unavailable |  |
| presence-10x10-f0-50792d92f305 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0, "sign": "hard"}` | 53.8462% | 87.5000% | unavailable |  |
| presence-10x10-f0-5f361d1ab0a4 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.05, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 83.3333% | unavailable |  |
| presence-10x10-f0-6ea251683cf9 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "11"}` | 38.4615% | 87.5000% | unavailable |  |
| presence-10x10-f0-6f3c4ea3c7dc | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.01, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 83.3333% | unavailable |  |
| presence-10x10-f0-72e2c7f38138 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 87.5000% | unavailable |  |
| presence-10x10-f0-73ed196882d2 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "176"}` | 38.4615% | 83.3333% | unavailable |  |
| presence-10x10-f0-7cd8ca626ef7 | `{"boundary": 10, "disable": 200, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-10x10-f0-80f5928f563d | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.005, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-10x10-f0-95fa6941c724 | `{"boundary": 0, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 79.1667% | unavailable |  |
| presence-10x10-f0-a0662bafff90 | `{"boundary": 0, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-10x10-f0-a218e0647bb9 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "44"}` | 61.5385% | 79.1667% | unavailable |  |
| presence-10x10-f0-a23b9b95fc05 | `{"boundary": 200, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 61.5385% | 83.3333% | unavailable |  |
| presence-10x10-f0-a5214c123bfd | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 65.3846% | 91.6667% | -61.5385% |  |
| presence-10x10-f0-ac0a6ac831d8 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.005, "sign": "hard"}` | 50.0000% | 70.8333% | unavailable |  |
| presence-10x10-f0-b43effe3a7db | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0.01, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 34.6154% | 83.3333% | unavailable |  |
| presence-10x10-f0-b7948ffa58e5 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.1, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-10x10-f0-b9052bd47e1c | `{"boundary": 10, "disable": 0, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 38.4615% | 41.6667% | unavailable |  |
| presence-10x10-f0-bd48d88a0fb1 | `{"boundary": 10, "disable": 50, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 65.3846% | 91.6667% | -61.1416% |  |
| presence-10x10-f0-cadf91c60125 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.01, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 83.3333% | unavailable |  |
| presence-10x10-f0-cdb6627a0ac2 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.3, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-10x10-f0-d58de4bef07f | `{"boundary": 10, "disable": 200, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 83.3333% | unavailable |  |
| presence-10x10-f0-d76114838bf5 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.1, "sign": "hard"}` | 34.6154% | 87.5000% | unavailable |  |
| presence-10x10-f0-d819a81f7f6b | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0, "sign": "hard"}` | 15.3846% | 41.6667% | unavailable |  |
| presence-10x10-f0-daeacfc423ad | `{"boundary": 200, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 38.4615% | 91.6667% | unavailable |  |
| presence-10x10-f0-dbe5aa3e4db7 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0.05, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 75.0000% | unavailable |  |
| presence-10x10-f0-dd3e5e02fb78 | `{"boundary": 10, "disable": 100, "far": 0.05, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 70.8333% | unavailable |  |
| presence-10x10-f0-e37ee84b7d76 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0.01, "p1": 0.0375, "sign": "hard"}` | 61.5385% | 79.1667% | unavailable |  |
| presence-10x10-f0-e52cf3900b82 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0.05, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 95.8333% | unavailable |  |
| presence-10x10-f0-eab4515c34ee | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.015, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-10x10-f0-eb71f109217c | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.015, "sign": "hard"}` | 65.3846% | 91.6667% | -60.9894% |  |
| presence-10x10-f0-ec964c636b9d | `{"boundary": 10, "disable": 25, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-10x10-f0-ff06f81e2408 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 38.4615% | 87.5000% | unavailable |  |
| presence-10x20-f0-03f7c52d8f68 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "44"}` | 30.7692% | 87.5000% | unavailable |  |
| presence-10x20-f0-05ef9a496f77 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "176"}` | 30.7692% | 70.8333% | unavailable |  |
| presence-10x20-f0-095cf9382729 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.2, "p0": 0.2, "p1": 0.0375, "sign": "hard"}` | 26.9231% | 79.1667% | unavailable |  |
| presence-10x20-f0-0a6b95daf499 | `{"boundary": 50, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-10x20-f0-0ee89d1c3af5 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.05, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 46.1538% | 79.1667% | unavailable |  |
| presence-10x20-f0-0f0ce0912d3f | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.2, "p0": 0, "p1": 0.005, "sign": "hard"}` | 50.0000% | 83.3333% | unavailable |  |
| presence-10x20-f0-24cacdd75d87 | `{"boundary": 10, "disable": 0, "far": 0, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 45.8333% | unavailable |  |
| presence-10x20-f0-299b306b82c3 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 75.0000% | -202.9867% |  |
| presence-10x20-f0-2cb0bf26f547 | `{"boundary": 10, "disable": 50, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-10x20-f0-2d33708bd653 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 65.3846% | 91.6667% | -55.1216% |  |
| presence-10x20-f0-34e128eb1a83 | `{"boundary": 10, "disable": 400, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-10x20-f0-359901695caa | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.2, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| presence-10x20-f0-4729b9dc74cb | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.2, "p0": 0.05, "p1": 0.0375, "sign": "hard"}` | 26.9231% | 70.8333% | unavailable |  |
| presence-10x20-f0-477d558e2c1d | `{"boundary": 10, "disable": 400, "far": 0, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 87.5000% | unavailable |  |
| presence-10x20-f0-4b78a98db933 | `{"boundary": 50, "disable": 100, "far": 0, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 46.1538% | 79.1667% | unavailable |  |
| presence-10x20-f0-4c75248e58bd | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 57.6923% | 83.3333% | -74.7976% |  |
| presence-10x20-f0-4caf2402642a | `{"boundary": 10, "disable": 0, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 33.3333% | unavailable |  |
| presence-10x20-f0-4e444f500b81 | `{"boundary": 10, "disable": 100, "far": 0.05, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 91.6667% | unavailable |  |
| presence-10x20-f0-55e749eba6af | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.2, "p0": 0.01, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 83.3333% | unavailable |  |
| presence-10x20-f0-5f361d1ab0a4 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.05, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 83.3333% | unavailable |  |
| presence-10x20-f0-617696ebd692 | `{"boundary": 10, "disable": 50, "far": 0, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 34.6154% | 45.8333% | unavailable |  |
| presence-10x20-f0-631564d18d25 | `{"boundary": 10, "disable": 200, "far": 0, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 87.5000% | unavailable |  |
| presence-10x20-f0-6ea251683cf9 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "11"}` | 38.4615% | 87.5000% | unavailable |  |
| presence-10x20-f0-6f3c4ea3c7dc | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.01, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 75.0000% | unavailable |  |
| presence-10x20-f0-72e2c7f38138 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| presence-10x20-f0-73ed196882d2 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "176"}` | 38.4615% | 83.3333% | unavailable |  |
| presence-10x20-f0-770b60a0a541 | `{"boundary": 200, "disable": 100, "far": 0, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 38.4615% | 70.8333% | unavailable |  |
| presence-10x20-f0-7cd8ca626ef7 | `{"boundary": 10, "disable": 200, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-10x20-f0-80f5928f563d | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.005, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-10x20-f0-97f1aabbd85e | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.2, "p0": 0, "p1": 0, "sign": "hard"}` | 46.1538% | 75.0000% | unavailable |  |
| presence-10x20-f0-9e1d0d14049e | `{"boundary": 10, "disable": 25, "far": 0, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 46.1538% | 79.1667% | unavailable |  |
| presence-10x20-f0-a0662bafff90 | `{"boundary": 0, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-10x20-f0-a5214c123bfd | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| presence-10x20-f0-adcd9106d518 | `{"boundary": 0, "disable": 100, "far": 0, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 26.9231% | 70.8333% | unavailable |  |
| presence-10x20-f0-b7948ffa58e5 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.1, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-10x20-f0-cadf91c60125 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.01, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 83.3333% | unavailable |  |
| presence-10x20-f0-cb6755dfcfd0 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "11"}` | 30.7692% | 70.8333% | unavailable |  |
| presence-10x20-f0-cbe78929c288 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.2, "p0": 0, "p1": 0.1, "sign": "hard"}` | 46.1538% | 79.1667% | unavailable |  |
| presence-10x20-f0-cdb6627a0ac2 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.3, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-10x20-f0-ce178b281fd7 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.2, "p0": 0, "p1": 0.015, "sign": "hard"}` | 34.6154% | 58.3333% | unavailable |  |
| presence-10x20-f0-d5268225622f | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.2, "p0": 0, "p1": 0.3, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| presence-10x20-f0-d6ead0421772 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "44"}` | 50.0000% | 70.8333% | unavailable |  |
| presence-10x20-f0-d819a81f7f6b | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0, "sign": "hard"}` | 15.3846% | 45.8333% | unavailable |  |
| presence-10x20-f0-daeacfc423ad | `{"boundary": 200, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 38.4615% | 91.6667% | unavailable |  |
| presence-10x20-f0-dd3e5e02fb78 | `{"boundary": 10, "disable": 100, "far": 0.05, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 70.8333% | unavailable |  |
| presence-10x20-f0-eab4515c34ee | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.015, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-10x20-f0-ec964c636b9d | `{"boundary": 10, "disable": 25, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-10x20-f0-ff06f81e2408 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 38.4615% | 87.5000% | unavailable |  |
| presence-20x20-f0-03f7c52d8f68 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "44"}` | 30.7692% | 87.5000% | unavailable |  |
| presence-20x20-f0-040de5cd74e6 | `{"boundary": 10, "disable": 400, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 95.8333% | -58.2732% |  |
| presence-20x20-f0-0a6b95daf499 | `{"boundary": 50, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-20x20-f0-0ad0de4b6497 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "176"}` | 50.0000% | 87.5000% | unavailable |  |
| presence-20x20-f0-0ee89d1c3af5 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.05, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 46.1538% | 75.0000% | unavailable |  |
| presence-20x20-f0-13f56f9f9451 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.3, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| presence-20x20-f0-22d41ac26bc8 | `{"boundary": 10, "disable": 25, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 95.8333% | -60.4307% |  |
| presence-20x20-f0-299b306b82c3 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 75.0000% | -226.4068% |  |
| presence-20x20-f0-2cb0bf26f547 | `{"boundary": 10, "disable": 50, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-20x20-f0-34e128eb1a83 | `{"boundary": 10, "disable": 400, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-20x20-f0-359901695caa | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.2, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| presence-20x20-f0-3b81f34c0aff | `{"boundary": 50, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 95.8333% | -59.8937% |  |
| presence-20x20-f0-41449ebff635 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0.2, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| presence-20x20-f0-41bbd5f7d429 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "11"}` | 53.8462% | 87.5000% | unavailable |  |
| presence-20x20-f0-4c75248e58bd | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 70.8333% | unavailable |  |
| presence-20x20-f0-4caf2402642a | `{"boundary": 10, "disable": 0, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 26.9231% | 20.8333% | unavailable |  |
| presence-20x20-f0-50792d92f305 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0, "sign": "hard"}` | 53.8462% | 87.5000% | unavailable |  |
| presence-20x20-f0-5f361d1ab0a4 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.05, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 83.3333% | unavailable |  |
| presence-20x20-f0-6ea251683cf9 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "11"}` | 26.9231% | 87.5000% | unavailable |  |
| presence-20x20-f0-6f3c4ea3c7dc | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.01, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-20x20-f0-72e2c7f38138 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| presence-20x20-f0-73ed196882d2 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "176"}` | 38.4615% | 83.3333% | unavailable |  |
| presence-20x20-f0-7cd8ca626ef7 | `{"boundary": 10, "disable": 200, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-20x20-f0-80f5928f563d | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.005, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-20x20-f0-95fa6941c724 | `{"boundary": 0, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 95.8333% | -58.4585% |  |
| presence-20x20-f0-a0662bafff90 | `{"boundary": 0, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-20x20-f0-a218e0647bb9 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "44"}` | 50.0000% | 87.5000% | unavailable |  |
| presence-20x20-f0-a23b9b95fc05 | `{"boundary": 200, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 95.8333% | -60.7164% |  |
| presence-20x20-f0-a5214c123bfd | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 95.8333% | -58.4585% |  |
| presence-20x20-f0-ac0a6ac831d8 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.005, "sign": "hard"}` | 50.0000% | 70.8333% | unavailable |  |
| presence-20x20-f0-b43effe3a7db | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0.01, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| presence-20x20-f0-b7948ffa58e5 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.1, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-20x20-f0-b9052bd47e1c | `{"boundary": 10, "disable": 0, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 38.4615% | 33.3333% | unavailable |  |
| presence-20x20-f0-bd48d88a0fb1 | `{"boundary": 10, "disable": 50, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 95.8333% | -60.9713% |  |
| presence-20x20-f0-cadf91c60125 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.01, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 83.3333% | unavailable |  |
| presence-20x20-f0-cdb6627a0ac2 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.3, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-20x20-f0-d58de4bef07f | `{"boundary": 10, "disable": 200, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 95.8333% | -58.3658% |  |
| presence-20x20-f0-d76114838bf5 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.1, "sign": "hard"}` | 38.4615% | 91.6667% | unavailable |  |
| presence-20x20-f0-d819a81f7f6b | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0, "sign": "hard"}` | 0.0000% | 16.6667% | unavailable |  |
| presence-20x20-f0-daeacfc423ad | `{"boundary": 200, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 26.9231% | 83.3333% | unavailable |  |
| presence-20x20-f0-dbe5aa3e4db7 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0.05, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 75.0000% | unavailable |  |
| presence-20x20-f0-dd3e5e02fb78 | `{"boundary": 10, "disable": 100, "far": 0.05, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 70.8333% | unavailable |  |
| presence-20x20-f0-e37ee84b7d76 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0.01, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| presence-20x20-f0-e52cf3900b82 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0.05, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 95.8333% | -58.7373% |  |
| presence-20x20-f0-eab4515c34ee | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.015, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-20x20-f0-eb71f109217c | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.015, "sign": "hard"}` | 50.0000% | 70.8333% | unavailable |  |
| presence-20x20-f0-ec964c636b9d | `{"boundary": 10, "disable": 25, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-20x20-f0-ff06f81e2408 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 42.3077% | 87.5000% | unavailable |  |
| presence-20x40-f0-03f7c52d8f68 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "44"}` | 42.3077% | 58.3333% | unavailable |  |
| presence-20x40-f0-040de5cd74e6 | `{"boundary": 10, "disable": 400, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| presence-20x40-f0-0a6b95daf499 | `{"boundary": 50, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 42.3077% | 66.6667% | unavailable |  |
| presence-20x40-f0-0ad0de4b6497 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "176"}` | 50.0000% | 70.8333% | unavailable |  |
| presence-20x40-f0-0ee89d1c3af5 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.05, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 46.1538% | 66.6667% | unavailable |  |
| presence-20x40-f0-13f56f9f9451 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.3, "sign": "hard"}` | 30.7692% | 75.0000% | unavailable |  |
| presence-20x40-f0-22d41ac26bc8 | `{"boundary": 10, "disable": 25, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| presence-20x40-f0-299b306b82c3 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 42.3077% | 70.8333% | -148.4243% |  |
| presence-20x40-f0-2cb0bf26f547 | `{"boundary": 10, "disable": 50, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 42.3077% | 70.8333% | unavailable |  |
| presence-20x40-f0-34e128eb1a83 | `{"boundary": 10, "disable": 400, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 42.3077% | 70.8333% | unavailable |  |
| presence-20x40-f0-359901695caa | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.2, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 70.8333% | unavailable |  |
| presence-20x40-f0-3b81f34c0aff | `{"boundary": 50, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| presence-20x40-f0-41449ebff635 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0.2, "p1": 0.0375, "sign": "hard"}` | 61.5385% | 87.5000% | -60.1005% |  |
| presence-20x40-f0-41bbd5f7d429 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "11"}` | 53.8462% | 87.5000% | unavailable |  |
| presence-20x40-f0-4c75248e58bd | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 75.0000% | unavailable |  |
| presence-20x40-f0-4caf2402642a | `{"boundary": 10, "disable": 0, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 19.2308% | 13.0435% | unavailable |  |
| presence-20x40-f0-50792d92f305 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| presence-20x40-f0-5f361d1ab0a4 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.05, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 75.0000% | unavailable |  |
| presence-20x40-f0-6ea251683cf9 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "11"}` | 30.7692% | 79.1667% | unavailable |  |
| presence-20x40-f0-6f3c4ea3c7dc | `{"boundary": 10, "disable": 100, "far": 0, "near": 0.01, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 66.6667% | unavailable |  |
| presence-20x40-f0-72e2c7f38138 | `{"boundary": 10, "disable": 100, "far": 0.2, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 70.8333% | unavailable |  |
| presence-20x40-f0-73ed196882d2 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "176"}` | 42.3077% | 58.3333% | unavailable |  |
| presence-20x40-f0-7cd8ca626ef7 | `{"boundary": 10, "disable": 200, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 42.3077% | 70.8333% | unavailable |  |
| presence-20x40-f0-80f5928f563d | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.005, "sign": "hard"}` | 42.3077% | 70.8333% | unavailable |  |
| presence-20x40-f0-95fa6941c724 | `{"boundary": 0, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| presence-20x40-f0-a0662bafff90 | `{"boundary": 0, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 42.3077% | 70.8333% | unavailable |  |
| presence-20x40-f0-a218e0647bb9 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "44"}` | 50.0000% | 79.1667% | unavailable |  |
| presence-20x40-f0-a23b9b95fc05 | `{"boundary": 200, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 87.5000% | unavailable |  |
| presence-20x40-f0-a5214c123bfd | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | -59.3696% |  |
| presence-20x40-f0-ac0a6ac831d8 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.005, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| presence-20x40-f0-b43effe3a7db | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0.01, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 34.6154% | 83.3333% | unavailable |  |
| presence-20x40-f0-b7948ffa58e5 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.1, "sign": "hard"}` | 42.3077% | 70.8333% | unavailable |  |
| presence-20x40-f0-b9052bd47e1c | `{"boundary": 10, "disable": 0, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 30.7692% | 34.7826% | unavailable |  |
| presence-20x40-f0-bd48d88a0fb1 | `{"boundary": 10, "disable": 50, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| presence-20x40-f0-cadf91c60125 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0.01, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 75.0000% | unavailable |  |
| presence-20x40-f0-cdb6627a0ac2 | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.3, "sign": "hard"}` | 42.3077% | 62.5000% | unavailable |  |
| presence-20x40-f0-d58de4bef07f | `{"boundary": 10, "disable": 200, "far": 0.01, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| presence-20x40-f0-d76114838bf5 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.1, "sign": "hard"}` | 53.8462% | 75.0000% | unavailable |  |
| presence-20x40-f0-d819a81f7f6b | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0, "sign": "hard"}` | 15.3846% | 50.0000% | unavailable |  |
| presence-20x40-f0-daeacfc423ad | `{"boundary": 200, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 42.3077% | 66.6667% | unavailable |  |
| presence-20x40-f0-dbe5aa3e4db7 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0.05, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 53.8462% | 75.0000% | unavailable |  |
| presence-20x40-f0-dd3e5e02fb78 | `{"boundary": 10, "disable": 100, "far": 0.05, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 70.8333% | unavailable |  |
| presence-20x40-f0-e37ee84b7d76 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0.01, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| presence-20x40-f0-e52cf3900b82 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0.05, "p1": 0.0375, "sign": "hard"}` | 50.0000% | 79.1667% | unavailable |  |
| presence-20x40-f0-eab4515c34ee | `{"boundary": 10, "disable": 100, "far": 0, "near": 0, "p0": 0, "p1": 0.015, "sign": "hard"}` | 42.3077% | 70.8333% | unavailable |  |
| presence-20x40-f0-eb71f109217c | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0, "p0": 0, "p1": 0.015, "sign": "hard"}` | 50.0000% | 70.8333% | unavailable |  |
| presence-20x40-f0-ec964c636b9d | `{"boundary": 10, "disable": 25, "far": 0, "near": 0, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 42.3077% | 62.5000% | unavailable |  |
| presence-20x40-f0-ff06f81e2408 | `{"boundary": 10, "disable": 100, "far": 0.01, "near": 0.2, "p0": 0, "p1": 0.0375, "sign": "hard"}` | 46.1538% | 66.6667% | unavailable |  |
