# Equal-scroll tracing comparison

Overall score: arithmetic mean of the five per-scroll means, **20% each**.
Lengths are mm. Best/tied values are bold; old pooled overall scores are omitted.
Classic is longer-trained; modern checkpoint ages are recorded below or in their source reports.
This comparison does not match training budgets.

## Overall

| Model | Equal-scroll score ↑ |
|---|---:|
| Classic | 5.723 |
| Best EMA99 | 6.426 |
| Best EMA999 | 6.742 |
| New current999 | 6.444 |
| New current99 | **6.743** |

## PHerc0139 — 10 fibers

20 directions; 1392.347mm directed reference.
Total segments = restarts + directions.

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ | Zero ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Classic | 564 | 2.384 | **0.175** | **0.0071** | 11.631 | 8 |
| Best EMA99 | **391** | **3.388** | 0.096 | 0.0065 | 16.007 | **3** |
| Best EMA999 | 427 | 3.115 | 0.085 | 0.0040 | 15.627 | 7 |
| New current999 | 392 | 3.379 | 0.097 | 0.0070 | **16.037** | 9 |
| New current99 | 430 | 3.094 | 0.072 | 0.0061 | 15.872 | 5 |

## PHerc0332 — 3 fibers

6 directions; 471.222mm directed reference.
Total segments = restarts + directions.

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ | Zero ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Classic | 27 | 14.279 | 4.206 | 0.0074 | 48.805 | 1 |
| Best EMA99 | 19 | 18.849 | **8.471** | 0.0034 | 107.402 | **0** |
| Best EMA999 | 17 | 20.488 | 7.823 | **0.0116** | 115.866 | 1 |
| New current999 | 19 | 18.849 | 4.019 | 0.0080 | 107.369 | **0** |
| New current99 | **16** | **21.419** | 4.263 | 0.0049 | **120.057** | **0** |

## PHerc1299 — 5 fibers

10 directions; 599.748mm directed reference.
Total segments = restarts + directions.

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ | Zero ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Classic | **159** | **3.549** | **0.803** | **0.0315** | **17.573** | **1** |
| Best EMA99 | 206 | 2.777 | 0.257 | 0.0137 | 13.963 | 2 |
| Best EMA999 | 214 | 2.677 | 0.238 | 0.0073 | 15.534 | 3 |
| New current999 | 210 | 2.726 | 0.276 | 0.0088 | 14.019 | **1** |
| New current99 | 197 | 2.897 | 0.321 | 0.0085 | 15.366 | 3 |

## PHerc1451 — 2 fibers

4 directions; 94.430mm directed reference.
Total segments = restarts + directions.

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ | Zero ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Classic | **19** | **4.106** | **1.862** | **0.0707** | 17.222 | **0** |
| Best EMA99 | 32 | 2.623 | 0.278 | 0.0137 | 17.268 | **0** |
| Best EMA999 | 28 | 2.951 | 0.544 | 0.0139 | **17.270** | **0** |
| New current999 | 34 | 2.485 | 0.283 | 0.0131 | 14.152 | **0** |
| New current99 | 37 | 2.303 | 0.115 | 0.0063 | 11.439 | 1 |

## PHercParis4 — 30 fibers

60 directions; 5523.811mm directed reference.
Total segments = restarts + directions.

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ | Zero ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Classic | 1226 | 4.295 | **0.185** | **0.0128** | 23.435 | 12 |
| Best EMA99 | 1169 | 4.495 | 0.127 | 0.0112 | 27.594 | **9** |
| Best EMA999 | 1173 | 4.480 | 0.113 | 0.0096 | 25.209 | 10 |
| New current999 | **1095** | **4.783** | 0.134 | 0.0098 | **28.273** | **9** |
| New current99 | 1321 | 4.000 | 0.082 | 0.0106 | 22.394 | 11 |

## Checkpoint provenance

| Label | Step | Samples | Decay | Test loss |
|---|---:|---:|---:|---:|
| Best EMA99 | 14700 | 1881600 | 0.99 | 0.373058 |
| Best EMA999 | 15000 | 1920000 | 0.999 | 0.380882 |
| New current999 | 20700 | 2649600 | 0.999 | 0.390237 |
| New current99 | 20700 | 2649600 | 0.99 | 0.387144 |

## Evaluation sources

- Classic: `/ephemeral/hendrik/fiber3d_train2/evaluations/classic_full_dynamic256_b8_20260911_v3_heap`
- Best EMA99: `/ephemeral/hendrik/fiber3d_train2/comparisons/combined_best_pair_20260912_19m/best99-retry1`
  SHA256 `dffd46a25b0ecca7f54373736b412cfbb92119f5c26c936ef7e3d968b4d99abf`.
- Best EMA999: `/ephemeral/hendrik/fiber3d_train2/comparisons/combined_best_pair_20260912_19m/best999`
  SHA256 `a55fd90653780ba7e37b3e838c1186bb62ca1e7884d4ada94de18a941eabf409`.
- New current999: `/ephemeral/hendrik/fiber3d_train2/comparisons/combined_current_three_20260912_26m/current999`
  SHA256 `a09b0ae4e25a5845096132a19e01c02cf6d8b276f1e26ba076a9d336a2b8d91e`.
- New current99: `/ephemeral/hendrik/fiber3d_train2/comparisons/combined_current_three_20260912_26m/current99`
  SHA256 `0b6aab1354b963a48709126a40d282d41472be7efc016a285989ec21d3831c73`.

Small-scroll support is limited (0332:3fibers;1451:2fibers).
No statistical significance claim. Training is not stopped or restarted by evaluation.
