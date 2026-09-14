# Embedding tracing: consolidated completed refinements

Frozen EMA99 update6500 / 832000 patches, SHA256
`0a2651b711e7e5ee6d445ac8c68ff482c1951798ced2f44896c25602e1c50417`.
Whole PHerc1299 fiber `lt_20260825T075316065_000069.json` (alias PHerc1229),
both directions, unchanged GT ellipse/restart evaluation. This is tuning data,
not independent validation. Training is stopped; GPU work uses the manager.

Completed rounds BEFORE the active boundary/multi-start round, not its final
results. Every listed winner reproduced its exact hash. Defaults are unchanged.
Settings below are retained exactly from reports; some modes ignore history
settings (e.g. EMA ignores window/count/spacing, LRU has no age expiry).

| Family | Restarts | Parameters | Replay SHA256 prefix |
|---|---:|---|---|
| anchors-lru-best-novel | 71 | `{"aggregation":"best","anchors":4,"count":8,"ema_mix":0,"falloff":"uniform","half_life":512,"mode":"anchors-lru","novelty":0.1,"spacing":512,"weight":0.25,"window":1024}` | `4169ae649391be7b` |
| appearance-best | 72 | `{"anchors":2,"count":8,"ema_mix":0,"falloff":"uniform","half_life":512,"mode":"appearance","spacing":512,"weight":0.25,"window":1024}` | `1e44d0e26c00a7e8` |
| appearance-support-softmin | 71 | `{"aggregation":"support-softmin","anchors":2,"count":8,"ema_mix":0,"falloff":"uniform","half_life":512,"mode":"appearance","spacing":512,"temperature":0.03,"weight":0.25,"window":1024}` | `212a820a30d955a0` |
| anchors-lru-best-plain | 66 | `{"aggregation":"best","anchors":2,"count":8,"ema_mix":0,"falloff":"uniform","half_life":512,"mode":"anchors-lru","spacing":512,"weight":0.1,"window":1024}` | `68d07241e4ac5045` |
| anchors-lru-mean-plain | 68 | `{"aggregation":"mean","anchors":4,"count":8,"ema_mix":0,"falloff":"uniform","half_life":512,"mode":"anchors-lru","spacing":512,"weight":0.3,"window":1024}` | `d565c850c366ad77` |
| anchors-lru-mean-novel | **64** | `{"aggregation":"mean","anchors":2,"count":8,"ema_mix":0,"falloff":"uniform","half_life":512,"mode":"anchors-lru","novelty":0.1,"spacing":512,"weight":0.1,"window":1024}` | `013d255be05b3b6f` |
| anchors | 67 | `{"anchors":8,"count":64,"ema_mix":0.9,"falloff":"uniform","half_life":128,"mode":"anchors","spacing":256,"weight":0.25,"window":1024}` | `d4b3e2e2d5c6309c` |
| ema | 69 | `{"count":8,"falloff":"gaussian","half_life":512,"mode":"ema","spacing":128,"weight":0.25,"window":1024}` | `6e590851a9a00555` |
| ema-gated | 75 | `{"count":8,"falloff":"gaussian","half_life":512,"mode":"ema-gated","spacing":128,"weight":0.25,"window":1024}` | `61c43dd265571b22` |
| ema-seed | 78 | `{"count":8,"falloff":"gaussian","half_life":128,"mode":"ema-seed","spacing":128,"weight":0.25,"window":1024}` | `4615251b2703c31d` |
| ema-seed-gated | 73 | `{"count":8,"falloff":"gaussian","half_life":1024,"mode":"ema-seed-gated","spacing":128,"weight":0.3,"window":1024}` | `0d142cb149d0319a` |
| best | 74 | `{"count":64,"falloff":"uniform","mode":"best","spacing":8,"weight":0.25,"window":256}` | `c0cf719af25c1bca` |
| mean | 70 | `{"count":8,"falloff":"uniform","spacing":32,"weight":0.05,"window":1024}` | `ecb3e717a0a644d5` |
| seed | 72 | `{"count":8,"falloff":"uniform","mode":"seed","spacing":128,"weight":0.3,"window":1024}` | `eca96c444d658f8e` |
| hit-lru-mean | 66 | `{"aggregation":"mean","anchors":2,"count":8,"ema_mix":0,"falloff":"uniform","mode":"anchors-lru","novelty":0.1,"spacing":512,"touch":"hit","weight":0.1,"window":4096}` | `48d9c4d6646ea8f6` |
| hit-lru-best | 68 | `{"aggregation":"best","anchors":2,"count":8,"ema_mix":0,"falloff":"uniform","mode":"anchors-lru","novelty":0.1,"spacing":768,"touch":"hit","weight":0.25,"window":4096}` | `22e9f834435bb33c` |
| delayed-ema-start0.25 | 69 | `{"count":8,"delay":0,"falloff":"gaussian","half_life":512,"mode":"ema-delayed","spacing":128,"weight":0.25,"window":1024}` | `6e590851a9a00555` |
| delayed-ema-start0.1 | 71 | `{"count":8,"delay":512,"falloff":"gaussian","half_life":1024,"mode":"ema-delayed","spacing":128,"weight":0.1,"window":1024}` | `bc1fff274e44c2e5` |

## Finite search evidence

Coordinate descent uses strict improvements, retains the center on ties, and
cycles until no tested move improves restarts. Neighbor-weight steps are0.05.
This does not prove global optimality. Distances are base voxels. Full hashes,
every replay and exact parameters:
- [embedding_ema99_all_refine_20260913_v1](/ephemeral/hendrik/fiber3d_train2/evaluations/embedding_ema99_all_refine_20260913_v1/report.json)
- [embedding_ema99_new_method_refine_20260914_v1](/ephemeral/hendrik/fiber3d_train2/evaluations/embedding_ema99_new_method_refine_20260914_v1/report.json)

| Family | Explicit axes | Neighbor weights |
|---|---|---|
| anchors-lru-best-novel | `{"anchors":[2,4,8],"ema_mix":[0,0.5,0.9],"half_life":[128,512,2048],"novelty":[0.02,0.05,0.1,0.2,0.3],"spacing":[8,32,64,128,512],"weight":[0.1,0.25,0.5]}` | 0.05 |
| appearance-best | `{"anchors":[2,4,8],"ema_mix":[0,0.5,0.9],"half_life":[128,512,2048],"spacing":[8,32,64,128,512],"weight":[0.1,0.25,0.5]}` | 0.05 |
| appearance-support-softmin | `{"anchors":[2,4,8],"ema_mix":[0,0.5,0.9],"half_life":[128,512,2048],"spacing":[8,32,64,128,512],"temperature":[0.01,0.03,0.1,0.3,1],"weight":[0.1,0.25,0.5]}` | 0.05 |
| anchors-lru-best-plain | `{"anchors":[2,4,8],"ema_mix":[0,0.5,0.9],"half_life":[128,512,2048],"spacing":[8,32,64,128,512],"weight":[0.1,0.25,0.5]}` | 0.05 |
| anchors-lru-mean-plain | `{"anchors":[2,4,8],"ema_mix":[0,0.5,0.9],"half_life":[128,512,2048],"spacing":[8,32,64,128,512],"weight":[0.1,0.25,0.5]}` | 0.05 |
| anchors-lru-mean-novel | `{"anchors":[2,4,8],"ema_mix":[0,0.5,0.9],"half_life":[128,512,2048],"novelty":[0.02,0.05,0.1,0.2,0.3],"spacing":[8,32,64,128,512],"weight":[0.1,0.25,0.5]}` | 0.05 |
| anchors | `{"anchors":[2,4,8],"ema_mix":[0,0.5,0.9],"half_life":[128,512,2048],"spacing":[8,32,128,256],"weight":[0.1,0.25,0.5],"window":[1024,4096,16384]}` | 0.05 |
| ema | `{"half_life":[64,128,256,384,512,768,1024,2048],"weight":[0.1,0.25,0.5]}` | 0.05 |
| ema-gated | `{"half_life":[64,128,256,384,512,768,1024,2048],"weight":[0.1,0.25,0.5]}` | 0.05 |
| ema-seed | `{"half_life":[64,128,256,384,512,768,1024,2048],"weight":[0.1,0.25,0.5]}` | 0.05 |
| ema-seed-gated | `{"half_life":[64,128,256,384,512,768,1024,2048],"weight":[0.1,0.25,0.5]}` | 0.05 |
| best | `{"spacing":[8,32,128],"weight":[0.1,0.25,0.5],"window":[128,256,1024,4096]}` | 0.05 |
| mean | `{"count":[8,64],"falloff":["uniform","linear","gaussian"],"spacing":[8,32,128],"weight":[0.1,0.25,0.5],"window":[128,256,1024,4096]}` | 0.05 |
| seed | `{"weight":[0.1,0.25,0.5]}` | 0.05 |
| hit-lru-mean | `{"anchors":[1,2,4,8],"ema_mix":[0,0.5,0.9],"novelty":[0.02,0.05,0.1,0.2,0.3],"spacing":[128,256,384,512,768,1024],"weight":[0.1,0.25,0.5]}` | 0.05 |
| hit-lru-best | `{"anchors":[1,2,4,8],"ema_mix":[0,0.5,0.9],"novelty":[0.02,0.05,0.1,0.2,0.3],"spacing":[128,256,384,512,768,1024],"weight":[0.1,0.25,0.5]}` | 0.05 |
| delayed-ema-start0.25 | `{"delay":[0,16,32,64,128,256,512],"half_life":[128,256,512,1024,2048],"weight":[0.1,0.25,0.5]}` | 0.05 |
| delayed-ema-start0.1 | `{"delay":[0,16,32,64,128,256,512],"half_life":[128,256,512,1024,2048],"weight":[0.1,0.25,0.5]}` | 0.05 |

Normalized-mean A/B scored75 at three borrowed settings versus ordinary64/68/66.
It is included in the active refinement round, not selected as the default.
See [detailed report](embedding_multi_anchor_20260913.md) for build hashes,
timings, tests, controls, mechanism descriptions and limitations.
