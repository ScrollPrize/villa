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
## Interrupted boundary sweep: verified partial results

Follow-up CPU-only replay at
`/ephemeral/hendrik/fiber3d_train2/evaluations/embedding_ema99_boundary_cpu_verify_20260914_v1/`
confirmed support-softmin67 twice with the exact interrupted-run replay hash
`7be2e5ca221dc6cde71e28c85936a2cc813697a8588302205585843de4a03a84`.
LRU64 and both original controls also reproduced their earlier hashes. Thus the
67 result is now repeat-verified, although the interrupted full boundary search
itself remains incomplete. All replay variants used `fill_missing=false` and
inferred zero tiles; no GPU reservation was requested. Command:
`vesuvius/scripts/run_fiber3d_experiment embedding-trace-sweep` with the checked-in
CPU-verification control. Process exited0; final LRU best-repeat also reproduced
64 and hash `013d255be05b3b6fb295a14b4a9c92fffef2558b724179d00d8b8903ed1b81a0`.

### Failure-location comparison: LRU64 versus support-softmin67

Inspecting the alternative trace's matches within±32 base voxels around the
four LRU-only events gives:

| Direction/case | LRU failure arc | Softmin maximum error/threshold | Softmin segments in neighborhood |
|---|---:|---:|---|
| forward/0 | 2102.14 | 1.006 | 5,6 (nearby restart) |
| reverse/1 | 9676.86 | 0.829 | 26 |
| reverse/1 | 12656.47 | 0.343 | 33 |
| reverse/1 | 12792.75 | 0.858 | 33 |

Below1 means those saved matched points satisfy the actual ellipse threshold.
This supports three locally cleaner neighborhoods, not three independently
recoverable errors: softmin had restarted near12574.57 before the latter two,
so its different history may explain that advantage. There is no demonstrated
GT-free rule for switching policies at these locations.

CPU-only analysis of saved `refine-0000/cases/{0,1}/trace.json` and
`refine-0106/cases/{0,1}/trace.json` in the interrupted report directory.
Both cover the same reference arc range in each direction (~14891.716 base
voxels). Counts are29+35 versus32+35. Sorted failure reference arcs were matched
one-to-one, greedily pairing earliest endpoints within the stated radius;
otherwise advancing the smaller endpoint. No event is matched twice.

| Radius (base voxels) | Matched failures | LRU-only failures | Softmin-only failures |
|---:|---:|---:|---:|
| 32 | 57 | 7 | 10 |
| 64 | 60 | 4 | 7 |
| 128 | 61 | 3 | 6 |
| 256 | 61 | 3 | 6 |

At64 voxels, matching counts by direction are28/29 and32/35 of LRU's failures.
Thus most failures occur near the same reference locations; the second policy
is not broadly complementary despite its different appearance representation.
This weakens the case for adding a general mixture solely from aggregate scores.
Unmatched failures are **not** demonstrated recoveries or an oracle improvement:
restart locations change later trace history, and proximity tolerance is an
analysis convention rather than the evaluator's ellipse criterion. The next
useful check would inspect the few unmatched regions and their full paths before
designing a selector. No new traces, inference, or GPU reservation were used.

Inspected `embedding_ema99_anchor_boundaries_20260914_v1/report.json` after
the requested stop: 141 variant records and eight completed family searches.
The final repeat stage was not reached. These are tuning-fiber results using
EMA99 update6500, not independent validation or confirmed global optima.

| Family / starting point | Restarts | Anchors | Spacing (base voxels) | Weight |
|---|---:|---:|---:|---:|
| Novelty mean LRU | **64** | 2 | 512 | 0.10 |
| Novelty nearest LRU, own seed | 68 | 4 | 768 | 0.25 |
| Novelty nearest LRU, transferred seed | 72 | 2 | 2048 | 0.10 |
| Whole-path appearance, own seed | 70 | 1 | 384 | 0.25 |
| Whole-path appearance, transferred seed | 72 | 2 | 1536 | 0.10 |
| Support-softmin, own seed | 71 | 2 | 512 | 0.25 |
| Support-softmin, transferred seed | 67 | 2 | 1024 | 0.10 |
| Plain nearest LRU | 66 | 2 | 512 | 0.10 |

Novelty thresholds above are0.1; support-softmin temperatures are0.03; EMA mix
is zero. The best recorded variant remains64, so this partial search does not
justify replacing the previously repeated64-restart policy.

Next useful evidence, when sweeps resume: exact-repeat the improved67-restart
support-softmin candidate and compare its failure locations against the64
policy. Complementary recovered regions could motivate a new combination;
aggregate restart counts alone do not establish that. Finish pending families
only with the new demand-scoped manager handling. No GPU work was restarted
for this report.
