# Stable gap-aware BP follow-ups

Frozen raw40500 iso20, CPU128, RelWithDebInfo; no retracing. Replay100% inherited; local-event oracle and constraint-removal metrics unavailable due existing calibration failure. Accuracy/26; shift observed-only (N and missing shown), deterministic rank1 for ties/unbounded. Overlapping baseline50.00% accuracy/83.33% shift; prior length300/128+boundary1000 at500:42.31%/62.50%. Full parameters, solver objective/residual, repair stats and ladders in adjacent JSON. Trial key: a=length blend, d=winding update fraction, b=boundary cost, c=length cost per128basevx, i=iteration cap, m=minimum piece length, best=checkpoint retention, r=run-opening cost, s=gap-repair sweeps. Objectives are comparable only within identical cost settings, not across grid rows.

| Trial | Accuracy % | Shift % (N) | Disabled % | Interior/endpoint/whole gaps | Boundaries | Components | Projected defects | Ties/unbounded | Missing | Selected iter | Objective / final | Seconds |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---|---:|
| a0-d0p5-b1000-c300-i500-m64-best0-r0-s0 | 30.77% | 54.17% (24) | 11.901% | 56/111/441 | 223 | 25 | 2 | 2/2 | ['4', '12'] | 500 | 640725 / 640725 | 39.64 |
| a0-d0p5-b1000-c300-i500-m64-best1-r0-s0 | 30.77% | 54.17% (24) | 11.901% | 56/111/441 | 223 | 25 | 2 | 2/2 | ['4', '12'] | 500 | 640725 / 640725 | 38.12 |
| a1-d0p5-b1000-c150-i500-m64-best0-r0-s0 | 30.77% | 39.13% (23) | 24.463% | 69/98/577 | 236 | 256 | 0 | 4/3 | ['4', '8', '12'] | 500 | 343062 / 343062 | 36.58 |
| a1-d0p5-b1000-c300-i500-m256-best0-r0-s0 | 15.38% | 13.04% (23) | 16.939% | 0/22/511 | 22 | 153 | 0 | 1/1 | ['4', '8', '12'] | 500 | 219327 / 219327 | 23.45 |
| a1-d0p5-b1000-c300-i500-m64-best0-r0-s0 | 42.31% | 62.50% (24) | 15.775% | 5/56/507 | 66 | 113 | 0 | 4/3 | ['4', '12'] | 500 | 238003 / 238003 | 38.77 |
| a1-d0p5-b1000-c300-i500-m64-best1-r0-s0 | 42.31% | 62.50% (24) | 15.775% | 5/56/507 | 66 | 113 | 0 | 4/3 | ['4', '12'] | 500 | 238003 / 238003 | 37.16 |
| a1-d0p5-b1000-c600-i500-m64-best0-r0-s0 | 26.92% | 37.50% (24) | 19.753% | 162/226/476 | 550 | 264 | 2 | 4/3 | ['4', '12'] | 500 | 923324 / 923324 | 36.06 |
| a1-d0p5-b2000-c150-i500-m64-best0-r0-s0 | 30.77% | 39.13% (23) | 24.722% | 70/96/579 | 236 | 256 | 0 | 4/3 | ['4', '8', '12'] | 500 | 533705 / 533705 | 46.27 |
| a1-d0p5-b2000-c300-i500-m64-best0-r0-s0 | 30.77% | 39.13% (23) | 24.473% | 68/99/577 | 235 | 255 | 0 | 4/3 | ['4', '8', '12'] | 500 | 684660 / 684660 | 52.12 |
| a1-d0p5-b2000-c600-i500-m64-best0-r0-s0 | 42.31% | 58.33% (24) | 15.786% | 6/58/506 | 70 | 116 | 0 | 4/3 | ['4', '12'] | 500 | 482731 / 482731 | 37.61 |
| a1-d0p5-b500-c150-i500-m256-best0-r0-s0 | 15.38% | 21.74% (23) | 17.680% | 1/21/523 | 23 | 146 | 0 | 3/2 | ['4', '8', '12'] | 500 | 115343 / 115343 | 21.74 |
| a1-d0p5-b500-c150-i500-m64-best0-r0-s0 | 42.31% | 62.50% (24) | 16.054% | 7/47/507 | 61 | 109 | 0 | 4/3 | ['4', '12'] | 500 | 118989 / 118989 | 37.39 |
| a1-d0p5-b500-c150-i500-m64-best0-r0-s1 | 42.31% | 62.50% (24) | 15.962% | 6/40/507 | 52 | 109 | 0 | 4/3 | ['4', '12'] | 500 | 114651 / 114651 | 38.09 |
| a1-d0p5-b500-c150-i500-m64-best0-r0-s10 | 42.31% | 62.50% (24) | 15.962% | 6/40/507 | 52 | 109 | 0 | 4/3 | ['4', '12'] | 500 | 114651 / 114651 | 35.81 |
| a1-d0p5-b500-c150-i500-m64-best0-r0-s3 | 42.31% | 62.50% (24) | 15.962% | 6/40/507 | 52 | 109 | 0 | 4/3 | ['4', '12'] | 500 | 114651 / 114651 | 37.82 |
| a1-d0p5-b500-c150-i500-m64-best0-r1000-s0 | 30.77% | 25.00% (24) | 20.399% | 39/99/520 | 177 | 219 | 2 | 5/3 | ['4', '12'] | 500 | 722654 / 722654 | 38.69 |
| a1-d0p5-b500-c150-i500-m64-best0-r1000-s1 | 30.77% | 25.00% (24) | 20.064% | 26/68/520 | 120 | 217 | 2 | 5/3 | ['4', '12'] | 500 | 662211 / 662211 | 35.56 |
| a1-d0p5-b500-c150-i500-m64-best0-r1000-s10 | 30.77% | 25.00% (24) | 19.759% | 23/61/520 | 107 | 212 | 2 | 5/3 | ['4', '12'] | 500 | 646777 / 646777 | 37.28 |
| a1-d0p5-b500-c150-i500-m64-best0-r1000-s3 | 30.77% | 25.00% (24) | 19.759% | 23/61/520 | 107 | 212 | 2 | 5/3 | ['4', '12'] | 500 | 646777 / 646777 | 35.43 |
| a1-d0p5-b500-c150-i500-m64-best0-r250-s0 | 15.38% | 12.50% (24) | 19.799% | 45/116/521 | 206 | 216 | 0 | 3/3 | ['4', '12'] | 500 | 339970 / 339970 | 39.06 |
| a1-d0p5-b500-c150-i500-m64-best0-r4000-s0 | 19.23% | 13.04% (23) | 19.583% | 57/95/499 | 209 | 189 | 4 | 3/3 | ['4', '8', '12'] | 500 | 2.28646e+06 / 2.28646e+06 | 39.48 |
| a1-d0p5-b500-c300-i500-m64-best0-r0-s0 | 26.92% | 37.50% (24) | 20.018% | 170/231/478 | 571 | 271 | 0 | 4/3 | ['4', '12'] | 500 | 473836 / 473836 | 35.81 |
| a1-d0p5-b500-c600-i500-m64-best0-r0-s0 | 19.23% | 20.83% (24) | 13.469% | 140/167/414 | 447 | 180 | 0 | 3/3 | ['4', '12'] | 500 | 483752 / 483752 | 33.18 |

## Stage comparison (best measured configuration per stage)

| Method | Replay % | Constraint score % | Oracle accuracy % | Non-oracle accuracy % | Shift % (N) | Disabled length % |
|---|---:|---:|---:|---:|---|---:|
| Overlapping iso20 baseline | 100.00% | −25.24% | 80.77% | 50.00% | 83.33% (24) | — |
| Prior length300/boundary1000 | 100.00% | Unavailable | Unavailable | 42.31% | 62.50% (24) | 15.775% |
| Checkpoint selection | 100.00% | Unavailable | Unavailable | 42.31% | 62.50% (24) | 15.775% |
| Refined cost grid | 100.00% | Unavailable | Unavailable | 42.31% | 62.50% (24) | 16.054% |
| Run-opening prior | 100.00% | Unavailable | Unavailable | 30.77% | 25.00% (24) | 20.399% |
| Native gap repair (with/without run prior) | 100.00% | Unavailable | Unavailable | 42.31% | 62.50% (24) | 15.962% |

## Interval repair details

| Trial | Proposals | Accepted moves | Restored base vx | Disabled base vx | Objective before → after | Repair seconds | Cap hits / unvisited pieces |
|---|---:|---:|---:|---:|---|---:|---|
| a1-d0p5-b500-c150-i500-m64-best0-r0-s1 | 740 | 9 | 935.441 | 147.892 | 118989 → 114651 | 0.0138113 | 0 / 0 |
| a1-d0p5-b500-c150-i500-m64-best0-r0-s10 | 1450 | 9 | 935.441 | 147.892 | 118989 → 114651 | 0.0216808 | 0 / 0 |
| a1-d0p5-b500-c150-i500-m64-best0-r0-s3 | 1450 | 9 | 935.441 | 147.892 | 118989 → 114651 | 0.0204965 | 0 / 0 |
| a1-d0p5-b500-c150-i500-m64-best0-r1000-s1 | 1132 | 48 | 3447.13 | 841.33 | 722654 → 662211 | 0.0152587 | 0 / 0 |
| a1-d0p5-b500-c150-i500-m64-best0-r1000-s10 | 3064 | 59 | 5829.81 | 841.33 | 722654 → 646777 | 0.0296718 | 0 / 0 |
| a1-d0p5-b500-c150-i500-m64-best0-r1000-s3 | 3064 | 59 | 5829.81 | 841.33 | 722654 → 646777 | 0.0315154 | 0 / 0 |

## Completed stages

- 2
- 3
- 4
- 5

Artifacts:`/ephemeral/hendrik/fiber_crop_bench/20261002_iso20_stable_followups_v1`. Approved runner action `iso20_stable_followups`; control key `stage` selects2–5. Stage0 refreshes this report only, without running a solver. One run per setting; runtime is observational, not isolated microbenchmark.

## Conclusion

None of these stages beats the preceding42.31% accuracy/62.50% shift result, and the overlapping-piece baseline50.00%/83.33% remains better. Checkpoint selection chose the terminal state in both controls. The cost grid tied the previous scores with slightly fewer gaps. All positive run-opening penalties hurt these benchmark scores. Local repair reduces the optimized objective and boundaries but not winding errors. No experimental feature or parameter is promoted to a production default. The prior blend/damping screen is in2026-10-02-iso20-stable-gap.md. After the explicit user correction, all subsequent runs use500iterations; no more budget sweeps.
