# Stable gap-aware BP experiments

Frozen raw40500 iso20 traces, CPU128, RelWithDebInfo. No inference/retracing. Replay100% inherited; local-event oracle/constraint metrics unavailable (existing calibration issue). Accuracy uses all26 references; shift uses observed N, missing omitted. Tied/unbounded readouts use deterministic rank1. Selection on this crop is tuning, not held-out validation.

Controls: overlapping baseline accuracy50.00%, shift83.33%; prior minimum64 incident tuned30.77%/54.17%; pure length tuned42.31%/62.50% at500,30.77%/58.33% at5000.

Cost=(1-alpha)*100*M + alpha*cL*L/128. Winding damping is the new-message fraction; prepass remains unchanged.

| Trial | Accuracy % | Shift % (N) | Disabled length % | Gaps interior/endpoint/whole | Boundaries | Components | Projection defects | Ties/unbounded | Missing | Seconds |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|
| default-control | 30.77% | 54.17% (24) | 11.901% | 56/111/441 | 223 | 25 | 2 | 2/2 | ['4', '12'] | 37.85 |
| a0-damp0p5-i500 | 30.77% | 54.17% (24) | 11.901% | 56/111/441 | 223 | 25 | 2 | 2/2 | ['4', '12'] | 39.25 |
| a0-damp0p25-i500 | 19.23% | 25.00% (24) | 16.952% | 120/200/463 | 440 | 35 | 0 | 4/4 | ['4', '12'] | 37.32 |
| a0-damp0p1-i500 | 19.23% | 20.83% (24) | 17.243% | 147/206/458 | 500 | 36 | 2 | 3/3 | ['4', '12'] | 36.14 |
| a0p25-damp0p5-i500 | 30.77% | 33.33% (24) | 16.179% | 141/162/469 | 444 | 36 | 1 | 4/4 | ['4', '12'] | 34.01 |
| a0p25-damp0p25-i500 | 19.23% | 20.83% (24) | 14.069% | 74/140/444 | 288 | 35 | 2 | 3/3 | ['4', '12'] | 35.61 |
| a0p25-damp0p1-i500 | 19.23% | 33.33% (24) | 17.069% | 69/178/465 | 316 | 49 | 0 | 4/4 | ['4', '12'] | 35.16 |
| a0p5-damp0p5-i500 | 26.92% | 37.50% (24) | 14.608% | 54/112/467 | 220 | 48 | 0 | 3/3 | ['4', '12'] | 34.77 |
| a0p5-damp0p25-i500 | 19.23% | 20.83% (24) | 18.097% | 98/201/477 | 397 | 68 | 0 | 4/3 | ['4', '12'] | 40.15 |
| a0p5-damp0p1-i500 | 19.23% | 20.83% (24) | 17.484% | 56/169/476 | 281 | 67 | 0 | 5/5 | ['4', '12'] | 45.95 |
| a0p75-damp0p5-i500 | 26.92% | 37.50% (24) | 15.082% | 35/106/475 | 176 | 49 | 0 | 3/3 | ['4', '12'] | 39.04 |
| a0p75-damp0p25-i500 | 19.23% | 16.67% (24) | 21.986% | 101/225/505 | 427 | 77 | 0 | 6/5 | ['4', '12'] | 33.57 |
| a0p75-damp0p1-i500 | 11.54% | 20.83% (24) | 18.368% | 37/145/496 | 219 | 68 | 0 | 5/5 | ['4', '12'] | 35.14 |
| a1-damp0p5-i500 | 42.31% | 62.50% (24) | 15.775% | 5/56/507 | 66 | 113 | 0 | 4/3 | ['4', '12'] | 37.14 |
| a1-damp0p25-i500 | 26.92% | 39.13% (23) | 22.356% | 141/148/544 | 430 | 312 | 0 | 3/2 | ['4', '8', '12'] | 38.07 |
| a1-damp0p1-i500 | 26.92% | 39.13% (23) | 20.859% | 51/124/543 | 226 | 220 | 0 | 3/2 | ['4', '8', '12'] | 37.22 |
| a1-damp0p5-i1000 | 30.77% | 58.33% (24) | 14.594% | 13/58/494 | 84 | 112 | 1 | 3/2 | ['4', '12'] | 60.84 |
| a1-damp0p5-i5000 | 30.77% | 58.33% (24) | 14.643% | 14/60/494 | 88 | 113 | 0 | 3/2 | ['4', '12'] | 234.56 |

Full solver fields, convergence diagnostics, ladders and parameter settings in adjacent JSON.
Artifacts: `/ephemeral/hendrik/fiber_crop_bench/20261002_iso20_stable_gap_v1`. Approved runner: `bash /home/hendrik/villa3/scripts/run_training_failure_mining`, action `iso20_stable_gap`.
