# EMA integration experiments — 2026-09-15

User requests all proposed cadence/range/nonlinear-integration options. Prior fixed-path implementation/results archived. Defaults remain unchanged. Intended matrix20EMA cadence/start +16finite-window +3starts each gate/angular/dual; GPU fit and actual full evaluation of each family winner. Same frozen model roles and content-grouped split. Native/PyTorch parity required; fitting warm-up extended4096 to support2048windows. Shared helpers will be extracted from existing fixed-path fitter, not copied.

## Implementation / fit completed

- Independent plan and implementation review by integration_review: causal update ordering, detached/padded histories, physical cadence/window expiry and selection-only ranking verified. Fixed identified antipodal native/Torch difference; exact antipodal update now retains prior reference in both paths, including a cap exceeding pi.
- Shared native EMA blend helper, opt-in integration families, native vector audit entry point, native presence export. Reused existing finite-window history helper. Shared fit preparation/loss/metrics extracted; split report shared in replay_metrics. Original default behavior preserved.
- New dataset embedding_integration_data_20260915_v1:148windows,29failure119control, same30/9/11fiber split;76/38/34windows across fit/selection/test. Extended warm-up4096 without moving original control branch locations. No coverage loss.
- GPU matrix45candidates/150steps each complete; exclusive GPU0 through existing manager. Native real-vector parity270paths,max absolute weighted-score error2.12950414e-7. Checkpoint identity includes dataset, matrix, implementation and shared-fit hashes. Repeat fit after identity hardening reproduced parameters.
- Workspace RelWithDebInfo GCC13 build passed;23replay+84trace native tests and230Python tests passed,8optional skips. Added causal cadence, zero-gate clock, angular/antipodal, dual branch isolation, window expiry and padded unequal warm-up tests.
- Fitting output /ephemeral/hendrik/fiber3d_train2/evaluations/embedding_integration_fit_20260915_v1/{matrix,parity}.json. Five finalist configs generated from the frozen mixed-current baseline, changing only output/label and embedding trace flags. Full evaluations next; no default promotion and no rendering.
- Preliminary surrogate: all family winners use spacing8; finite window256. Heldout full-window GT preference remains0%; actual tracing required, not a claimed improvement.

## Full evaluation progress

- All five finalist prepare/validate-full commands passed; queued `full` on the same reusable approved runner, exclusive64CPUs/all8GPUs each, serialized by manager. Inference caches reused; new off-route demands may compute additional tiles, no cache deletion.
- Cadenced EMA full180directions:117restarts,58.200231mm mean,native285.595s. Finite window full180directions:111restarts,61.318100mm mean,native300.391s. Prior fitted EMA110/61.871mm. Neither improves prior fit.
- Exact window-identity and scored-tail comparison:148same keys and zero differences in branch embeddings, physical distances or base costs after extending warm-up. Shared-fit hash current. Independent re-review approves native/math/configs after antipodal fix.
- Final reporting hardening validates equal per-scroll coverage and metric definitions, adds commands; latest regression231Python pass8skip,107native cases pass. Gate/angular/dual full evals pending/running.
- Gate full180directions completed:110restarts,61.870516mm mean, ties previous fitted EMA. Angular and dual remain queued. No promotion based on tie.
- Angular full180directions completed:115restarts,59.203683mm mean, worse than previous fitted EMA. Gate trade-off:0139/0332/1299/1451/Paris4=50/4/8/0/48 versus previous46/4/7/0/53; heldout both20. Finite window heldout19 but111overall. Dual remains last evaluation.

## Completed outcome

- All five full runner commands exited0. Dual113restarts/60.242344mm mean. All50fibers/90eligible spans/180directed runs/6867.627253directedmm; report validates identical per-scroll coverage and protocol. No evaluation from this experiment remains running.
- Final aggregate (restarts/mean/median/P05/P95 mm):EMA117/58.200/10.011/.232/102.469;window111/61.318/10.490/.251/89.563;gate110/61.871/10.685/.228/96.830;angular115/59.204/10.507/.262/102.700;dual113/60.242/9.984/.237/108.852. Gate ties previous110restarts, no overall improvement; no default promotion.
- Fit/selection/heldout counts:EMA58/38/21,window57/35/19,gate53/37/20,angular56/37/22,dual56/35/22. Previous fitted56/34/20. Original55/35/21. Off59/36/18. Small cohort and surrogate mismatch preclude significance claims.
- Native/coordinator seconds:EMA285.595/359.027,window300.391/371.988,gate285.558/356.595,angular286.323/355.875,dual286.845/357.936. One full evaluation each, cache misses differ, not controlled throughput benchmarks.
- Final Markdown+JSON saved under repo docs/benchmarks/embedding_integration_20260915 and ephemeral embedding_integration_fit_20260915_v1/comparison. Includes all45candidate starts/fitted settings, full scroll and split tables, exact commands, effective native params, hashes and parity. No visualization or commit requested.
- Final independent review passed:recomputed all9summaries against report, verified hashes, coverage and split sums; no blockers. Gate selection split worsens34→37,heldout ties20; no generalized improvement claim. Task complete.
