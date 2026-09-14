# Multi-anchor appearance model

- Prior EMA/best-history state committed053f76eff. Reference frozen EMA99 update6500/832000; winner EMA512weight.25=69 (28+41), exact repeat hash6e590851a9a00555dbc14edc3aa6f1fdccfc57a47b4a8c3c85a22ba900050484. Training continues. Reuse prediction cache.
- Plan independently approved: explicit nearest-anchor/EMA convex mixture, actual observed medoids, stable ties and bounded history. Implemented shared native helper, per-beam state and CLI; native/Python regression build running. Grid: K2/4/8, bank32spacing128 vs64spacing256, anchor-only vs half-EMA. No default changes.
- Review clarified anchor score uses parent-window bank, expiry on selected-child update (one-step grace). Explicitly documented, not claiming candidate-time strict filtering like mean/best. Existing expiry/isolation unit test exercises retained parent versus pruned child.
- Approved test runner: QuickBuildO3 build,2native suites pass1.57s and113Python pass38.69s; logs cooperative-tests/20260913_211352_827829. Anchor grid started via approved embedding-trace-sweep manager nonexclusive20GiB/16CPU.
- All12 anchor settings completed: short bank K2/4/8 standalone76/84/76,halfEMA83/78/84;long bank standalone87/78/82,halfEMA78/82/74. EMA control69 exact. Best anchor longK8 halfEMA74 still worse. Continue finer EMA half-lives256/384/640/768/1024 and lighter anchor contributions; also refine longK8 blend. No early stop at failed initial grid.
- User added next LRU anchor experiment; scheduled after refinement instead of stopping. Capacity4/8, admission128/512, nearest/average/EMA blend. Finer EMA results256/384/640/768/1024=74/72/87/84/73; no improvement yet. ShortK2 mixes.75/.9=91/76.
# Latest continuation

- Seventh boundary search complete:support-softmin transfer converged67 atK2/spacing1024/weight.10/T.03; improvement over prior71, final repeat pending. Plain nearestLRU boundary now running from66; sevenof15 searches complete, global64 unchanged.

- New provisional support-softmin improvement67 atK2/spacing1024/weight.10/T.03 (`refine-0106`), versus prior71. Found from transferred seed; active72112 continues refinement and exact repeat remains pending. Global64 unchanged.

- Sixth boundary search complete:support-softmin own-seed remains71 atK2/spacing512/weight.25/T.03. Alternative start follows; sixof15 complete, global64 and exact-repeat phase still pending.

- Fifth boundary search complete:whole-path transferred seed72, does not improve own-seed70. Support-softmin own-seed family now active72112; fiveof15 complete, overall64 unchanged and final repeats pending.

- Fourth boundary family complete:whole-path appearance own-seed converged70 atK1/spacing384/weight.25, versus previous72. AlternativeK2/.10 start now running (initial76). Fourof15 searches complete; final repeats pending, overall64 unchanged.

- Whole-path own-seed refinement now provisional70 atK1/spacing384/weight.25, improving72→71 withK1 then70 on next spacing cycle. Active72112; no new global best or exact-repeat claim yet.

- Third boundary search complete:nearest-novel transferredK2/.10 converged72 atspacing2048, worse than own-seed68. Largest tested spacing is a search boundary, not a proven optimum; this seed does not beat existing policy best. Whole-path appearance refinement follows on live72112; threeof15 complete.

- Boundary round second family complete:novelty nearestLRU improved71→68 atK4/spacing768/weight.25 (noEMA,novelty.1). Weight.30=79,.50=83; next transferredK2/.10 seed started. Overall64 unchanged; final exact repeats remain pending.

- Consolidated18 completed refinement entries with exact parameters/axes/hash prefixes and full report links in docs/embedding_refinement_summary_20260914.md; clearly pre-boundary, not a final claim. Active nearest-novelLRU improved71→68 atspacing768/K4/.25; still refining and final repeats pending.

- Boundary sweep72112 first family complete:novelty meanLRU remains64 atK2/spacing512/weight.10. Testedspacing192/256/384/768/1024/1536/2048, K1/4/8, weights.05/.15/.25/.50 without gain. K1=71 andK8=68; exact final repeat still pending with otherfamilies. Nearest noveltyLRU own-seed family now running.

- Independent completion audit: user-requested no-expiry appearance summaries/LRU/algorithmic variants and independent refinements covered. Remaining active15-family finite round+repeats, consolidated parameters/bounds/hashes and final state/docs. No diagnostic currently justifies conditional exemplar/reliability variants; retain as untested future hypotheses. Fiber is tuning data, not independent validation. Live72112 confirmed, first spacing192/256/384 trials did not improve64.

- Normalized A/B4047 exited0:10 replays, all old control hashes preserved, overall64 exactrepeat. Three normalized seeds75 each (different hashes) vs ordinary64/68/66. All6 coherence histograms sum to positivecounts; mean.9752–.9817, nonebelow.1. Started approved boundary/multi-start sweep72112 at `embedding_ema99_anchor_boundaries_20260914_v1`:12 prepared anchor families plus3 normalized families for fair per-method tuning. Training stays stopped; nonexclusive manager grant.

- Normalized A/B live4047: rebuilt64 hash exact; normalized same settings75. Plain mean68 also remains68, normalized75. Optimized-path coherence control verified7,233,138 candidate-parent states equals histogram total, mean.9751943, fractionbelow.1=0. This is candidate-parent weighting, not GT/selected-only distribution; no near-cancellation evidence in this control.

- Normalized-mean native implementation committed c60fe9bcf after native2 pass1.66s/Python118 pass32.63s (`cooperative-tests/20260914_071900_187504`). Started approved manager-aware A/B handle4047 at `embedding_ema99_normalized_mean_20260914_v1`; checks old64/68/66 controls plus normalized variants and coherence profiles. No training restart.

- New-method refinement66263 exited0:91 replays, five family winners72/66/68/69/71 and overall64 all exact-hash repeats. Restored full native-build/native-tests/Python suites and started approved regression launcher for normalized-mean; no rebuild overlapped the sweep.

- Revalidated live handle66263 after snapshot-location reply: four families finished (seed72, hit-mean66, hit-best68, delayed start.25→ordinary EMA69). Lower-weight delayed family still running, best71 at delay512/half1024/.10; final repeats pending. No native rebuild or concurrent cache writer while this sweep is active.

- Hit-only nearest local refinement completed68 (K2/.25/spacing768/novelty.10); three new-method families complete. Two delayedEMA starting seeds plus final repeats remain live66263; no earlier termination or manager bypass.

- Hit-only nearest refinement improved74→68 at K2/weight.25/spacing768/novelty.10; still refining. Checked parallel sweep feasibility:DemandStore takes nonblocking exclusive writer.lock, so concurrent extensions of this same frozen cache are not safe/useful; retain sequential sweeps, no duplicate GPU reservation.

- New-method refinement:seed-only settled72 at weight.30; hit-only meanLRU settled66 at originalK2/.10/spacing512/novelty.10 (unused window4096 equivalent). Remaining hit-nearest/delayed families still live66263. Normalized-mean A/B config prepared with old64/68/hit66 controls; not yet run.

- Independent normalized-mean review found profile counters after fused scoring's early return; moved accounting before scoring branch, serial/common to both paths. Renamed JSON count to candidate-parent states (includes candidates later rejected). Added constant-input profile assertions; require nonzero counters on real optimized eval after rebuild. Python118pass30.93s before this native-only follow-up (`20260914_063307_062873`).

- Implemented reviewed normalized-mean LRU aggregation (unbuilt while66263 runs):cached unit sum on bank changes, exact cancellation cost1; shared norm helper reused. Added candidate-weighted coherence histogram in existing trace profile (bins[0,.1),...,[.9,1),1) for both ordinary/normalized LRU, collected outside parallel scoring. Existing cost arithmetic unchanged; new math/cancellation tests and Python option coverage added. Must build/test and recheck existing64 hash after current sweep before evaluating it.

- Mechanism grid finished exit0,15replays,64 winner exact. Seed weights.10/.25/.50=80/76/82; delays32/64/128/256=86/78/80/78 vs EMA69; hit mean66/nearest74 vs existing64. All rebuilt controls and delay0 exact. New-method refinement started session66263 via approved launcher:seed,hitmean,hitnearest,delayedEMA two weight seeds. Tests retain nonexclusive manager allocation after user-requested training stop. Prepared anchor boundary sweep remains next.

- User requested stopping overfitting paired embedding training. Approved production-status identified the expected174110 launch; production-stop exited0 and verified root dead, manager task6f8a7826 finished with zero CPU/GPU allocation (generation67). Latest logged test update32000/4,096,000patches; do not infer final snapshot step from that log. Experiments continue under nonexclusive manager grants, frozen6500 EMA99 unchanged.

- New native methods built/tested via approved launcher:2native suites pass1.90s;117Python pass42.27s (`cooperative-tests/20260914_061209_344524`). Committed83e695caf/3b2f79701. Started seed/delay/hit comparison through exact approved experiment launcher, session47701, nonexclusive manager request. Baseline and zero-delay hashes will be checked before interpretation.

- Overnight all-family sweep finished exit0:270 replays, all13 family winners and global64 repeated identically (29+35). Other improved families windowed67, gated-seed73, best-history74, mean70. Native build/test now safe and started via exact approved runner. A nonessential manager-status command incorrectly triggered approval and delayed agent continuation overnight; do not repeat it or add approval-dependent diagnostics. Reuse only exact approved launchers/process handles for external actions.

- Six families complete: novelty meanLRU finished its finite local refinement at64 (`refine-0118`); no later axis/weight neighbor improved it. Exact repeats remain scheduled after all13 families. Python-only approved runner117passed44.55s (`cooperative-tests/20260914_000935_303937`), including new hit-touch validation. Native tests intentionally wait for active binary users.

- Implemented reviewed hit-only LRU recency as opt-in `--embedding-anchor-touch hit`: familiar observations touch/skip; novel observations do not refresh an unmatched anchor before LRU eviction. Default nearest unchanged. Native eviction/default/threshold tests and Python validation added. Unbuilt until active sweep finishes; no measured quality claim yet. This is a replacement-policy experiment, not a bug fix to the previously specified policy.

- New provisional global best `refine-0118`: novelty-gated meanLRU K2/spacing512/.10/novelty.10/noEMA =64 restarts. Family first improved75→68 by weight.10, then68→64 by K2. Continue axes/cycles and exact-repeat checks; not finished at first gain. Monitoring now reads best-so-far as well as last row so improvements between polls are not overlooked.

- Verified frozen cache geometry:sourceL1,tile256,outputscale4→embedding grid8base voxels. Winning512base admission equals full input tile width, not blend stride. Final local refinement must include intermediate/larger distances192/256/384/768/1024/1536/2048 and K1; initial LRU axes ended at512/K2 and cannot establish optimum beyond that boundary. No claim tile boundaries cause the result.

- MeanLRU family completed68 (K4/spacing512/.30/noEMA), second cycle found no better axes. K2/.30=70,mixtures.5/.9=71/89. Five families complete; nearestLRU66 still best, next novelty-gated meanLRU. Live sweep49134 verified throughout; no native rebuild.

- MeanLRU improved73→68 at weight.30 (`refine-0099`,K4/spacing512/noEMA) during fine .05 neighbor search. Further neighbors/cycles continue; nearestLRU66 remains global best. This confirms importance of refining nonwinning families rather than abandoning initial grids.

- Added post-round cross-seed check: transfer successful K2/spacing512/.10 anchor parameters to other policies, then refine if improved. Current novelty-LRU starts from71 and may reject an initially worse K2 move, missing a favorable K2×weight interaction. Do not equate a single-start coordinate minimum with maximal/global quality.

- Plain nearestLRU family completed at66 after three coordinate cycles:K2/spacing512/.10/noEMA. Weight.15=81,spacing32/.10=76,spacing128/.10=86; no further axis improvement. Four families complete; continue meanLRU next. Equivalent-scoring half-life128/2048 both reproduced66 (half-life128 exacthash checked); scheduled final exact-parameter repeats remain outstanding.

- Provisional66 replay hash68d07241e4ac50456fc1f1ff52263c9374ecfb4e7dde451097037dfa783fe172,41.98s. Actual68 trace segments:median112/mean469.41base voxels,53 shorter than512. However54.49% of total traced length is beyond the first512, so sparse histories are NOT equivalent to seed-only by distance; the planned explicit seed control remains necessary.

- New provisional global best `refine-0072`: plain nearestLRU K2 spacing512 weight.10 noEMA =66 restarts, versus EMA69. Found on second coordinate cycle after K4→K2 improvement. Exact-repeat still pending; continue refinement and all remaining families, do not stop at first gain.

- Verified live sweep49134 continuation:70 replays now; three families complete71/72/71. Plain nearestLRU improved73→72 with K2 (spacing512,.25,noEMA), so coordinate descent is cycling again around K2. Mixtures.5/.9 scored84/82; weight.30 scored77. No overall improvement over EMA69. No native rebuild while sweep active; staged seed/delay still awaits native validation.

- Reviewed conditional observed-exemplar appearance variant: unchanged clustering/admission, choose between merged clusters' observed representatives for scoring instead of synthetic centroid. Concrete representation hypothesis, not exact medoid; inspect centroid/exemplar gap before broad experiments. Scheduled only after seed/delay results, not implemented yet.

- Checked whether live-wrapper setup dominates sweeps: mechanism EMA69 candidate35.33s versus cached-only identical repeat37.78s (bothzero new tiles). No evidence for speculative cached-first rewrite; keep current validated execution path.

- All-family refinement: whole-path nearest appearance converged72 (K2 spacing512,weight.25,noEMA); nearby.20/.30 weights85/74. Next support-softmin family. Latest seed/delay Python116 tests44.27s passed (`20260913_225248_815482`).

- Independent methodology review identified sparse-history/seed-only confound. Added explicit fixed-seed mode as control for next delayed-EMA experiment (EMA+seed is not seed-only). No updates/expiry, shared existing cosine scoring, per-trace reset. Defer any embedding-independent ambiguity gate until evidence/margin measurements justify it.

- First all-family refinement complete: novelty-LRU nearest stays71 (K4 spacing512 novelty.1 weight.25, noEMA); denser8/32 worsened76/84. Whole-path appearance now refining; spacing8 worsened72 to82. Keep hypotheses tentative; no new global winner yet.

- Delayed implementation review passed semantics/plumbing; caught empty-deque allocation affecting all beams. Switched pending FIFO to an empty nonallocating vector (batch front erase), so unrelated modes gain no heap allocation. Python116 tests44.72s pass (`20260913_224408_775734`); native validation pending active sweep completion.

- Delayed EMA hypothesis independently reviewed: bounded4096 FIFO per beam, preflight delay/actual-step capacity check plus loud runtime overflow; seed active immediately, score parent then selected-child updates. Store original per-observation step distances to preserve EMA arithmetic; delay0 directly calls old helper. Typical delay32..256/step8 pending4..32. Implemented native helper/integration, CLI and sweep option/tests; native rebuild deliberately waits for active13-family sweep to retain fixed binary. Python-only test runner used meanwhile. Dense noveltyLRU8/32 scored76/84 vs sparse71: contamination remains a hypothesis, not established causality.

- Mechanism24-replay grid complete, EMA69 repeats exactly. Best noveltyLRU71 and support-softmin71; no global win. Started `embedding_ema99_all_refine_20260913_v1`:13 independent families, dense admissions first and .05 neighbor weight descent. Per-family progress now persisted as each finishes; updated Python116 tests39.16s passed (`20260913_222829_257218`). Training remains manager-cooperative/nonexclusive; no model or geometry changes.

- Important diagnostic: whole-path72-restart trace has74 segments, actual traced lengths mean432.11/median144.00/max6088.02base voxels. At spacing512,61/74 end before first admission after seed (mean1.55 observations); spacing12836/74 (mean3.84); spacing32only1/74 (mean14.07). Initial grids undersample anchor updates on this restart-heavy case. Prioritize spacing8/32/64 for all anchor-family refinement; do not infer poor appearance-memory potential from sparse initial trials. Calculated from actual trace_points_base_xyz, not GT lengths.

- Mechanism implementation committed1425f53c5 after native2/Python116 tests; mechanism sweep running. Fine .05 weight neighbors now integrated into independent family coordinate cycles (opt-in), with up-front tick validation; Python116 passed31.81s (`20260913_221326_604149`). This checks neighboring weights even if a coarse axis winner lies between tested endpoints.

- Whole-path initial grid complete: K2spacing12881/78,51272/81; K4spacing12878/83,51275/84; K8spacing12884/84,51276/84 (nearest/halfEMA). EMA69 repeat exact. Committed initial implementations/harness844856518.
- Novelty LRU and support-softmin implemented and independently reviewed; native tests pass1.59s, Python suite running. Support-softmin uses stable expm1/log1p for large temperatures after review; normalization/split/limits tests included. Mechanism grid prepared; then all-family refinement remains required.

- Initial LRU complete: K4/spacing128 nearest/mean/halfEMA75/88/77; K4/51273/73/81; K8/12876/87/79; K8/51275/80/84. EMA69 repeat exact. Whole-path build fixed seed configuration scope; native2 tests1.95s/Python11545.24s pass (`20260913_215750_020249`). Whole-path12-setting grid launched next.

- LRU native2/Python113 tests passed; sweep running under nonexclusive manager, controls reproduced. Whole-path spherical-summary implementation independently reviewed, no blocking issues; tests added for support conservation, old appearance, cancellation/recovery, capacity and beam isolation. Build deferred until LRU finishes to keep its native binary fixed.
- Added finite per-family coordinate-refinement helper for subsequent sweeps; each family starts from its own best seed, strict improvement and memoized candidate evaluations. Tests exercise multiple cycles and retaining nonwinning families. User additionally requests algorithmic refinements, not only parameters; novelty-aware LRU admission and support-aware aggregation are next mechanism hypotheses.

- Refinement complete: EMA half-lives256/384/640/768/1024 scored74/72/87/84/73 restarts. Additional windowed-anchor mixtures scored91/76/80/88/85/83/73. Control EMA512 remains69 and repeated identically. No improvement; proceed to LRU.
- User correctly identified the rolling-window restriction as deviating from whole-path descriptive/diverse anchors. After LRU, remove that restriction using bounded streaming appearance support; do not present windowed results as fulfilling that requirement.
