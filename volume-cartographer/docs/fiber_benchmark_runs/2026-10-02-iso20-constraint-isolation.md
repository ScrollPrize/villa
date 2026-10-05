# Constraint isolation on old overlap512/128

Frozen raw40500 iso20 Paris4 1024³ crop [10240,22016,6144]–[11264,23040,7168], hendrik_crop1,128CPU workers, RelWithDebInfo,500iterations. No inference/retracing or repair. Replay100% inherited for all rows. Old oracle80.77%, constraint score−25.24% retained as historical baseline; this first isolation matrix measures independent readout, not new oracle/removal runs. Accuracy/26; observed shift excludes missing refs and reports N; tied rank1 is not unique.

| Variant | Non-oracle % | Shift % (N) | Disabled length % | Pieces | Changed prepass / common | Missing refs | Seconds |
|---|---:|---|---:|---:|---|---|---:|
| baseline | 50.00% | 83.33% (24) | 10.49% | 2732 | 0/2732 | ['4', '12'] | 48.3 |
| baseline-frozen | 50.00% | 83.33% (24) | 10.49% | 2732 | 0/2732 | ['4', '12'] | 51.1 |
| baseline-rebuilt | 50.00% | 83.33% (24) | 10.49% | 2732 | 0/2732 | ['4', '12'] | 51.6 |
| binary-frozen | 53.85% | 79.17% (24) | 11.42% | 2732 | 0/2732 | ['4', '12'] | 50.9 |
| binary-score | 53.85% | 79.17% (24) | 11.42% | 2732 | 1/2732 | ['4', '12'] | 49.6 |
| certify | 26.92% | 62.50% (24) | 22.78% | 2732 | 0/2732 | ['4', '12'] | 42.5 |
| certify-frozen-mass | 30.77% | 70.83% (24) | 16.29% | 2732 | 0/2732 | ['4', '12'] | 33.3 |
| closest | 46.15% | 70.83% (24) | 14.89% | 2732 | 0/2732 | ['4', '12'] | 51.6 |
| closest-certify | 23.08% | 62.50% (24) | 23.64% | 2732 | 0/2732 | ['4', '12'] | 37.8 |
| closest-frozen-mass | 46.15% | 70.83% (24) | 15.97% | 2732 | 0/2732 | ['4', '12'] | 48.8 |
| events-frozen-classes | 34.62% | 29.17% (24) | 27.70% | 2732 | 0/2732 | ['4', '12'] | 21.5 |
| events-frozen-classes-mass | N/A | N/A (None) | 12.72% | 2732 | 0/2732 | ['4', '12'] | 21.6 |
| events-frozen-mass | 38.46% | 58.33% (24) | 15.44% | 2732 | 386/2732 | ['4', '12'] | 22.2 |
| events-median | 34.62% | 33.33% (24) | 23.75% | 2732 | 386/2732 | ['4', '12'] | 21.8 |
| events-no-winding-cert | 15.38% | 33.33% (24) | 33.77% | 2732 | 369/2732 | ['4', '12'] | 27.5 |
| events-normalized-one | 19.23% | 20.83% (24) | 31.70% | 2732 | 386/2732 | ['4', '12'] | 19.4 |
| events-overlap | 34.62% | 29.17% (24) | 27.70% | 2732 | 386/2732 | ['4', '12'] | 21.6 |
| events-refine0 | 19.23% | 8.33% (24) | 25.01% | 2732 | 386/2732 | ['4', '12'] | 18.1 |
| events-replicated | 11.54% | 12.50% (24) | 31.22% | 2732 | 386/2732 | ['4', '12'] | 26.2 |
| events-unit | 15.38% | 4.17% (24) | 25.27% | 2732 | 386/2732 | ['4', '12'] | 21.6 |
| events-unit-one | 15.38% | 8.33% (24) | 25.61% | 2732 | 386/2732 | ['4', '12'] | 19.6 |
| fine64-old-readout | 15.38% | 4.17% (24) | 12.92% | 8273 | 386/588 | ['4', '12'] | 32.5 |
| fine64-tuned-old-readout | 38.46% | 45.83% (24) | 16.05% | 8273 | 386/588 | ['4', '12'] | 33.3 |
| local-scorer-closest-off | 15.38% | 8.33% (24) | 29.71% | 2732 | 377/2732 | ['4', '12'] | 16.3 |
| local-scorer-closest-on | 15.38% | 8.33% (24) | 17.15% | 2732 | 401/2732 | ['4', '12'] | 14.0 |
| local-scorer-frozen-both | N/A | N/A (None) | 21.90% | 2732 | 0/2732 | ['4', '12'] | 19.6 |
| local-scorer-frozen-classes | 19.23% | 16.67% (24) | 23.94% | 2732 | 0/2732 | ['4', '12'] | 22.0 |
| local-scorer-frozen-mass | 19.23% | 20.83% (24) | 24.46% | 2732 | 377/2732 | ['4', '12'] | 20.1 |
| local-scorer-median-off | 19.23% | 16.67% (24) | 23.94% | 2732 | 377/2732 | ['4', '12'] | 18.4 |
| local-scorer-median-on | 15.38% | 8.33% (24) | 17.47% | 2732 | 401/2732 | ['4', '12'] | 13.4 |
| old-boundary500 | 50.00% | 83.33% (24) | 10.39% | 2732 | 0/2732 | ['4', '12'] | 52.8 |
| old-length150-boundary0 | 46.15% | 79.17% (24) | 8.16% | 2732 | 0/2732 | ['4', '12'] | 56.9 |
| old-length150-boundary500 | 53.85% | 79.17% (24) | 10.18% | 2732 | 0/2732 | ['4', '12'] | 49.2 |
| reference-events | 46.15% | 83.33% (24) | 10.49% | 2732 | 0/2732 | ['4', '12'] | 46.3 |

## Row definitions

- **baseline:** Old overlapping baseline, explicit old reference extraction.
- **baseline-frozen:** Frozen-prior zero-change control.
- **baseline-rebuilt:** Rebuilt supplemental binary, old baseline.
- **binary-frozen:** Binary scores, fixed baseline classes and disabling masses.
- **binary-score:** Old measured geometry/distances, binary rather than soft orientation scores.
- **certify:** Old candidates/scorer; distance estimator and/or winding probe check.
- **certify-frozen-mass:** Consistency rejection only, original disabling costs.
- **closest:** Old candidates/scorer; distance estimator and/or winding probe check.
- **closest-certify:** Old candidates/scorer; distance estimator and/or winding probe check.
- **closest-frozen-mass:** Closest estimator only, original disabling costs.
- **events-frozen-classes:** Full new events, frozen baseline classes.
- **events-frozen-classes-mass:** Full new events, frozen baseline classes-mass.
- **events-frozen-mass:** Full new events, frozen baseline mass.
- **events-median:** Full source events with median estimator.
- **events-no-winding-cert:** Full source events without winding consistency rejection.
- **events-normalized-one:** Normalized evidence, nearest event per owner pair.
- **events-overlap:** Full new source measurements on exact old overlap layout.
- **events-refine0:** Full source events without transition refinement.
- **events-replicated:** All overlapping owners; conserved total event mass.
- **events-unit:** Same as events-overlap, unit evidence.
- **events-unit-one:** Unit evidence, nearest event per owner pair.
- **fine64-old-readout:** Fine64 original costs, old reference extraction.
- **fine64-tuned-old-readout:** Fine64 tuned costs, old reference extraction.
- **local-scorer-closest-off:** Old piece-pair candidates; local geometric scorer.
- **local-scorer-closest-on:** Old piece-pair candidates; local geometric scorer.
- **local-scorer-frozen-both:** Local geometric scorer only, frozen baseline both.
- **local-scorer-frozen-classes:** Local geometric scorer only, frozen baseline classes.
- **local-scorer-frozen-mass:** Local geometric scorer only, frozen baseline mass.
- **local-scorer-median-off:** Old piece-pair candidates; local geometric scorer.
- **local-scorer-median-on:** Old piece-pair candidates; local geometric scorer.
- **old-boundary500:** Old graph, boundary prior only.
- **old-length150-boundary0:** Old graph, coverage-partitioned physical-length unary.
- **old-length150-boundary500:** Old graph, coverage-partitioned physical-length unary.
- **reference-events:** Readout-only: certified source reference events.

## Controls and caveats

- All graph rows force the legacy reference extraction independently of graph options. Reference-events is the deliberate readout-only exception; initial states must exactly match baseline.
- Local scorer bundles short one-sided tangents, reciprocal projections and geometric probe checks. Winding consistency and closest-vs-median are separate switches. Refined points outside their old owning piece are rejected.
- Event-overlap layout exactly matches old intervals before any graph selection. Unique owner is last-start covering interval; replicated case conserves mass across all covering owner pairs, not historical full-mass duplication.
- Event multiplicity/weights affect both factors and incidence-based disabling cost. Frozen-mass/classes controls explicitly separate these coupled effects. Frozen identities must match original trace and exact source arc endpoints.
- Disabled length splits overlapping arc coverage equally, not double-counting physical length. This is a coverage-normalized fraction, not union of all disabled intervals.
- Reference observations use the old method but necessarily depend on current surviving pieces. Changed reference coverage is reported; this is not identical cross-evidence across different layouts.
- N/A means multiple independent gauges prevented a single comparable reference ladder; all gauge summaries are retained in JSON, not silently combined.
- One run per variant, wall times observational. No production-default changes. Full command, binary/shared-library and input hashes live in each run.json.

Artifacts: `/ephemeral/hendrik/fiber_crop_bench/20261002_iso20_constraint_isolation_v1`. Approved runner action `iso20_constraint_isolation`; report_only refresh supported.

## Prepared ordinary measurement inventory

Excludes same-source continuation. Includes prepared observations subsequently pruned from the active fixed-orientation grid; these are not active solver-factor counts. Pair comparisons require identical piece identities, not fine layouts. Mean targets are evidence-weighted quantized signed targets.

| Variant | Measurements | Evidence mass | Pairs | Lost baseline pairs | Changed pair class | Target changes >0.25 | Contradictory hard-sign pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 242762 | 242762.00 | 242762 | 0 | 0 | 0 | 0 |
| baseline-frozen | 242762 | 242762.00 | 242762 | 0 | 0 | 0 | 0 |
| baseline-rebuilt | 242762 | 242762.00 | 242762 | 0 | 0 | 0 | 0 |
| binary-frozen | 242762 | 242762.00 | 242762 | 0 | 0 | 0 | 0 |
| binary-score | 242762 | 242762.00 | 242762 | 0 | 0 | 0 | 0 |
| certify | 174731 | 174731.00 | 174731 | 68031 | 0 | 0 | 0 |
| certify-frozen-mass | 174731 | 174731.00 | 174731 | 68031 | 0 | 0 | 0 |
| closest | 243872 | 243872.00 | 243872 | 0 | 0 | 1457 | 0 |
| closest-certify | 174765 | 174765.00 | 174765 | 68031 | 0 | 43 | 0 |
| closest-frozen-mass | 243872 | 243872.00 | 243872 | 0 | 0 | 1457 | 0 |
| events-frozen-classes | 238464 | 103383.89 | 71230 | 171612 | 0 | 999 | 1 |
| events-frozen-classes-mass | 238464 | 103383.89 | 71230 | 171612 | 0 | 999 | 1 |
| events-frozen-mass | 238464 | 103383.89 | 71230 | 171612 | 0 | 999 | 1 |
| events-median | 238448 | 103367.89 | 71215 | 171619 | 0 | 994 | 1 |
| events-no-winding-cert | 264928 | 122132.26 | 79854 | 162992 | 0 | 1771 | 2 |
| events-normalized-one | 71230 | 42701.32 | 71230 | 171612 | 0 | 671 | 0 |
| events-overlap | 238464 | 103383.89 | 71230 | 171612 | 0 | 999 | 1 |
| events-refine0 | 168855 | 102670.02 | 70362 | 172480 | 0 | 998 | 1 |
| events-replicated | 397176 | 103383.89 | 103348 | 139510 | 0 | 1351 | 1 |
| events-unit | 238464 | 238464.00 | 71230 | 171612 | 0 | 998 | 1 |
| events-unit-one | 71230 | 71230.00 | 71230 | 171612 | 0 | 671 | 0 |
| local-scorer-closest-off | 72735 | 72735.00 | 72735 | 170132 | 0 | 1100 | 0 |
| local-scorer-closest-on | 61332 | 61332.00 | 61332 | 181530 | 0 | 527 | 0 |
| local-scorer-frozen-both | 72715 | 72715.00 | 72715 | 170139 | 0 | 1081 | 0 |
| local-scorer-frozen-classes | 72715 | 72715.00 | 72715 | 170139 | 0 | 1081 | 0 |
| local-scorer-frozen-mass | 72715 | 72715.00 | 72715 | 170139 | 0 | 1081 | 0 |
| local-scorer-median-off | 72715 | 72715.00 | 72715 | 170139 | 0 | 1081 | 0 |
| local-scorer-median-on | 61312 | 61312.00 | 61312 | 181537 | 0 | 523 | 0 |
| old-boundary500 | 242762 | 242762.00 | 242762 | 0 | 0 | 0 | 0 |
| old-length150-boundary0 | 242762 | 242762.00 | 242762 | 0 | 0 | 0 | 0 |
| old-length150-boundary500 | 242762 | 242762.00 | 242762 | 0 | 0 | 0 | 0 |
| reference-events | 242762 | 242762.00 | 242762 | 0 | 0 | 0 | 0 |

## Findings

- Old baseline, rebuilt baseline, frozen-prior baseline, and readout-only initial states match exactly.
- New reference readout alone changes50.00% to46.15% accuracy; shift remains83.33%. It is not the main regression.
- Local scorer on old candidates loses 70.08% of baseline piece-pair links; accuracy19.23%/shift16.67%. Freezing old classes gives identical scores; freezing old disabling mass gives19.23%/20.83%.
- Winding consistency alone removes 28.02% of baseline links; 26.92%/62.50%, or30.77%/70.83% with frozen costs. Closest estimator also regresses46.15%/70.83% even with frozen costs.
- Full source events have 238,464 measurements but only 71,230 unique owner pairs versus 242,762 baseline pairs: repetition is not equivalent to coverage. Frozen disabling mass helps34.62%/29.17% to38.46%/58.33%, but does not recover baseline.
- Binary orientation scores alone are not harmful here:53.85%/79.17%, also with frozen classes/costs. Old graph + physical-length150/128 + boundary500 gives the same aggregate scores (not necessarily the same ladder). Each gains one exact reference but loses one shift-score step versus baseline; neither dominates both metrics.
- Fine64 tuned with old reference readout remains38.46%/45.83%. The new measurement rules regress before changing the layout, so these results do not establish that fine splitting itself is inherently worse.
- Scope limitation: local tangent, reciprocal projection and geometric probes remain one bundled scorer; this matrix localizes its regression but does not isolate every internal geometric test. New oracle/removal metrics were not run. Two frozen-both controls have multiple active gauges and no single comparable ladder.
- No production defaults changed. Tested on one tuning crop, not held-out validation. 445 native cases and45 Python checks passed. Native solve times are individual runs, not repeated speed benchmarks.

Reproduction: set scripts/training_failure_mining_control.json to action iso20_constraint_isolation (supplement:true for supplemental controls; analyze:true for saved CSV analysis; report_only:true for report). Run `bash /home/hendrik/villa3/scripts/run_training_failure_mining`. Build/test uses the same runner with action build.
