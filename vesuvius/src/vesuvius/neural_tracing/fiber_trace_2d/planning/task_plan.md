# Stopped-run evaluation plan

1. Existing approved watcher targets activefinetuning launch andlastcomplete test;
verify current raw/EMA writes beforeboundedstop. Leave stopped.
2. Extend shared series catalogue with bestraw; explicit expected_training_state
alive(default) or dead for thisseries. Check it beforefreeze, before/aftereval;
unknown iserror. Preserve oldlivebehavior, no skipchecks or trainingrestarts.
3. Freeze4exact snapshots beforefirstGPUevaluation; freshoutputroot. Add previous
currentraw/currentEMA99/currentEMA999 to Classic andprevious bestpair comparisons.
Include ALL earlier evaluatedrows (NML/JSON/combined600k,early3EMA,bestpair,
previouscurrent3) directly asbaselines; no optional archival-only scope.
4. Run4serial managedexclusive evaluations andaudits, fulltables aftereach.
Record best rawselection testmetric/step and compare finetuning vsprior results.
5. User extension: freeze bestEMA99/bestEMA999 from stoppedfinetuning, then
evaluate both afterbestraw. New series uses all19 completedrows asbaselines.
No training restart; reuse existing runner and artifact audits unchanged.

## Spec update

Reporting refinement: replace the arithmetic aggregate with Classic's exact
arithmetic mean times the geometric mean of five model/Classic scroll ratios.
Require exactly one Classic row, positive baseline means, finite nonnegative
model means; zero model mean gives zero score. Keep all per-scroll metrics.
Test baseline identity, proportional gains, scroll-scale invariance of ranking,
and invalid baselines. Regenerate the completed EMA99 comparison without GPUs.
Use actual snapshot sample/update counts and weight kind in labels, not
early/previous/current ages. Mark finetuning explicitly; preserve provenance.
Also show selection reason (latest/current or best test loss), exact test metric
and its value; best selection is per raw/EMA track, not trace-score selection.

No training or inference math changes. Stopped-run mode explicit; previous alive
invariant retained bydefault. AllGPUwork usesexistingmanagerexclusiveadmission.

## Docs updates

Preserve reports/provenance/tasklog; explain stopped-run option and conclusions.

## Validation

CPU exactbestraw order/rawvalidation/stoppedstate/defaultalive regressions;
4full100direction GPUartifactaudits. Independent plan/finalreport review.

## Changelog

Record stoppedfinetuning evaluationresults.
