# Finetuning continuation plan

1. Use existing approved production-watch bound to active launch/root, checking
complete currentraw/EMA snapshot serialization before boundedstop. No unrelatedsignals.
2. Derive dedicated config from exact launch config; onlyLR0.01→0.001,runlabel,
resume currentraw path change. Existing restore applies configLR afteroptimizerload.
3. Use existing approved production-start after originalroot confirmeddead. Verify
resume step,optimizerLR,EMAcontinuation andmanager-running/logprogress.

## Spec update

None; no trainingcode or numerical changes beyond requestedLR. Rawresume retainsEMA.

## Docs updates

Record stopstep/latestcheckpoint,newconfig,launch/log andLRverification in tasklog.

## Validation

Read-only configdiff; existing launcher regressiontests; requestedrealmanagedrun
startup validates restoration and learningrate. Independent planreview.

## Changelog

Record finetuning launch and0.001LR in docs.
