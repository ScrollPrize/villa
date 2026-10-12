# Fresh state-LR and width-LR follow-up

User requests another10x state LR with positionweight0.5, then10x/0.1xwidth
factor on whichever stateLR givesbetterrefL1. Interpretedwidthfactor aswidth
optimizer LR, disclosed beforelaunch: current2 ->20/.2, no lossweightchanges.
All trials fresh originalinitialization, not resumes; same10k sampledpass
budget and1kdiagnostics, globalbatch24576,sixexclusiveGPUs,seed0.

1. Run stateLR0.7348469228349533,widthLR2 versus savedstateLR.07348469228349533.
2. Select lower final10k referenceL1 (reportretention/cohort caveat too).
3. Run widthLR20/.2 with selectedstateLR, all otherparams unchanged.
4. Report fullprogress tables includingloss,L1,within2,retention,sign,widthfactor,
   fitted/referenceq10q90; persistcommands/results. Sameobjective allowsloss
   comparison intheseLRtests. No production code/default change.

Validation: independentconfigreview, existingtest, productioncompiledsixrank
checks, finite final states. Reuseapprovedrunner. Spec update: none.
Docs: task/status/log,results uv_bucket_lr_results.md,README/changelog.
