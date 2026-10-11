# Larger-batch LR comparison

- User requested documenting the scaling rule and committing. README now states
  lr(B)=.03*sqrt(B/4096), only for state optimizer; width_lr remains2.
  Evidence limited to16k trial,8k/32k rates explicitly extrapolated; no automatic
  config/default changes. Documentation-only follow-up; diff check validation.

- All3completed exit0. LR.03control exactly reproduced prior16kquality.
- LR.03/.06/.12:L1=11.745175/11.660019/11.571921;retainedlength66.6851/
  65.4857/62.1046%;signerrors3.6633/3.1654/3.3124%;seconds397.86/414.58/435.26.
  Peakallocated7.430GiB each. Recommend.06asbalancedcandidate,defaultsunchanged.
- Fullprogress,command,metriccaveats and4kcomparison:uv_lr16k_results.md.

- User authorized3parallel GPUs; plan uv_lr16k_plan.md.
- B16384 LR.03/.06/.12; width_lr2 unchanged, no solver code changes.
- Independent review approved configs; exclusive task
  a2ea6a69-1a58-490f-bb58-9a37907c672b granted all3GPUs.
- Same freshinitialL1=12.2423449311,within2=12.991196442%,offset+12.
  HigherLRinitially reduces sign errors faster but rejects length sooner.
- Freshseed0CCW151,4032updates each,matched data budget. Exclusive manager
  parent reserves3GPUs,24CPU,32GiB/GPU; existing approved runner.
