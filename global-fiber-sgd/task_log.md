# Balanced classes and fixed sign charge

User requests fixed sign cost enabled, class balancing of all constraint terms,
and aggregate perpendicular:parallel weight10:1. Prior task/results committed
866b13494; historical result tables remain in dedicated results documents.

Plan independently reviewed. Chosen normalized mean event weight1; no class
rebalance per batch/rank. Current graph weights: E/(11P),10E/(11Q).
Initial multiplier16 proposal failed local rejection dominance after parallel
downweighting; reviewer caught threshold>17.1 at other-endpoint retention50%.
Selected/disclosed32. Retain sign hinge10, all other parameters, original
initializations and old hard-readout protocol. No assertion of global feasibility.

Implemented opt-in full-graph weights multiplied into sampling weights before
all pair terms and event phase priors. Unary and continuation terms unchanged.
Unsupported categorical cost-target mode rejects balancing explicitly; no
silent partial objective. Default configs preserve original objective.
Production distributed validation now compares the charged/class-weighted
compiled gradients to concatenated eager gradients for this configuration.

Validation:47 tests passed (class balance, sign charge, distance objectives,
pair backend), command:
`/home/hendrik/.venv_las/bin/python -m pytest -q --noconftest -o pythonpath=/home/hendrik/villa2/vesuvius/src vesuvius/tests/test_fiber_uv_class_balance.py vesuvius/tests/test_fiber_uv_sign_charge.py vesuvius/tests/test_fiber_uv_distance_objectives.py vesuvius/tests/test_fiber_uv_pair_backend.py`

Prepared run_uv_balanced10000_{fresh,reference,suite}.json. Same audited subset,
six GPUs,10080 updates,diagnostics1008, seed0. Implementation review approved;
production launched with approved runner. Scalar loss changes definition; compare geometric
quality and unweighted class-specific signs, not raw old/new losses alone.

User requested stop and10x/0.1xLR comparison. Interrupted suite session47802
withSIGINT; exit130. Manager4495f33b-0832-493e-afc7-8f23bb17f658 finished,
noheldresources. Fresh completed10k,reference lastdiagnostic2k withno final
snapshot. Preserve outputs. Newfourtrial suite only changesstateLR,notwidthLR2;
same originalinitializations,10k each,sixGPUs. User informed of widthLR choice.
Broader regression33passed; total80tests passed before production. Charged
compiled sixrankpreflight passed for both starts. Independent LRconfigreview approved.
Launched run_uv_balanced_lr_suite.json with the approved runner. Manager task
b697d6ed-6481-4190-86c9-351967569b8e, four sequential sixGPUtrials underonelease.

User asked width coupling while trials run. Explained width factor diagnostic
only; widths convert H arc length to winding fraction, affecting pair distance,
sign,phase andphysical-Ucontinuity. Priors .1relative radial deviation and .01
neighbor ratio smoothing averaged over151widths; referencewidths initonly.
Proposed (not implemented) per-H-piece positive localturns/voxel slope shared
acrosshypotheses, weakrobust compatibility withglobalwidth, localq/Vcontinuity
to avoidreintroducingglobalcoupling. Requirescontrolling slope freedom. Existing
LRtrials remain unchanged; user says higherLRsounds good fornow, not cancellation.

Further width-decoupling discussion only: topology in localturncoordinates,
unwrapped measuredangle asinitiallocalphase withsmoothcorrections; widthfree
sign/distance/q-Vcontinuation. Separate metricfit of widths usingUarc-length
measurements, robustlinearleast squares forfixedturnpositions, optionally
stopgradient frommetricbacktotopology. Widths neededforphysicalU,notinteger
turntransitions. Notimplemented; proposedstrictdecoupling explicitly notclaimed
asfullyjointbackprop. Angle unreliableinfolds,so notahardphaseconstraint.

Allfour LRtrials finished10k, suiteexit0,finitefinalarrays,chargedcompiledsixrank
andcoordinatepreflightspassed. Managerleasefinished/released. HighfreshL1=1.2058,
retained65.5079%,widthfactor.851445; highrefL1=.633062,retained78.2474%,width
.908215. LowfreshL1=1.91641,retained64.9559%,width1.361157;lowrefL1=.499274,
retained93.0114%,width.917636. Solverwall185.607/185.917/182.172/184.589s;
peak1.573GiB/GPU. Full1ktables,commands,limits inuv_class_balance_lr_results.md.
