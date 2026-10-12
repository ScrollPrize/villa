# Adaptive loss-weight tuning

Prior state already committed at102d25a55; unrelated .pytest_tmp_private_hot/
preserved. Plan uv_loss_search_plan.md independently reviewed. Added controller
using existing solve/manager path; no objective changes. Includes all15 exposed
loss weights (including width regularizers), fixes LR and seed. Coarse factors
.25/1/4 then fine .25/.5/1/2/4 relative to baseline, finalL1 improvement >1e-4.
Immediate acceptance restarts coordinates; final complete no-improvement pass
certifies only finite-grid coordinate optimum. Reference is now tuning data.
Retention/matched counts reported; no retention gate added to user criterion.
Resume manifest hashes inputs and all fiber_uv code, full base config/provenance.
Incomplete/nonfinite/no-matched-fiber outputs fail rather than become candidates.
No 1k diagnostics; start/end only; loss now whole-run online mean, unlike earlier
last1k mean. Baseline rerun ensures same reporting schedule for comparisons.
Validation: 13 tests passed2.91s (weight_search + bucket_split), including disk
resume, incomplete-output rejection and changed-manifest checks. Independent
review approved; ranking code hash and solution identity checks included.
Implementation/plan committed dba4ffa6d. Approved runner started session48387,
manager98ec0fe8-2d8a-4627-bc12-73a62350072a acquired six exclusive GPUs.
First six-rank compiled gradient and coordinate validation checks passed.

User stopped search. Sent Ctrl-C to owned runner session48387; exit130.
Monitor cell1192 finished. Manager98ec0fe8-2d8a-4627-bc12-73a62350072a terminal,
held CPU0/GPUs{} verified. Completed96 trials,16 acceptances; unfinished key
8ed0c7e316d39355c587 excluded. No local optimum or fine-grid completion claimed.
Selected eb142efc153b00cd9e83: L1=1.0382352941, within2=88.4117647%,
retained=65.7748537%, widthfactor=.97857282, signs0%, matched1700fibers.
Canonical future benchmark config run_uv_current_baseline.json, provenance in
uv_current_baseline_result.json; all compact trials/acceptances committed in
uv_loss_search_trials.json and readable accepted chain uv_loss_search_results.md.
No new solve during stop/commit. Historical configs and solver fallback defaults
unchanged; README explicitly marks prior recommendations historical.
Final validation:13 tests passed2.73s; independent baseline review approved.
Verified selectedminimumL1among96, exactweights/settingsmatch, finite saved state.
Reported signs are bucket-sign violations, not physical-position violations.
