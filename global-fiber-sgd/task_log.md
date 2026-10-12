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
Validation: 12 tests passed2.65s (weight_search + bucket_split).
