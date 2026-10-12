# Enabled GT constraint L1 tuning

Previous corrected solves, scorer and detailed results committed as 016cdf185.
Plan gt_l1_search_plan.md independently reviewed. New continuous enabled-event L1
uses an L1-optimal global integer offset (full median interval, closest-to-zero
tie rule). Historical modal vote score unchanged. Shared event_residuals helper.
Frozen 6735 events reused; input/code hashes checked. Empty cohorts cannot win.
No GT events are used inside training; this is hyperparameter tuning.

Fresh seed0,10080updates/globalbatch24576, six exclusive managed GPUs.
Existing coordinate search now supports explicit gt_enabled_l1, preserving spiral
ref_l1 as a diagnostic. Factors .5/1/2 then .25/.5/1/2/4 around initial values;
revisit coordinates on every acceptance. Report coverage, retained length,
per-type L1, loss and width factor. Enabled-only selection can favor rejection;
no unrequested coverage penalty. GT is tuning data, not holdout.

39 CPU tests passed (masking, gauge ties/invariance, empty cohort, objective
selection, historical resume and solver math). Independent code review approved.
No geometry extraction required. No numeric training changes.

Approved reusable command:
```
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_gt_l1_search.json
```
Output data/uv_gt_l1_search_v1: results.json, best.json, acceptances.json,
active_trial.json, per-trial config/summary/solution. Root summary.json only
exists once finite-grid refinement completes; no global-optimum claim.
