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

Started approved runner, PID654446, manager task
0aa106fa-2af6-4c58-bddb-0b922e262d7d, GPUs0–5 exclusively allocated.
Baseline finished: GT enabled L1 .5777874823241785, offset -2,
5419/6735 events (80.4603%), parallel L1 .5743005468 (4844), perpendicular
L1 .6071626398 (575). Retained length79.3598%, spiralL1 .946903,
widthfactor .900953, mean online loss12.386964, solve184.87s.
Matches original saved-event analysis and previous solve exactly (floating
summation tolerance for L1). First perturbation distance_weight .25->.125
running; distributed gradient/coordinate validations passed. Search ongoing,
not converged. Implemented search/scorer commit f1bcf1e2f.

User requested continued iteration. Controller never stopped (PID654446).
First accepted distance_weight .25->.125: L1 .4981992495, coverage82.9844%,
retained80.4114%, loss12.175396,widthfactor .979491. Distance .5 and sign20/80
did not beat incumbent. Continuing automated revisit loop, not a new solve or
duplicate controller. No convergence claimed.
