# Predicted normal tracing comparison

Already on fiber-exp5 at915e31e0f, latest evaluation branch. Prior neural implementation preserved uncommitted. Selected EMA999u12100 and best-loss EMA99u19100 (same3errors as raw best, fewerpieces/morecoverage). Native replay shares normal sampler with GT acceptance: separate tracing normals to preserve metric. Auxiliary normal head not exported yet; explicit new product needed. Existing queues/resources unchanged.

Independent plan and implementation reviews approved. Added opt-in six-sigmoid normal output after width, accumulated using existing products then compact nx/ny encoding; sheet_grad_mag is255 validity (not a measured winding density). Native sampler aliases these channels on the same lattice; readiness includes all three channels before grouped or ordinary sampling. Forward and reverse use predicted normals; GT acceptance and restart seed search use original Lasagna. No silent Lasagna fallback. Explicit cache identity, missing-head rejection and no auxiliary TTA.

Native RelWithDebInfo build succeeded;9Python tests and99native cases passed. Full first launch failed because the new driver did not propagate single-thread BLAS environment to evaluate subprocesses; huge automatic OpenBLAS pools exhausted thread capacity. Set env before imports/children and resumed with original192CPU/6GPU,128input/accumulation/publication workers andbatch4 unchanged. No completed scientific result from failed launch. Existing cache reused where complete. Logs at scripts/reviewed_predicted_normals.run.log. Both full runs queued via approved command.

Resume guard required archiving the failed full directory; reused existing archive_incomplete_output helper, no deletion of caches or artifacts. Added independent known-axis tests: at nz=0 both antipodal signs are valid, corrected test to check axis not arbitrary sign. Final12Python/99native passed. No numerical code change from that test adjustment. EMA999 run now processing; hashes/source bindings and native effective policy are audited against historical control.

EMA999u12100 full completed182/182: min512 3errors/232pieces/81.070% retained vs historical0/233/82.086%. Total successful-attempt wall439.44s includes startup/inference/shutdown; initial failed attempt excluded. Additional matched-cache Lasagna controls queued to isolate normal-source effects from fresh accumulation roundoff. These preserve exact emitted primary/width bytes and all trace settings. Report generator wording corrected to avoid claiming enabled FW checks for these disabled-FW policies.

All four full182-case runs completed. Min512:

| Model | Normal source | Errors | Presence stops | Pieces | Retained% | Mean mm | Median mm |
|---|---|---:|---:|---:|---:|---:|---:|
| EMA999u12100 | historical Lasagna |0|86|233|82.086|13.079|3.814|
| EMA999u12100 | matched-cache Lasagna |1|87|234|82.031|12.776|3.830|
| EMA999u12100 | predicted |3|86|232|81.070|12.486|3.706|
| EMA99u19100 | historical Lasagna |3|111|258|80.681|9.725|3.376|
| EMA99u19100 | matched-cache Lasagna |3|110|257|80.773|9.822|3.363|
| EMA99u19100 | predicted |4|109|258|80.821|9.827|3.351|

Unfiltered error counts identical to min512 for these four new runs. Additional
control inference was needed for divergent routes; previously published shared
chunks remain unchanged. Predicted full-run walls439.44s/371.03s; matched controls
193.96s/166.81s with mostly warm caches and small additional fills. Not an isolated
speed comparison; no speedup established. No clear quality improvement, no default
change. Fresh-cache historical variation visible in EMA9990to1error. Both policy
and cohort/normal-source-content bindings audited. Complete per-scroll/min0/min512,
P05/P95 and provenance stored in docs/benchmarks/reviewed_predicted_normals_20261007_v1.md
and JSON. Approved command is bash /home/hendrik/villa3/scripts/run_training_failure_mining;
control action reviewed_predicted_normals, optional paired_controls:true. Build/tests
use operation:build/tests. Completed runs are analyzed, not rerun.
