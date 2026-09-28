# Parallel post-trace extraction

- Checkpoint commit: 8ed2c7911.
- Baseline: native finished17:14 UTC (286503 evaluated,14631 pruned); serial auditing reached179920 at18:51 UTC, about31 audits/s. Subsequent union export also serial.
- Preserve completed native batch; no retracing or inference tuning.

## Correctness discovery and approved scope change

- Independent review found the global-trial descriptor used run-relative lo/hi while native sliced the full original annotation. Affected 1277/1343 candidates,294815/301134 submitted trials,283298/286503 evaluated trials. Example c5563a72aadc21b72f10: run start4800.0037; intended trial start8014.7549, actual3214.7512; XYZ error3821.157 base voxels.
- User explicitly approved discarding old reproduction outputs (not inference), correcting coordinates and rerunning. This supersedes the original preserve-trial requirement.
- Protocol v2 requires explicitly named original-fiber arc start/end, expected start/end XYZ. Expected XYZ comes independently from historical eligible-segment geometry. Native validates before tracing; Python validates saved endpoints. Tolerance0.25 base voxel. Legacy coordinate contracts fail closed.
- Real-data validation:1343 candidates,10016 endpoints,maximum error3.625691246197609e-10 base voxels.
- Removed old reproduction_search from active use and launched deletion PID203143. Only that tree is deleted; original evaluations, GT and every inference cache remain untouched. Cleanup rejects intermediate symlinks escaping the discarded tree.
- Corrected workflow restarted PID203564 with existing approved no-argument runner. Manager granted192 CPU cores and GPUs0–5. Input/accumulation/publication/batch settings unchanged.

## Parallel postprocessing

- Extracted exact audit math into CPU-only failure_trial_audit.py, shared by serial callers/process workers. Bounded16-path batches, compact IPC, coordinator duplicate/count validation, threaded result writes. Audits retained in RAM for selection rather than rereading hundreds of thousands of files.
- Completed-batch recovery accepts only compatible protocol-v2 geometry, avoids retracing completed results after interruption. Invalid v1 output is not recovered.
- Downstream reference JSON parsing uses processes; annotation export, chunk links, corridor computation and deduplicated chunk-existence checks use bounded threads. Corridor bounds vectorized and identical chunk boxes visited once. Selection and interval merge remain ordered and serial (small work).

## Measurements and verification

- Command: bash /home/hendrik/villa3/scripts/run_training_failure_mining (control actions benchmark_export,validate_trial_geometry,build,discard_invalid_reproduction,start). No new elevated entrypoint.
- Audit benchmark:512 saved trace JSONs,2421804430 bytes,3 repetitions each. Python3.14/NumPy,unchanged audit math. Serial mean14.3651s,median14.3758s,min14.2099s,max14.5096s.16 processes mean1.7188s,median1.1822s,min1.0985s,max2.8757s.32 processes mean1.2022s,median0.5979s,min0.5969s,max2.4119s. First parallel repetition includes startup. Outputs exactly equal. ~24x median improvement; not an end-to-end extraction speed claim.
- Initial benchmark invocation used grace80 instead of production160 and correctly failed native/audit agreement; fixed benchmark to read actual evaluation grace,then measured above. No production configuration changed.
- Corridor regression:50000-point seeded polyline,anisotropic geometry,3 repeats. Original mean0.129416s/median0.129320s/min0.129124s/max0.129804s; vectorized mean0.011093s/median0.011018s/min0.010928s/max0.011333s. Exact chunk-set equality (~11.7x).
- Native RelWithDebInfo workspace build via approved runner:23 replay,86 tracing,18 reference replay tests passed;4 added end-to-end trials passed (forward/reverse,correct coordinates/deliberately wrong coordinates).
- Python command: /home/hendrik/.venv_las/bin/python -m pytest -o pythonpath=/home/hendrik/villa3/vesuvius/src -q scripts/test_training_failure_mining.py vesuvius/tests/neural_tracing/test_failure_trial_audit.py vesuvius/tests/neural_tracing/test_short_failure_benchmark.py vesuvius/tests/neural_tracing/test_live_inference_cache_namespace.py —63 passed.
- Limitation: full corrected extraction is a background run, not yet completed. New scientific results/ETA cannot be inferred from the invalid previous run.
- Startup verified:18100/301134 trial positions handled,2729 pruned.128 new saved artifacts independently audited; all expected-XYZ and native/Python decisions agree (14 reproduced). GPUs0–5 observed68–100% utilization; GPU6/7 belong to other work and are not requested. Background deletion finished successfully; old invalid trial data is no longer recoverable except by rerunning, as requested.
