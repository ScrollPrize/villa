# Integration log

- Source branch is fiber-training-retrain, commit 32d92bb0a. Merge conflicts affect task documents and train.py progress/checkpoint paths. Retain shared checkpoint helpers and cooperative test/handoff handling; add sample-progress fields at their shared call sites.
- LR 0.01 was the recorded short-run choice before global normalization and embedding supervision, not a proven embedding optimum. No GPU runs planned.
- Independent review approved the plan and caught two integration edges: RAM handoff must not reapply a disk-resume sample override; the benchmark metric parser must accept embedding fields. Both corrected.
- CPU embedding/sample-progress/accumulation tests: 43 passed, 12 GPU tests skipped. The legacy direction-only test fixture explicitly filled every new embedding field with a differentiable zero; corrected it to leave unused embedding fields at their defaults.
- Local config parses; train/test globs resolve 8 NML files and 1 JSON. All three local normals arrays have shape (4737,2044,2044), base shape (75784,32693,32693). No data download or GPU training performed; real embedding GPU memory/performance remains unmeasured.
- Cooperative test expectation changed from update steps [1,2,3] to sample steps [5,10,15]; added assertions for sample metadata across cohort handoff.
- Final cooperative suite: 6 passed, 1 GPU-dependent skip. Total targeted validation: 49 passed, 13 skipped; git diff --check clean.
