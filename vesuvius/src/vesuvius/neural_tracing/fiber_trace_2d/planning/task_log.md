# Task log

## 2026-09-22

- Diagnosed the pessimistic ETA: processed-tile wall time was multiplied by all
  remaining candidates, even though most candidates in sparse volumes are
  empty and complete quickly.
- Planned separate empty-input and resume-complete skip counters. The estimator
  will learn sparsity only from processed plus empty tiles, and will not add a
  pre-scan or source read to the resume path.
- Implemented skip classification using the output/source checks already made
  by tile work discovery. The existing early resume rejection remains intact;
  classification adds only counter bookkeeping.
- Changed tile ETA to `observed wall / (processed + empty) * remaining`.
  Resume-complete candidates are excluded from the denominator, so a completed
  prefix cannot make the unfinished suffix appear free.
- Added clean-sparse, resume-prefix and resume-only ETA regression cases.
- Validation:
  - focused ETA tests: 4 passed;
  - `py_compile` passed for the shared runner, compatibility wrapper and tests;
  - `git diff --check` passed.
- Broader end-to-end tests were started but did not reach their first result in
  the available interval on the busy host and were stopped; no GPU workload was
  launched by this change.
