# Initial AFV reconnaissance — 2026-10-08

- Preserved old tracked/untracked experiments in named stash; fetched origin/main
  and switched to fresh `global-fiber-sgd` at e0bbb8b40.
- Followed existing planning convention in this new standalone workspace rather
  than replacing main's unrelated fiber_trace_2d active task.
- Independent plan review approved scope and provenance/read-only safeguards.
- Downloaded 412,086,272-byte AFV into ignored data directory. Initial download
  went directly to final name; it completed successfully and was validated before
  use. Documented partial-name download procedure for future use.
- Read existing AFV specification and C++ reader; no duplicate reader or solver
  implemented. Inspected SQLite using mode=ro.
- Verified native 2.4um April-2026 Paris4 scan; AFV traces originate from level-4
  prediction (38.4um). Remote CT shape and prediction scale agree.
- Full finite-coordinate/block-count/segment-continuity validation passed for
  all 512,698 fibers and 5,265,142 points. SQLite quick_check and foreign-key
  checks passed. Byte count and local SHA256 recorded in README.
- Scope complete: dataset/format/provenance investigation only. PyTorch SGD
  formulation and implementation are future work, not claimed complete.
