# Dynamic batching status

- [x] Task and plan recorded; previous results archived.
- [x] Independent plan review.
- [x] Shared dynamic worker and live configuration implemented.
- [x] Correctness/ownership/profiling tests and implementation review (97 focused +6 shared-worker passed).
- [x] Exclusive fresh-inference pilot: caps1/2/4/8 completed;16OOM cleanly aborted.
- [x] Pilot results, memory limits, docs and logs recorded (not task completion).
- [x] Full50-test-fiber/100-direction matched heap batch1/cap8 and old-scan cap8 comparisons completed, fresh inference in every valid row.
- [x] Legacy CLI fixtures repaired;64 preprocess tests/6subtests passed. Shared batching/demand/evaluation/profile tests102passed. Final-flush display regression fixed and independently reviewed.
- [x] Admission bottleneck measured and replaced with exact-order heaps. Full cap8 wall684.879→516.699s; matched heap batch1 wall860.380s. Standard classic256³/cap8 config prepared/validated in fresh namespace; genericdefault1 retained.
- [x] Full-set MD/JSON, limitations and final independent evidence/config/code review recorded.106targeted+64preprocess tests/6subtests passed. All successful benchmark manager tasks completed and released resources. No automatic commit or unrelated edits reverted.
