# Status: mixed-integer crop-fiber labeling

- [x] Capture the requested task in `task.md`.
- [x] Inspect constraint semantics and installed HiGHS integration.
- [x] Write implementation, testing, spec, docs, and changelog plan.
- [x] Complete independent plan review and incorporate findings.
- [x] Reject measured winding distances at or above `1.5`.
- [x] Implement and integrate the HiGHS MILP.
- [x] Write five label OBJ outputs and CLI reports.
- [x] Add focused regression tests.
- [x] Update specs, docs, changelog, status, and task log.
- [x] Build and test GCC.
- [ ] Build and test Clang (not run before the requested wrap-up).
- [ ] Complete the representative default-radius Release solve (stopped after
  183.66 seconds; dense MILP scalability remains unresolved).
- [x] Measure the untightened LP on centered 256 and 1024 crops.
- [x] Update task and task plan for stable XOR variables and triangle cuts.
- [x] Complete independent review of the LP-tightening plan.
- [x] Implement stable gated differences, component gauges, and triangle cuts.
- [x] Add focused triangle/gauge regression tests.
- [x] Update specs, documentation, changelog, status, and task log.
- [x] Build and run focused GCC tests.
- [x] Benchmark tightened LP on centered 256/384 and attempt 512/1024 crops.
- [x] Add and document thresholded five-layer LP OBJ visualization.
- [x] Add explicit relaxation-only HiGHS parallel and solver controls.
- [x] Benchmark parallel automatic LP selection on the centered 384 crop.
- [x] Attempt parallel HiPO on the centered 384 crop and record the linked
  build's missing-backend failure.
