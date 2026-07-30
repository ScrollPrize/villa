# Native Fiber Trace VC3D Corner-Batch Sampling Status

- [x] Read repository and fiber-trace workflow instructions
- [x] Inspect current sampler/cache and VC3D blocking coordinate APIs
- [x] Write task and implementation plan
- [x] Update sampling and float-math specifications
- [x] Export shared one-level VC3D Zarr cache construction
- [x] Implement ordered batched eight-corner channel sampling
- [x] Port fiber prediction and Lasagna normal batches
- [x] Convert fiber tracer internal math to float
- [x] Add focused regression tests
- [x] Build and run focused native tests
- [x] Run approved representative benchmark and compare trace quality
- [x] Fuse persisted corner decoding with candidate scoring
- [x] Add direct-corner scoring coverage and rerun focused native tests
- [ ] Measure fused decode/scoring on the approved representative workload
- [ ] Update task log, changelog, and final consistency review

The combined prediction-plus-normal sampler supports the workload's mixed
chunk grids (64-cubed prediction and 32-cubed normals) while sharing voxel
coordinate/fraction construction. Its first contended benchmark measured
76.724s and 8 restarts. Worker utilization reduced that to 70.867s, and option
storage retention plus compact-normal lookup reduced it to 66.384s, both with
8 restarts. Compact dependency metadata, tensor lookup, and direct raw-corner
consumption reduced it to 57.789s with 8 restarts. Joint decode, retained raw
buffers, and parallel final-frontier materialization reduced it to 44.070s
with 8 restarts. Retained/parallel candidate construction, static scoring, and
heap-based exact pruning regressed to 74.329s under the competing workload.
The two regressing scheduling changes are removed; reusable task storage plus
the successful heap pruning reduce runtime to 37.743s with 8 restarts. The
within-chunk base-offset corner specialization is built and tested, pending
explicit approval before another representative run. Its first run reduced
corner batching to 8.721s but total runtime was 38.117s due to slower contended
OpenMP stages. A moderate `dynamic,64` scheduling control reduced the same
contended workload to 37.277s wall / 1049.197s CPU. Trace quality remains at 8
restarts, versus the 5-restart pre-float baseline.

The next revision removes the full decoded prediction/normal sample arrays from
the persisted fast path and scores candidates directly from the retained corner
batch. All 26 fiber-trace, 15 strict corner-sampler, and 11 Lasagna normal tests
pass. Its representative performance and restart result have not yet been
measured.
