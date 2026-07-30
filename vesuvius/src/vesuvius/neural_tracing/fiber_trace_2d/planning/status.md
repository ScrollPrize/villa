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
- [ ] Run approved representative benchmark and compare trace quality
- [ ] Update task log, changelog, and final consistency review

The combined prediction-plus-normal sampler supports the workload's mixed
chunk grids (64-cubed prediction and 32-cubed normals) while sharing voxel
coordinate/fraction construction. Its first contended benchmark measured
76.724s and 8 restarts. Worker utilization reduced that to 70.867s, and option
storage retention plus compact-normal lookup reduced it to 66.384s, both with
8 restarts. Compact dependency metadata, tensor lookup, and direct raw-corner
consumption are built and tested but require explicit approval before another
representative run.
