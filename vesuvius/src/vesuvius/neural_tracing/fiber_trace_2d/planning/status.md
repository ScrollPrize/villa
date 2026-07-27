# Correct Fiber 3D Tiled Inference Output Status

- [x] Re-read current task/spec/planning context.
- [x] Check current Lasagna `predict3d` normal encoding and pyramid behavior.
- [x] Check current fiber 3D inference adapter output schema.
- [x] Replace active `task.md` with the corrective fiber inference task.
- [x] Replace active `task_plan.md` with the implementation plan.
- [x] Reset `task_log.md` for the current task.
- [ ] Review plan against specs before implementation.
- [ ] Implement Lasagna-style fiber persisted output schema.
- [ ] Implement fiber direction conversion to compact `nx/ny`.
- [ ] Add `.lasagna.json` manifest writing for fiber inference.
- [ ] Build presence and `nx/ny` pyramids.
- [ ] Update fiber inference tests.
- [ ] Update specs/docs/changelog.
- [ ] Run focused validation commands.

## Current Plan Items

- [ ] Persist each fiber option as `presence`, `nx`, and `ny`.
- [ ] Keep seven-channel model outputs internal only.
- [ ] Encode direction with Lasagna's hemisphere `nx/ny` formula.
- [ ] Encode presence as uint8 fixed point.
- [ ] Use OME-Zarr data level at configured inference resolution.
- [ ] Build coarser OME-Zarr pyramids above that data level.
- [ ] Make `.lasagna.json` the authoritative output manifest.
- [ ] Preserve crop-composable chunk-existence resume behavior.
