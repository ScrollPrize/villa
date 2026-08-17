# Task: direct replay visualization manifests

Restore the original viewer workflow for fiber replay diagnostics. Each
failure-local visualization produced by `vc_fiberlets fiberlet-replay --vis`
must be directly openable with `view_fiber_presence --replay <manifest>`.
Remove the viewer's `--index` argument and retain the aggregate
`fiber_replay.json` only as a report/index. Preserve loading of existing strict
version-1 single-visualization replay artifacts.
