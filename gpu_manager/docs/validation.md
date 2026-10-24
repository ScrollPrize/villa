# Validation scope

## Standalone regression suite

From the repository root, with the package's test dependencies available:

```bash
python -m pytest -q gpu_manager/tests
```

The suite uses fake inventories and temporary private Unix sockets/journals.
Process tests create and clean up their own CPU-only children. No CUDA context,
production manager restart or GPU reservation is needed. Coverage includes
protocol/schema validation, priority/FIFO admission, exclusivity, pause/resize,
manual splits, process identity/reuse, crash recovery, ownership barriers,
driver-query failures, automatic startup/shutdown and command/cohort cleanup.
Socket/process permissions are required; a restricted sandbox may prevent these
tests. The optional `GPU_MANAGER_SCHEMA_PYTHON` selects another interpreter with
`jsonschema` installed; it is not needed in a complete test environment.

The extracted suite passed **157 tests in 50.90 seconds** on Linux/Python 3.14.
Linux Python 3.11/3.14 CI and the CPU-Torch integration job passed after the
fixes below (run 39480512827). Only Linux is supported. macOS CI exposed an
orphaned-loader cleanup check that still observed a live process; macOS was
removed from the supported matrix rather than weakening that safety assertion.

Initial CI exposed an undeclared optional Torch test dependency and a macOS
page-rounded shared-memory mapping bug. Payload writes now address only their
serialized length; a padded-mapping handoff/resume regression exercises this on
Linux too. Torch cases explicitly skip without their dependency and run in a
dedicated CPU-Torch CI job. Failure-injection tests require the intended failure
message, so missing imports cannot masquerade as successful cleanup coverage.

## Historical real-data evidence

This manager was developed while running fiber training and tracing on real
scroll volumes. A 2026-09-10 reference test used S1A NML annotations, 128-voxel
patches, effective batch 128 and two H100 GPUs. It exercised GPU-free pause,
partial-VRAM allocations and two-to-one GPU resizing.

Same-allocation six-update restarts preserved model/optimizer/test results
byte-for-byte for FP32 and BF16. Fixed versus dynamically repartitioned runs
were not bitwise equivalent: final six-update losses were 2.242254823 versus
2.242277861 (FP32), and 2.239006564 versus 2.239104435 (BF16). One 100-update
comparison measured 7.945 seconds/update with fixed managed allocation versus
8.010 without admission/RPC; this single comparison is not a speedup claim.
Repeated reconfiguration increased wall time to 11.030 seconds/update.

These are historical integration results, not a new GPU run of this extraction.
The training adapter and experiment artifacts are intentionally outside this
standalone package. They do not establish arbitrary-device calibration,
long-run convergence equivalence or hard GPU isolation. Cross-user credential
tests simulate identities rather than provision OS users.

The [service responsiveness report](responsive-service-validation.md) separately
records the real process-table bottleneck and CPU-only regression measurements.
