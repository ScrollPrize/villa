# Local evaluator build

From this checkout, run the same commands each time:

```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_replay_tests
```

Build artifacts live in `volume-cartographer/build/fiber-evaluation`, including
`bin/vc_fiber_trace_metric`, `bin/vc_cache_root`, and `bin/test_fiber_replay`.
The no-argument build runner configures an incremental RelWithDebInfo build,
with eight build jobs and no native CPU tuning. No environment installation is
performed. Historical evaluation configurations retain their original paths;
new thickness40/grace160 configurations use this workspace build.

This is a machine-local Linux/GCC13 runner, not portable dependency setup.
It reuses the already installed libbacktrace and existing delta3d source under
`/home/hendrik/villa/volume-cartographer/build/dev-quickbuild-gcc/_deps` read-only.
`libbacktrace.pc` supplies missing discovery metadata for that installation.
Missing dependencies fail preflight rather than requesting installation.
