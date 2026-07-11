# Load-Only Parallelism Diagnostics Log

Current task: add real process-level CPU/wall timing to load-only profile output
so the loader's summed worker timings can be compared against actual process CPU
usage.

Implementation notes:

- Added per-batch `time.process_time()` deltas around the benchmark batch body.
- Added `cpu` and `ctf` profile row columns:
  - `cpu`: process CPU milliseconds per patch for the whole process.
  - `ctf`: process CPU time divided by batch wall time.
- Kept existing `work` and `tf` columns unchanged:
  - `work`: summed per-candidate loader worker elapsed time.
  - `tf`: `work / load_batch wall`.

Validation:

- Compile check:
  `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py`
  passed.
- Focused loader tests:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: `106 passed in 4.97s`.
- Load-only profile benchmark:
  `PYTHONPATH=/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/volume-cartographer/build/python-bindings/python:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3 python -m vesuvius.neural_tracing.fiber_trace_2d.train /home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --benchmark --load-only --profile`
  completed with `batches=100`, `patches=12800`, `elapsed_ms=86678.4`,
  `patches_per_second=147.67`.
- Profile summary from that run:
  - `process_cpu=25.848 ms/patch`
  - `loader_wall=6.760 ms/patch`
  - `loader_worker=202.474 ms/patch`
  - `loader_thread_factor=29.951`
  - `process_cpu_factor=3.817`

Finding:

- The existing `tf` column was only a synthetic summed-worker-timer ratio. It
  can report about 30x while the actual process consumes about 3.8 CPU cores on
  this workload. The new `ctf` / `process_cpu_factor` columns are the numbers
  to compare with system CPU monitors.
