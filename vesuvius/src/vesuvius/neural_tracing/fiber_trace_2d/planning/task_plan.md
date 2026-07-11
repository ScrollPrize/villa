# Training Throughput Parallelization Plan

## Implementation

- Re-establish a current benchmark/profile baseline using the approved command.
- Inspect and simplify the nested pipeline/loader threading model so CP sample
  loading can use available CPU workers without contention between outer batch
  workers and inner sample workers.
- Test variants by rewriting `/tmp/fiber_trace_p_d2_w1.json` only:
  - current defaults;
  - high loader worker count with shallow training pipeline;
  - deeper training pipeline only if it improves wall throughput;
  - moving CUDA prep concurrency from one stream/thread to multiple streams only
    if measured beneficial.
- Keep any variant only if it improves measured patches/s on the same command.
- Add profiling counters where needed to distinguish wall time, worker time,
  wait time, prep time, and train time.

## Spec Update

- Document the measured training pipeline/loader defaults and the benchmark
  method used to select them.
- Document that deterministic sample ordering must not depend on worker count.

## Docs Updates

- Update `planning/local_development.md` only if the benchmark workflow changes.
- Update `docs/code_structure.md` if config defaults or pipeline behavior change.
- Keep `planning/task_log.md` limited to this throughput task.

## Testing

- Run focused loader tests:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
- Run compile check for touched Python files.
- Run the approved benchmark command after each meaningful variant:
  `PYTHONPATH=/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/volume-cartographer/build/python-bindings/python:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3 python -m vesuvius.neural_tracing.fiber_trace_2d.train /tmp/fiber_trace_p_d2_w1.json --benchmark --benchmark-batches 30`

## Changelog

- Add a concise entry if a retained change improves throughput or changes the
  training pipeline defaults.
