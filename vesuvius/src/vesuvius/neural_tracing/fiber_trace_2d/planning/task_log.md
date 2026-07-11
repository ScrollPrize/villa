# Loader Serialization Bottleneck Task Log

## Baseline

Command:

```bash
PYTHONPATH=/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/volume-cartographer/build/python-bindings/python:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3 python -m vesuvius.neural_tracing.fiber_trace_2d.train /home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --benchmark --load-only --profile
```

Result before current changes:

- batches: 100
- patches: 12800
- elapsed_ms: 65382.5
- patches_per_second: 195.77
- process_cpu_factor: 3.219
- loader_thread_factor: 0.848
- ms/patch: coord_gen=9.033, coord_cache=5.980, line=2.405,
  loading=7.848

## Findings

- Python GIL is not the obvious bottleneck for VC3D sampling: the
  `vc.volume.Volume.sample_coords` binding releases the GIL.
- `load_batch` parallelizes CP sample preparation, then runs
  `_finish_prepared_batch_samples` as a serial tail in the batch future.
- That finish stage groups all prepared samples by sampler and calls one large
  `sample_coord_batch` per sampler. With one shared VC3D sampler this makes
  volume sampling one large C++ call per batch instead of work distributed
  through the CP worker pool.
