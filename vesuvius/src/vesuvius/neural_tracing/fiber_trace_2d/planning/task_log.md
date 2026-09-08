# Task log: overlap Fiber3D batch preparation

- Baselines: GPU replay 95.708 global samples/s; shared real pipeline trials
  84.028 and 83.595 samples/s (83.812 mean); standalone CPU pipeline 222.09
  production samples/s.
- Integrated profile attributes the remaining approximately 24ms/step gap to
  13.6ms synchronous H2D, 4.0ms target construction, 3.7ms mean ordered batch
  wait/assembly, and residual host/synchronization jitter.
- Planned implementation keeps deterministic sample production shared and adds
  a bounded asynchronous preparation stage rather than another loader.
- Independent review rejected explicitly reusable device slots without consumed
  events. The implementation will instead keep three allocator-owned live
  batches (current plus two future), use `record_stream`, report allocated and
  reserved memory, and quiesce before evaluation.
- Implemented a depth-two coordinator around the existing shared sample
  pipeline. It pins and releases ordered CPU batches off-thread, submits H2D and
  target construction on a dedicated CUDA stream, and exposes failures/events
  in order. The synchronous path remains the default.
- Two matched runs reached 91.792 and 91.646 global samples/s (91.719 mean), up
  9.4% from 83.812 and reaching 95.8% of the 95.708 replay ceiling. Exposed
  wait was below 0.7ms at p95. Peak allocated/reserved device memory was
  14.99/20.31GiB per rank; bounded pinned storage was 240.0MiB per rank.
- Steps 98--100 matched the synchronous run's losses and metrics exactly across
  many queue wraps. The remaining gap is GPU-side copy/target contention rather
  than input starvation.
- Validation: 290 focused Fiber3D, profiling, and ordered-pipeline tests passed;
  Python compilation and `git diff --check` also passed.
