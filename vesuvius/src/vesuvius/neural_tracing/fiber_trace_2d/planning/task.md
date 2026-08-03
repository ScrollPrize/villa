# Task: overlap rolling-accumulator flush with inference

The shared 3D tiled inference runner currently performs each finalized Z-band
flush synchronously. With multi-GPU inference this blocks ordered result commit,
result-slot recycling, and eventually all GPU workers.

Implement asynchronous flush overlap in the shared runner used by both Lasagna
and Fiber inference:

- retain one circular mmap accumulator rather than copying bands;
- enlarge its bounded physical Z capacity so one finalized/frozen band can be
  flushed while the next band continues accumulating in disjoint slots;
- allow at most one flush operation in flight;
- before beginning the next flush, wait for the previous flush, release its
  slots, and propagate any error;
- do not introduce a band-sized RAM snapshot or full-volume scratch;
- preserve canonical accumulation order, output bytes, sparse/resume behavior,
  progress semantics, cleanup, and the common Lasagna/Fiber implementation.
