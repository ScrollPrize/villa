# Task: stage-parallel fiberlet extraction

Remove duplicated fiberlet preparation and fix preprocessing parallelization.

- Execute extraction as explicit global stages over the complete candidate set.
- Prepare each candidate's curved domain, mapped local nodes, and native
  interpolation corners exactly once, in parallel.
- Deterministically merge candidate corner sets into one global unique sample
  request set before any volume sampling.
- Batch only that unique coordinate set through the existing prediction and
  normal volume readers. Changing batch size may change sampler call count and
  temporary memory, but must never change total sample requests.
- Retain the prepared candidate geometry and reuse it directly during parallel
  DP instead of rebuilding domains/nodes.
- Preserve candidate ordering, admissibility, objective, paths, graph, and
  serialized artifact bytes across worker and batch counts.
- Benchmark wall time, process CPU use/effective cores, sample counts, and peak
  memory for every stage on the fixed Paris4 reference interval.
