# Prefetch Download Priority By Earliest Sample Index

Change `fiber_trace_2d` training prefetch so missing chunk downloads are ordered
as much as practical by the earliest raw deterministic sample index that needs
the chunk.

- Keep dependency producers parallel.
- Keep global chunk deduplication.
- Do not flood the download executor with later-sample chunks before earlier
  pending chunks can be prioritized.
- `idx` should advance sooner when an earlier sample's chunks are known and
  waiting behind later-sample downloads.
- Active transfers already in progress do not need to be cancelled or
  reprioritized.
- `prefetch_sampler_workers` should be the practical CPU-side generation cap:
  prefetch source/dependency generation must not fan each producer out through
  PyTorch's full global CPU thread pool.
