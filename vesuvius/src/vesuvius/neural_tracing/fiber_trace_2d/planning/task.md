# Task: process-parallel rolling-mmap flush

Replace the shared 3D inference runner's Python background flush thread. The
threaded implementation regressed a representative eight-GPU crop from 178.8 s
to 305.4 s and can contend with the coordinator through the GIL and
process-global native-library state.

Implement a persistent spawn-process flush pool shared by Lasagna and Fiber:

- workers reopen the existing rolling accumulator mmap files and receive only
  immutable chunk descriptors, never chunk arrays;
- distinct finalized output chunks are normalized, finalized, compressed, and
  written in parallel processes with independent GILs;
- retain the single enlarged mmap ring, one frozen flush batch, completion-
  gated reuse/progress, bounded one-chunk-per-worker RAM, canonical numerical
  semantics, sparse/resume behavior, and atomic output writes;
- expose a bounded worker count and an explicit synchronous baseline mode;
- remove Python threading from the flush path and preserve portable spawn
  behavior on Ubuntu/macOS and amd64/arm64.
