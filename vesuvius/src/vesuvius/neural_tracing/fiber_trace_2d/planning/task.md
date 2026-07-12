# Top-View Loader Performance Parity

Top-view training became much slower after adding the additional slice. Fix the
top-view data path so it follows the same infrastructure and accelerations as
the side-view path.

- Reuse the same source-cache mechanism for top-view CP-local source geometry,
  without invalidating existing side-view cache entries.
- Repeat the side-view vectorization and batching pattern for top-view samples:
  batched augmentation-map construction, batched line/control-point lookup,
  batched coordinate augmentation, and grouped `sample_coord_batch` calls.
- The only intended difference from the side-view path is the source coordinate
  generation function: top view uses the top-strip grid builder, side view uses
  the side-strip grid builder.
- Keep deterministic sample order and current top-view labels/loss semantics.
- Make profiling attribute top-view work to the normal aggregate stages so
  top-view load time is visible instead of hidden in `outside`.
