# Plan: continuous deterministic crop tracing and zero-copy graph access

## Contract

- Preserve anchor ordering, candidate geometry, coverage suppression, accepted
  line order, termination reasons, and numerical objectives exactly.
- Parallel completion order must not affect the committed result.
- Keep memory bounded when an early slow candidate delays the commit frontier.
- Cache-backed views pin their payload or derived buffer for the view lifetime;
  search states retain IDs and values, not cache leases.
- Missing/corrupt sparse chunk behavior remains unchanged.

## Implementation

1. Add move-friendly directional borrowed-view types for outgoing arcs, route
   points, segment lengths, and segment costs. Each view has indexed
   forward/reverse access and at most one opaque shared lease. Preserve the
   existing owned query methods as compatibility materializers. An immutable
   outgoing item carries a stable edge handle plus public ID/arc data, avoiding
   another ID lookup; persistent search state still retains only public IDs.
2. Implement immutable views over stable storage without leases. Replace the
   immutable graph's pointer-heavy maps with sorted contiguous anchor, edge,
   and transition arrays plus flat outgoing adjacency. Resolve public storage
   IDs at the API boundary and use stable indices internally.
3. Make cache-backed queries use the same view boundary with one aggregate
   owned result per query. Multi-chunk filtered adjacency and compact route
   reconstruction each produce one derived buffer; the view owns that buffer,
   remains valid across cache eviction, and does not pin unrelated chunks.
   Never reference-count individual elements or retain owners in search state.
4. Update crop lookahead and committed tracing to consume directional views.
   Compute crop-exit fraction without constructing a clipped point vector;
   materialize clipped geometry only for the selected committed fiberlet.
5. Replace fixed `workerCount` batches with a continuous coordinator/worker
   scheduler. Assign dense monotonically increasing tickets in strongest-first
   seed scan order, skipping only anchors already inactive when claimed.
   Maintain at most `workerCount + max(1, workerCount / 8)`
   submitted-but-uncommitted tickets. This measured headroom avoids a batch
   barrier without allowing a slow frontier candidate to generate a complete
   second batch of usually invalidated work.
   Workers compute outside locks and publish completions; the coordinator
   drains every consecutive ready ticket, invokes progress callbacks without a
   scheduler mutex held, and refills the window. Claimed seeds invalidated by
   an earlier commit remain tickets and are discarded only at ordered commit.
6. Stop claiming when attempt/fiber limits are reached, join outstanding work,
   and ignore results or exceptions strictly beyond that serial stop point.
   A failure at the ordered frontier propagates after all earlier tickets have
   committed, matching one-thread semantics. Retain timing counters with
   documented continuous-scheduler semantics.

## Tests

- Verify immutable forward/reverse views and owned compatibility queries return
  identical IDs, points, lengths, and costs.
- Verify cache-backed views retain valid data for their complete lifetime.
- Destroy the source buffers after constructing an owned view and verify its
  forward/reverse route/profile data remains valid. Cover immutable
  forward/reverse adjacency, route, and cost-profile parity against the owned
  compatibility API.
- Compare serial, ordinary parallel, and deliberately skewed parallel crop
  results exactly, including accepted lines/order, coverage, limits, and
  termination metadata. Allow only speculative computation counters to differ.
- Verify attempt/fiber limits ignore later speculative failures while an error
  at the ordered frontier propagates after the same committed prefix as serial.
- Build and run focused GCC Release and Clang test targets.
- Benchmark the same Paris4 1024-base-voxel crop with 500 attempts before and
  after; report wall/CPU times and exact result equality.
- Run `git diff --check`.

## Spec update

Specify ordered continuous candidate finalization, bounded speculation, stable
directional graph views, and lease lifetime rules. State that these are exact
execution/data-layout changes, not numerical changes.

## Documentation updates

Document crop scheduler determinism, speculative discard behavior, timing
counters, immutable views, and cache pinning lifetime.

## Changelog

Add a crop-tracing performance entry after implementation and validation.
