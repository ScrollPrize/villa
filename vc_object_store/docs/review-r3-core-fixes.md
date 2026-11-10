# R3 core fixes

Baseline `cf8406afc`; shared core C3-1–4 plus the catalog-transform atomicity sibling. Preserve schema 21 and history-off identity/routing. No copied referenced geometry, proxy baselines or extra persistent metadata.

- [x] C3-1 publish first inherited deletions in history and feed.
- [x] C3-2 validate keep-recorded without creating unchanged overrides.
- [x] C3-3 use selected-geometry lengths consistently across queries and regions.
- [x] C3-4 operation-wide shared savepoints for surface creation, umbilicus replacement and catalog transforms.
- [x] Permanent regressions and focused native validation.

Implementation and focused validation complete after dependency restoration by the main agent. Integrated suite validation remains owned by the main agent.

First inherited deletion publishes the actual deletion as a root tombstone when there is no local predecessor; the source version remains in its immutable owner. No fabricated pre-edit local history or geometry payload copy is created. Keep-recorded still validates endpoint/dependency semantics, without turning an unchanged inherited relationship into a local record.

Length selection now shares one measured path in native and registered/unregistered volume frames. Queries no longer use ingestion summaries to filter or order measured geometry, or to break cross-frame ties. Equal lengths sort by UUID everywhere; cursor measurement is returned in the existing nonpersistent `volumeLength`/`volume_length` field even for native queries. Persisted detail metadata is unchanged. Placeholder records lacking stored geometry remain in ordinary listings but are omitted when a measured length is required. Spatial and viewport filtering follow the same measurement. This explicitly fixes the prior inconsistent registration-dependent policy; it does not claim unchanged tie order or summary-only filtering.

Native composite helpers use the existing shared `Savepoint`, including its transaction-local state rollback, around the full operation. The regression catches each failure and deliberately commits the outer transaction to verify no partial object, retirement, registration or commit remains.

## Validation

Linux, existing RelWithDebInfo build, self-cleaning synthetic native fixtures (no full-volume or macOS claim):

```sh
cmake --build local/vcobj-pr-tests/build --parallel 4 --target test_coherent_store test_object_store_transforms test_object_store_volume_target test_object_store_geometry_region
ctest --test-dir local/vcobj-pr-tests/build --output-on-failure -R 'coherent|transforms|volume_target|geometry_region' --parallel 2
```

Final targeted run: **4/4 passed, 3.77 seconds**. The coherent test covers all four base/overlay history-policy combinations for first inherited deletion and unchanged keep-recorded/default edit-builder behavior, source version availability, deletion receipt/feed and relocated tombstone source paths. History-off correctly invalidates an old feed cursor instead of retaining a feed. Composite failure regressions deliberately catch exceptions and commit, for both history modes. Length tests use inconsistent ingestion summaries, native/volume/region queries, pagination, an unrelated registration, referenced geometry and a geometry-less placeholder. The existing multi-frame pagination test now asserts UUID tie order consistently. `git diff --check` passed.

Correctness tradeoff: first length-required queries may read more geometry than the former summary-only shortcut, using the existing snapshot measurement cache. No new persistent cache/schema/index was introduced and no performance claim is made. Ordinary unfiltered listings remain unaffected.
