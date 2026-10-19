# Consumer performance improvements

The object store now supplies bounded catalog/link queries and a store-backed
graph read view for vc3d and Spiral adapters. **Spiral does not need an export:**
`graph_read()` pins a store read transaction, builds native CSR topology arrays,
and reads projected metadata and geometry from that same snapshot. No files,
sidecars or second database are created. These changes prepare the storage API;
vc3d and spiral-fitting have not yet been rewired to use it.

See the [API and examples](../object_storage.md#consumer-queries-and-store-backed-graph-reads),
[original investigation](performance.md), and [follow-up raw results](performance/2026-10-19-after/).
The subsequent [Spiral input-loading work](performance-spiral.md) further improves
graph construction and adds packed geometry reads; the measurements below retain
the results of this first implementation pass.

## Changes

| Consumer access | Implementation |
| --- | --- |
| Crossing review and attachment lookup | `select_edges` applies kind, approval, tag and endpoint filters before hydration, supports field projection and UUID cursor pages. Endpoint-scoped queries probe endpoint indexes rather than scanning an edge kind. |
| Fiber/track catalog and slices | Typed family/minimum-length filters, descending length with stable UUID ties, and cursor pagination. Spatial region filters run before the limit. |
| Spiral iterative graph access | Read-only NumPy views of native adjacency, endpoint and edge-index arrays; retained UUIDs, object generations, geometry revisions and journal sequence; lazy metadata/geometry reads on the pinned transaction. Parallel links, self-links and isolated nodes retain their identities. |
| Point gathers | Batch ranges are scheduled by physical chunk, then scattered back to caller order. Each touched chunk is loaded/verified once per call, including overlapping/duplicate/unaligned ranges. Small explicit single reads use the existing verified cache. |
| Autosave and review changes | Identical JSON assignments and removal of absent keys preserve generation, approval and journal. `set_meta_many` provides one atomic durable transaction for metadata triples. |
| Component selectors | Keep membership as internal IDs, batch UUID conversion, and reuse bounded membership within a read transaction. |
| Scoped detector reruns | Limit existing-edge reads to candidate/stale-scope endpoints while preserving unscoped stale replacement and reviewed/manual links. |
| Existing metadata export API | Apply projections before hydration and batch endpoint revision bindings, retaining dependency-aware fallback when heads differ. This API optimization is independent of Spiral's direct graph reader. |

Schema 7 adds catalog and edge indexes. Writable opens migrate existing stores
transactionally; old read-only stores remain readable without the new indexes.
The migration does not rewrite geometry or record/journal history. Older binaries
cannot open schema-7 stores. Index construction adds work to the first writable
open of an existing database.

## Measurements

Same machine/build configuration and synthetic fixtures as the baseline, native
Python bindings, warm-up followed by three timed samples; values below are
medians in milliseconds. The raw reports include every sample. These are
component measurements, not end-to-end vc3d/Spiral fitting speedups or CI gates.

| Workload | Baseline | After |
| --- | ---: | ---: |
| 100 approved crossings out of 50k, projected SQL page replacing load-all/filter | 552.46 | 0.41 |
| 200 family/length catalog rows out of 50k, projected SQL page replacing load-all/filter/sort | 486.99 | 0.65 |
| 512 point reads cycling over nine chunks | 292.16 | 5.58 |
| 512 random point reads from a 24 MiB fiber | 188.76 | 15.10 |
| Spatial fiber block query plus small geometry read | 0.552 | 0.007 |
| Full 50k crossing records, legacy API | 501.37 | 491.84 |
| First record of a 50k connected component, limit 1 | 228.21 | 123.85 |
| Existing projected metadata export, 50k chain | 1412.91 | 842.55 |

The crossing page now uses UUID ordering, whereas the baseline workaround sliced
insertion-order results. Both return 100 approved links; each benchmark validates
its specified order. The catalog page verifies the same selected UUIDs and order.
Both new page measurements request the fields needed by the benchmark, rather
than all record metadata. Applications that request large `meta` payloads will
pay for those returned payloads.

Bounded query scaling:

| Dataset size | Approved crossing page (100) | Family/length page (200) | One track's two crossings |
| --- | ---: | ---: | ---: |
| 1,000 | 0.348 | 0.642 | 0.026 |
| 10,000 | 0.379 | 0.636 | 0.028 |
| 50,000 | 0.409 | 0.651 | 0.028 |

Building the 50k-node/50k-edge graph view takes **508 ms**. Afterwards, 1,000
neighbor slices take **0.24 ms**, with no SQL in those accesses. Construction is
reported separately and includes topology plus identity/revision bookkeeping;
it does not include loading application-specific crossing metadata or geometry.
The benchmark explicitly includes all approval states to retain the fixture's
ring topology. Python `graph_read()` defaults to approved edges.

## Costs and remaining limits

The 50k metadata fixture grows from **109.3 MB to 124.4 MB** (about 14%) with the
additional indexes. Two hundred genuine edge-metadata changes in one transaction
increase from **7.30 to 10.22 ms**; indexes and semantic no-op checks add write work.
Batching still avoids a durable commit per edit. Identical edge metadata writes
fall from **1.26 to 0.013 ms** and now produce zero journal entries, zero generation
increments and no approval reset. No durability setting was weakened.
Seeding 50k objects and approximately 100k edges took 4.70 seconds versus 4.09
seconds before (one untimed-setup observation per run, not a repeated median).

Small cache-resident point gathers remain approximately 0.04 ms; batching fixes
the cache-thrashing case, not the inherent cost of loading/verifying new chunks.
Full array reads, bulk track loading and unrelated geometry edits are not expected
to improve materially. Geometry checksum and post-write cache-invalidation tests
remain enabled.

A globally sorted `connected_to` page still needs the complete component before
applying its limit. The membership cache avoids repeating traversal inside one
read transaction; separate Python convenience calls use separate snapshots.
Graph construction still scales with selected nodes and edges. Scoped detector
reconciliation can still be large when the selected endpoints have high degree.

Graph views hold a reader-pool slot and WAL snapshot. Close the view after input
preparation; obtained topology arrays remain valid. Reopen explicitly to accept
changed inputs. The consumer adapter must map crossing metadata and point anchors
to Spiral's numerical arrays, convert coordinate conventions, and choose approval
policy. These adapters and an end-to-end fit benchmark remain the next migration
step. No application-specific interpretation is silently added to the store.

## Validation and reproduction

`ninja -C build vc_object_bindings test_object_store test_object_records test_object_links`
succeeded. All Python object-store tests passed (**167**); native store, record
and link suites passed (**5 + 13 + 11**). Coverage includes migration, cursor ties
and deleted cursor rows, atomic metadata rollback, unchanged approval/history,
projected self/parallel links, pinned graph metadata/geometry during writes,
array ownership after view close, thread confinement, geometry range ordering,
corruption detection, synchronization, and detector reconciliation.

The sync oracle failure recorded in the original investigation is resolved in
the test harness: copying a live native database now uses a separate-process
SQLite backup, avoiding interference between Python's and the native library's
SQLite file locks. Production sync semantics were not changed.

```sh
VC_BUILD_PYTHON="$PWD/build/python" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest python/tests/test_object*.py -q
build/bin/test_object_store
build/bin/test_object_records
build/bin/test_object_links
.venv/bin/python scripts/benchmark_object_store_metadata.py --output /tmp/vcobj-after-metadata.json
.venv/bin/python scripts/benchmark_object_store_workflows.py --output /tmp/vcobj-after-workflows.json
.venv/bin/python scripts/benchmark_object_store.py --output /tmp/vcobj-after-baseline.json
.venv/bin/python scripts/benchmark_object_store_workflows.py --worker patch --patch core/test/data/segments/20241113070770
```

The saved follow-up environment records the parent revision and working-tree
source hashes; the implementation was uncommitted when measured. Original
baseline reports are preserved unchanged.
