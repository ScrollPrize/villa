# Spiral: resident fitting inputs from the store

The priority for Spiral is **steady-state fitting performance**, not cold startup
latency. The store now supplies packed, verified geometry arrays that can remain
resident alongside the graph's CSR arrays. Once inputs are prepared, ordinary
NumPy indexing performs point gathers and graph traversal without database calls.
No export, sidecar, alternate database or persisted packed format is needed.

[API and examples](../object_storage.md#consumer-queries-and-store-backed-graph-reads) ·
[Previous implementation measurements](performance-improvements.md) ·
[Raw results and validation](performance/2026-10-19-spiral/)

## Steady-state result

The same 512 random point indices into a 24 MiB float64 XYZ array were gathered
using each path below. Expected bytes/values are checked outside the timed region.
The two NumPy cases use the same indexing operation and precomputed indices.

| Access path | Median ms | Min–max ms |
| --- | ---: | ---: |
| Repeated store range reads | 14.9729 | 14.8999–15.5604 |
| NumPy gather from store-loaded resident array | **0.00560** | 0.00509–0.00765 |
| NumPy gather from warm mmap array | **0.00546** | 0.00458–0.03258 |

The two resident-array paths are comparable in this microbenchmark. This is not
an end-to-end fit measurement: it establishes that using the store as the input
source does not require SQL or chunk verification inside the point-gather loop.
The older Python-list mmap comparison is also retained in the raw report, but
matching NumPy operations provide a more relevant comparison for Spiral.

One thousand neighbor slices from the graph view take **0.227 ms**. The CSR
`offsets`, `partners`, `edge_indices`, and `endpoints` arrays are also directly
available for vectorized/native traversal, without calling a Python method for
each node. Obtained topology and geometry arrays survive closure of the view.

## Implementation

`graph.read_geometry_arrays(array_name, nodes=None)` reads pinned geometry
revisions into a `PackedGeometryArray`. `values` is a read-only NumPy view with
the original dtype, byte order and dimensions; `offsets` identifies the first-axis
slice for each requested node. Node subsets preserve order and duplicates. The
store-level `read_geometry_arrays(revision_uuids, array_name)` and C++
`readGeometryArrays` expose the same bulk reader for known revisions.

The native reader batches manifest and chunk-reference queries, validates the
manifests, allocates the packed payload once, and copies verified chunks directly
into that buffer. Tiled arrays are reconstructed in their logical row-major order.
Manifest checks, chunk checks and full-array chunk copying are shared with the
ordinary reader. There is no additional Python bytes copy when obtaining NumPy
views. A caller can partition selected nodes to bound resident memory.

Graph construction now uses the shared object/edge selection queries to obtain
numeric IDs and topology directly. It avoids record hydration, repeated UUID
lookup/conversion, and reconstructing endpoints through Python records. Filters,
node/edge ordering, geometry revisions, approval policy, parallel edges and
self-links retain their existing semantics. This pass adds no schema migration.

Prepare crossing metadata, source-ID mappings and point arrays once from the
same pinned view, then close it and retain the arrays for fitting. Reopen explicitly
to accept changed inputs. Store-backed arrays are working memory, not a second
persistent representation of the graph.

## Startup measurements (secondary)

Cold startup latency is not the acceptance criterion. These measurements are
included to make the preparation cost visible; all use warmed benchmark runs,
so they are **not cold filesystem-cache timings**.

| Preparation | Previous path | New path |
| --- | ---: | ---: |
| Graph view, 1k nodes / 1k edges | 6.53 ms | 1.63 ms |
| Graph view, 10k nodes / 10k edges | 82.19 ms | 26.00 ms |
| Graph view, 50k nodes / 50k edges | 508.47 ms | 145.63 ms |
| 10k small tracks, known revisions | 101.10 ms, per-track reads | 71.52 ms, packed read |
| 24 MiB fiber, complete payload | 31.36 ms, ordinary read in this run | 23.75 ms, packed graph read |

The graph fixture is the existing synthetic track/crossing ring, including all
approval states and no geometry payload. The track fixture contains 10,000
100-point int32 tracks. Its fresh baseline also measured **200.51 ms** for
per-track current-head lookups plus reads. Loading those tracks in graph-node
order from an already-created view takes **82.87 ms**; view construction is
separate, and graph order differs from the original insertion order.

Copying the same tracks from packed mmap storage remains much faster at
**1.21 ms**. Verified store loading still performs manifest decoding and integrity
checks. This difference is accepted as preparation overhead; it should not be
reintroduced into fitting iterations. Likewise, repeatedly calling the packed
loader repeats its verification work: retain the returned arrays instead.

## Profiling and validation

Native `perf` sampling was unavailable (`perf_event_paranoid=4`). Python cProfile
places the per-track loading time at the native-call boundary and cannot resolve
native functions. Temporary native stage timing and SQLite query plans exposed a
planner problem in the initial packed-reader prototype: it scanned the chunk
reference table for every batch. Chunk-stage time was about **80 ms** for 10k
tracks. Keeping requested revisions outermost with `CROSS JOIN` reduced that
stage to about **24 ms**. Manifest processing was about **39 ms**; subsequent
changes removed descriptor copying/revalidation and repeated output-buffer growth.
The saved stage profiles describe those intermediate prototypes. All timing
instrumentation was removed before the final build, tests and benchmark runs.

The Ninja build succeeded. **177 Python object-store tests and 29 native cases**
passed. New regression coverage checks packed-array order across SQL batch
boundaries, duplicates and empty arrays, missing/incompatible inputs, multichunk
and mixed tiled/linear layouts, exact special floating-point bits, read-only view
ownership after closing the store, pinned revisions during writes, corruption
and missing-chunk detection after an earlier successful read, and graph selection
parity with ordinary selectors.

The checked-in real TIFF fixture `20241113070770` (129 × 129 × 3) was copied to a
temporary directory, ingested, and read through the packed graph path. Its full
payload matches the ordinary reader byte-for-byte. This validates a real geometry
input, but a real Spiral training dataset and complete fitting loop have not yet
been connected. Source/point identity mapping, coordinate conventions and crossing
metadata interpretation remain responsibilities of that consumer adapter.

Measurements use the existing arm64 RelWithDebInfo build, native Python bindings,
one warm-up and three timed samples per case. Reports retain mean, median, min,
max and all samples; the environment includes source hashes for the uncommitted
working tree. Original benchmark archives remain unchanged.

```sh
ninja -C build vc_object_bindings test_object_store test_object_records test_object_links
VC_BUILD_PYTHON="$PWD/build/python" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest python/tests/test_object*.py -q
build/bin/test_object_store
build/bin/test_object_records
build/bin/test_object_links
.venv/bin/python scripts/benchmark_object_store_workflows.py --output /tmp/vcobj-spiral-workflows.json
.venv/bin/python scripts/benchmark_object_store_metadata.py --output /tmp/vcobj-spiral-metadata.json
.venv/bin/python scripts/benchmark_object_store.py --output /tmp/vcobj-spiral-baseline.json
.venv/bin/python scripts/benchmark_object_store_workflows.py --worker patch --patch core/test/data/segments/20241113070770
```
