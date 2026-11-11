# Independent current-main integration review

Frozen commit: `7b08dfebd030168a6127f43e060301c55c7c20c3`; comparison baseline:
`371961148`. Reviewed the cumulative main integration, including `9d15061bc`,
not only the last merge's small incremental diff. Existing untracked
post-receipt review logs are preserved. No production/test edits, installs,
builds or runtime probes by this lane.

## Scope and independent dispositions

`git diff --name-only 371961148 7b08dfebd -- vc_object_store volume-cartographer`
returns no changed files. Consequently the previous receipt/container and
native store reviews remain applicable to unchanged code; they are not
represented as freshly re-read whole-file evidence. The integration surface
is shared `vc_reader` infrastructure and changed Python consumers.

| Changed family / inspected boundary | Disposition |
| --- | --- |
| `vc_reader/src/cpp/array.hpp`: channel registration, `selected_span`, region/point dispatch | Channel-aware cache keys retain the existing full array cache namespace plus physical chunk and channel. Shared source identity controls policy, not decoded-byte identity. Whole-chunk and pending data can seed selected planes; partial reads do not publish incomplete whole-chunk disk entries. No vcobj UUID/version owner reconstruction. |
| `channel_policy.hpp`; `zarr/open.cpp` | Policy registration uses immutable snapshots, while byte-cache identity retains encoding/quality distinctions. Child process registry abandons inherited locks. Repeated source registration accumulates channels. No call to `request_channel_cache` was found in vcobj/VC/Spiral/Vesuvius production trees: selected-channel mode is opt-in, not a silently changed store protocol. |
| `channel_decode.hpp`; `zarr_storage.hpp`; shard read integration | Inspected complete selective decoder and ranged storage boundary. Channel indexes and decoded partition sizes are checked; raw sizes, Blosc directories/blocks and read bounds are validated. Remote ranges retain size/revision checks. Whole-source paths still use shared storage. This is source evidence, not a fresh codec fuzz or network fault-injection run. |
| `memory_cache.hpp`, `available_memory.hpp` | Inspected budget, get/put/erase/clear, asynchronous reservation/retry and platform-memory integration. New shared byte budget/cache maintenance affects resource use and eviction, not vcobj selected identity. Admission retries release insertion locks before reclamation. Documented external buffer lifetime and asynchronous budget shrink mean cache budget is not a hard RSS bound. This is not an exhaustive proof of scheduler fairness or every concurrent interleaving. |
| Python `memory_cache.py`, `zarr_reader.py`, native bindings | Auto settings and channel registration are forwarded to native owners. Fork/pickle paths rebuild process-local engines and preserve registrations instead of reusing inherited native mutexes. The changed pickle tuple is process/cache configuration, not an alternate saved vcobj reference representation. |
| Spiral `satisfaction_metrics.py`, `spiral_helpers.py` | Evaluation can retry with per-quad arrays on CPU; model batches remain on their fit device and overlay gathers move only the requested batch. Read the device-change diff. No saved reference/receipt/schema path is modified. Numerical equivalence across CPU/GPU is not asserted by this source review. |
| Spiral `theta_crossing_map.py`, `losses.py` | Refresh invalidates and releases old device tables; tree events move through topology-device storage. Pair sampling no longer materializes the quadratic pair list. These are numerical/resource algorithms, not store placement or reference codecs. New RNG consumption is an upstream algorithm change, not a vcobj identity regression; no bit-for-bit stochastic trajectory equivalence is claimed. |
| Vesuvius loader and interpolation callsites | Loader uses explicit fixed-order products for tiled transform consistency. Source surface interpolation callsites request `bicubic` through the shared reader. No SQL/current-owner reconstruction or vcobj schema adaptation added. Whole training/full-volume numerical validation is outside this store integration pass. |
| `lasagna/tiled_predict3d.py::_omezarr_chunk_exists` | Adds `FileNotFoundError` to existing missing-output handling. This is the Zarr writer boundary, not a fallback vcobj resolver. |
| Prize automation/site/workflow changes | Separate product; excluded from vcobj/store integration review. They do not modify the store or shared reader path inspected here. No setup or automation executed. |

No concrete integration regression established by this independent source
pass. Old immutable vcobj identities, selected geometry/metadata pairing,
receipt admission, tombstone reservation, atomic writes, and physical
closure/relocation are unchanged. Their prior exact/local/referenced/history
coverage remains explicitly inherited rather than claimed as new runtime
execution against every reader configuration.

## Validation caveat

The coordinator found that the local runner stages current reader Python
source alongside native binaries from another checkout. New Python APIs
such as `MemoryCache.automatic` and channel registration require matching
native bindings. A failure caused by stale binaries is a runner provenance
problem, not evidence of a source-level vcobj regression. Conversely, a
passing old-binary run cannot certify the new reader implementation. The
coordinator owns matching-build and approved-runner verification; this lane
has made no fresh runtime claims.

## Final adjudication

Read `native.md` and `consumers.md` completely. Agree with their fresh
integration traces and explicit unchanged-evidence reuse. In particular,
HostedAttachment uses the generic LRU under its own lock, not the new
asynchronous CT-reader cache; shared decoded buffers survive cache eviction.
Fixed numeric cache budgets remain supported, but cache maintenance
scheduling has changed upstream, so unchanged resource/performance behavior
must not be inferred from unchanged constructor callsites.

The consumer lane correctly distinguishes patch/model ordinals from stored
selected identities and explicitly acknowledges RNG/device numerical
changes rather than calling the upstream algorithms bit-identical. No
candidate requires acceptance or a production fix in this pass. Both lanes'
source-only results are consistent with the independent integration checks.

Final source verdict at the frozen merge: **no additional confirmed vcobj
integration defect or regression**. All lane adjudications are complete.
Runtime validation remains coordinator-owned; the source/binary provenance
caveat above must be resolved before claiming merged-reader runtime coverage.
Limitations include no GPU/full-volume run, GUI interaction, macOS/arm64
execution, codec fuzzing or exhaustive concurrent-cache proof. This is not
a claim of a fresh line-by-line audit of all unchanged production code.
