# Native vcobj / current-main integration review

Frozen merge: `7b08dfebd030168a6127f43e060301c55c7c20c3`, incorporating `origin/main` `f0e441016`. Comparison base: previously reviewed `371961148`. Read-only source review: no production/test changes, installs, builds or concurrent test runs.

## Result

No concrete native vcobj correctness or shared-API integration defect established. The cumulative comparison contains **no changes under `vc_object_store`**. Upstream reader changes were nevertheless followed through native store/GUI sampling consumers, rather than assuming an unchanged store tree proves integration compatibility.

This report covers store boundaries and reader integration. It does not claim a new exhaustive proof of the upstream cache's eviction algorithm, all codec kernels, or third-party implementations. Actual compilation/test outcomes belong to the coordinator's approved runner.

## Cumulative diff and fresh integration coverage

The complete cumulative changed-file list was enumerated from `git diff --name-only 371961148 HEAD`; changes to prizes/site automation and independent Spiral model losses are outside this native store lane. Native reader changes are in `array.hpp`, `memory_cache.hpp`, `lru.hpp`, `channel_policy.hpp`, `channel_decode.hpp`, `available_memory.hpp`, `shard_index.hpp`, `zarr_storage.hpp`, `zarr/open.cpp`, bindings and build/test declarations.

| Fresh-reviewed boundary / siblings | Disposition |
| --- | --- |
| `vc_reader/src/cpp/lru.hpp` → `vc_object_store/src/HostedSession.{hpp,cpp}` and `SurfaceTileCache` | Upstream adds replacement/erase/maintenance helpers to generic `vcr::Lru`; existing `get`/`put`/`clear` retain their interface and semantics. HostedAttachment's record/geometry/payload cache is this generic LRU under its own mutex, **not** the new asynchronous `MemoryCache`. Cache entries retain complete selected metadata/reference keys. No cache-owner or reference reconstruction change required. |
| `memory_cache.hpp` → `CtSheetNormal.cpp`, `OpenedVolumes.cpp`, native Lasagna sampler/cache callers | Fixed-byte constructor remains supported; returned chunk storage is a shared pointer, so eviction does not invalidate a consumer-held decoded buffer. Maintenance joins at cache destruction. New automatic budgeting is opt-in; fixed cache callers do not accidentally opt into a changing memory budget. Background maintenance is now default behavior of the reader cache, documented upstream, not a new vcobj history/transaction worker. No performance/RSS measurement is claimed here. |
| `array.hpp` / `channel_policy.hpp` → native opened-array sampling | Channel policy is source-identity shared and opt-in. Existing `CtSheetNormal`/LasagnaSheetNormalSampler paths use ordinary opened arrays and `read_many`/sheet-normal readers, not a locally reimplemented channel cache. Whole-chunk requests explicitly ask for all channels once policy is enabled. No vcobj geometry chunk or selected-object reference is represented as a Zarr channel-cache key. |
| `zarr/open.cpp` source identity / decoded identity | New `channel_identity` is separate from encoded-quality-sensitive memory-cache identity. Remote decoded identity still includes `allowLossyCache`; channel registration does not make a quantized decoded chunk satisfy an exact decoded reader. Existing consumers pass shared cache pointers through OpenOptions rather than reconstructing cache keys. |
| `zarr_storage.hpp`, `shard_index.hpp`, `channel_decode.hpp` selective source boundary | Local shard range reads and remote ranged channel reads live in shared reader source/codec code. Native vcobj consumers do not perform their own partial-file cache writes or codec interpretation. Remote selected reads validate ranges/source revision; partial bytes are not published as a complete disk-cache object. Full-source paths remain available to ordinary native readers. This is source integration tracing, not new corrupt-codec/ranged-server fault injection. |
| Native caller siblings | Enumerated `MemoryCache` and channel-policy use in `volume-cartographer` including OpenedVolumes, FiberTraceEvaluationCli, grow/render/fiberlet tools, LasagnaGroupArrays, native normal samplers and fiber-prediction option structs. Existing constructor/open-options calls remain supported. No caller-specific format, UUID routing or version-history adaptation was introduced by these upstream changes. |

Python fork/cache-wrapper changes are owned by the consumer/reader lane. Native inherited-cache use after fork is explicitly unsupported upstream; this pass did not establish a new native vcobj caller doing that. No speculative fork workaround or extra per-consumer cache implementation is proposed.

## Broad native contract reopening

The following invariant families were cross-checked against the merged source and cumulative changes, not limited to an upstream cache compile signature:

| Shared family / entry points | Result |
| --- | --- |
| `ObjectCollection` UUID routing, scope callbacks, linked-store loading and transaction contexts | Store/database/lineage/commit pins, readonly dependencies, shard-before-base precedence, and winning tombstones remain native responsibilities. Reader caches do not replace those callbacks or expand their scope. Exact reference lookup and visible-current selection remain distinct. |
| `History` exact/current object and relationship reads; publication snapshots; `Versions::relationshipsAt` | Complete owner/generation/version identities retained. Public selected relationship API rejects duplicate selected object UUIDs before the private membership map. Optional history follows the winning owner's policy. No reader cache field substitutes for selected identity. |
| `RecordResolver`, `RecordOverrides`, `Records`, `Metadata`, relationship writes, `Transactions` | Shared visibility/tombstone resolution and operation savepoints remain in place. Overrides preserve before-state descriptors without copying referenced payloads. Checked-action failures roll back; unchanged inherited receipt outputs retain their actual owner. No new local consumer SQL path or proxy table appeared in merge. |
| `Versions` receipt codec; `Commands`, hosted submit/receipt and saved replay; draft/document repinning | PCS-01 strict map/scalar admission and two-way live-result/full-reference consistency remain intact. Shared decode used for all ingress siblings. Retained geometry, artifacts and metadata are not repinned by a numeric receipt/root-owner combination. |
| `GeometryReference`, `GeometryReader`, selected `GeometryViews`, geometry export | Retained revision resolution uses actual source/descriptor; selected object metadata survives into transformed views/export. Decoded CT/Zarr caches are separate from vcobj geometry revision identity and its retained chunk resolution. |
| `Extensions`, `ExtensionHistory` current/exact scans and writes | Extension rows resolve from winning owner metadata or exact supplied owner. Snapshot chunks use shared payload lookup; append/replacement savepoints and no-op behavior are unchanged. No upstream cache policy introduces new extension provisioning or version counters. |
| Initial import, source topology, snapshot import, native source/output adapters | Common source admission and batch rollback retained. Live equal-content snapshot import remains idempotent; conflicting content/cross-kind/deleted UUID reservations reject. UUID routing and source ordinals retain their separate documented meanings. |
| `PackedStore`, `PackedCollection`, copy/export closure, artifacts | Shared framing/member admission and locator relocation remain. Packing/copying dependency closure is not replaced with physical-root-only copying. No new reader path interprets linked object placement outside the store layer. |
| Shared JSON/native-wire/selection boundaries | Prior array admission and strict complete-reference codecs remain unchanged. Typed array iterations are legitimate consumers, not missing per-iteration checks. No legacy relationship-archive repair or alternative command format reintroduced. |

## Reused evidence / explicit limits

The preceding full native inventory and detailed sibling dispositions remain in [post-receipt/native.md](../post-receipt/native.md), [class-sweep/native-references.md](../class-sweep/native-references.md), and the underlying per-file core/format/spatial/linking ledgers. Their unchanged file-level evidence is reused openly; it is not claimed as a new numerical line-by-line audit. Fresh work here is cumulative-diff enumeration, changed reader API/storage boundary tracing and the broad invariant cross-check above.

No new finding candidate required severity/regression classification. No builds, GUI interaction, native fork experiment, codec fault injection, HTTP fault server, macOS/arm64 run, concurrency stress or full-volume throughput run was performed in this lane. Upstream reader tests/build declarations were inspected as evidence of intended semantics, not counted as tests executed by this agent.
