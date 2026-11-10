# R3 storage, transport and linking review

Baseline: `cf8406afc`. Review only, no production changes. Composite-write atomicity is folded into core finding C3-4, not counted twice.

## T3-1 — P2: packed compaction changes relative external locator targets

`src/PackedStore.cpp:1073–1080`: `compactPackedStore` imports physical members into the destination builder without the shared relocation used by `packObjectCollection`, copying and conversion. Moving the file to another directory changes relative `storage_ref` and `tracks.dataset_ref` targets, including retained records.

Runtime probe: a sealed base in directory `a` has dataset `data.tracks`; pack in `a`, compact into `b`. The compacted store resolves `b/data.tracks` instead of `a/data.tracks`. This was observed before the later environment failure below. Fix through shared private-copy relocation/finalization, not consumer fallbacks or logical revision changes. Test current and retained relative locators when compacting to another directory.

## T3-2 — P2: hosted graph disk cache treats mutable membership as immutable

`src/HostedSession.cpp:602–612` caches `relationshipsAt(versions)` by endpoint references, attachment UUID and generation. `payloadKey` (70–74) omits commit sequence. The cached result's `asOf` status is discarded without freshness validation; refresh does not invalidate this disk entry. `src/History.cpp:929–1015` can include current overlay edges recording those exact endpoints. Adding an overlay edge does not need to change borrowed endpoint identities. An empty cached graph can therefore remain empty after refresh/reopening.

Evidence: code trace, not runtime confirmation. The prepared server probe could not run after its final edit because runtime libraries are missing. Fix through the shared graph-query snapshot/freshness contract, separately from immutable payload caching. Test edge add/remove around inherited endpoints with unchanged references, refresh and reopen the cache, comparing direct hosted reads.

## T3-3 — P2: track grouping confuses raw dataset locators with identity

`src/linking/TrackTrack.cpp:49–80` groups tracks by raw `datasetRef` and requires source ordinals unique within each group. Shared override/copy code legitimately rebases those locators. The same dataset can split after overriding one inherited track (`data.tracks` versus `../a/data.tracks`), losing crossings. Conversely unrelated referenced stores using `data.tracks` with overlapping ordinals merge and fail duplicate-ID checks or produce inappropriate comparisons.

Evidence: code trace. Relative dataset locators are supported elsewhere; absolute-only ingestion is not the contract. Use shared selected-record source resolution for canonical dataset identity before grouping/filtering; do not add a detector-local path resolver. Test equivalent relative locators and identical relative strings owned by different directories.

## T3-4 — P2: transform-choice creation lacks a membership guard

`src/VolumeTransforms.cpp:107–129`: when no choice exists for a volume pair, the session helper guards a newly minted UUID and existing registrations, not the absence of a choice. Two concurrent calls can observe absence, create different UUIDs and both commit. `volumeTransformsFrom` (174–179) then throws for multiple active choices.

Evidence: code-traced interleaving, not runtime reproduction. Reuse shared selection/commit guards, as the umbilicus request already does, rather than new UUID uniqueness or a local-only SQL constraint. Test two requests prepared from one no-choice snapshot; the second must conflict.

## C3-4 sibling: catalog-transform batch atomicity

`ObjectStore::recordCatalogTransforms`, `src/VolumeTransforms.cpp:211–237`, validates and writes registrations incrementally without an operation-wide savepoint. A later invalid/duplicate registration throws after earlier mutations. Catching and committing preserves partial import. Code-traced sibling of the core report's reproduced surface-creation failure; use the same shared operation boundary.

## Coverage and verification

Freshly inspected store-sensitive implementations/callers: collection routing and copy, packed collection/VFS allocation/cache/writeback/finalization, source paths, record overrides and ownership; initial/parallel/AFV import and provisioning; local/hosted sessions, fetch queue, project/tool/checkpoint helpers; artifacts, conversion/export publication; transforms, umbilicus, link inputs, stored geometry, track grouping, candidate publication and pinned surface-overlap loading. Relevant hosted server/client endpoints and wire identity codecs were traced. Core, extensions, native consumers and Python have separate coverage reports.

This is not full numerical/GUI algorithm inspection, every HTTP/parser line, platform testing or a performance benchmark. Trusted import omitting redundant validation is intentional, not a finding. Numeric-only publication helpers were inspected, but no additional reachable identity-collision defect was established and hypothetical cases were excluded.

The fixed local command is `bash /home/hendrik/villa3/local/scripts/run_vcobj_review_checks`. Its reusable approval is saved. It accepts no arguments, runs explicit checks with a 60-second timeout, allows a temporary loopback server and automatically removes fixtures. No installation or production-data operations. Reuse this entrypoint instead of approving individual executable paths. Shell syntax validation passed.

The root probe source/executable are local-only. Initial execution confirmed T3-1; its hosted part needed a client-minted edge UUID. After that probe correction, the current environment cannot load the existing binary: `libopencv_core.so.406`, `libblosc.so.1`, `libtiff.so.6`, `libjpeg.so.8` are missing. No installs or incompatible aliases were added. T3-2 stays code-traced. No full regression suite was rerun for this review-only pass.
