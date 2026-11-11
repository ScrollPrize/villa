# Astra pre-fix adjudication

Independent review completed before implementation of the previously unadjudicated candidates. Scope: FM-P03, FM-C01, FM-C02 and the optional FM-A02 relationship-key cleanup. Read the individual findings, README runtime evidence, actual producer/consumer implementations and sibling paths. No usage-limit blocker, production edits, tests or builds in this review. Root's temporary runtime probes are reported evidence, not independently rerun here.

## FM-P03 — Surface numbered-selector coercion: accept

`surfaces.read_surface` applies `int(version)` before `ObjectStoreSession.read_surface_grid`; `2.5` therefore becomes 2 before native policy sees it. The source and root's successful version-2.5 probe agree. Complete-reference reads are unaffected; ordinary explicit positive numbered selectors remain legitimate.

Minimal fix: stop coercing the wrapper's version and have the shared native ingress reject noninteger/bool/string/overflow/nonpositive values. Merely removing `int` is sufficient only if the binding is demonstrated to reject every malformed form: `optional<int64_t>` language conversion may differ for booleans or integral-looking Python types. If needed, extract the existing positive-version JSON numeric validator from the shared reference codec and reuse it at selector adaptation; do not synthesize fake owner references just to validate a number or duplicate numeric rules in Python.

Retain None/current semantics, generation-only refusal, full-reference exclusivity, version-only current-owner discovery and exact reads after deletion. Check direct session, wrapper and delegated Vesuvius surface reads for valid integer/None/full-ref and malformed fractional/bool/string/overflow cases. Native C++ callers already use integer types; their positive/version-shape validation still belongs to shared selection policy.

## FM-C01 — Point-map decoding: accept shared extraction

`StoredCollection.cpp::storedCollection` uses permissive `stoll` and ignores `map::emplace` collision results. Native `formats/PointCollectionInput.cpp` already checks full numeric consumption, nonnegative int64 IDs, numeric aliases, XYZ shape and finite values, then preserves non-position annotations. Root's `1`/`01` collapse and `1junk` acceptance probes match the source.

Extract a small reusable native point-map decoder for sorted IDs, finite XYZ and annotations, then migrate both the store importer and GUI stored-collection adapter. A public in-memory decoder is preferable to invoking a file importer or copying its code. Preserve metadata outside the point map in application/source adapters. Preserve double-valued coordinates until an application deliberately converts its display types.

Keep document-level collision/renumbering policy separate: the importer detects identities shared across collections, optionally assigns new IDs, records source_point_id, and refuses renumbering linked documents. Intra-map numeric aliases must still reject, even if document renumbering is enabled. Do not silently transform an alias into a new point. Preserve overflow checks when allocating replacement IDs.

Standalone `PointCollections.cpp` has both point-map `stoull` and collection-key `stoull` paths. Inspect/migrate the common point-map boundary where compatible, but do not conflate document collection IDs with store point IDs or silently narrow a deliberately uint64 standalone ID contract. The standalone loader parses a complete temporary document before replacing application state; maintain that atomicity. If a standalone range difference is real, document an explicit adapter/range policy rather than retaining a weaker prefix parser. Normal generated canonical GUI JSON is valid and must remain unchanged.

Required regression siblings: importer and storedCollection for suffix/negative/overflow IDs, aliases, nonobject maps, bad XYZ/nonfinite data, annotation retention and deterministic numeric ordering; importer cross-collection collisions with renumber on/off and linked-document refusal. Existing standalone round trips must remain valid.

## FM-C02 — Saved object manifest references: accept shared ingress validation

`SpiralStoreInputs.cpp::sameStore` validates all saved relationship references, including before partial filtering. In contrast `objectPin` and `manifestPins` read UUID/kind/geometry/content fields without validating object reference identity, and the outer shell takes the same path. `manifestPins` is also called directly by `SpiralWorkspace.cpp`, so fixing only `sameStore` would miss a sibling ingress. Python `validated_manifest_references` currently enforces complete object references and duplicate membership.

Introduce shared native manifest-reference validation returning decoded complete ObjectVersion/RelationshipVersion handles; use it from Python and native ingress. Validate every object, outer shell and relationship before current-head filtering or partial-refresh selection. Reject absent/incomplete/wrong-object references, conflicting optional numeric versions and duplicate canonical object identities including shell collisions. Use shared UUID/reference codecs; do not create a second tuple or flattened representation. Carry returned handles into pins/comparison construction instead of reparsing them in every helper.

Keep the native domain's intentional **current-content comparison**. A valid saved reference need not still have retained historical payloads merely to validate its syntax/identity: do not replace current display with historical reads or fail history-off comparison just because old data is unavailable. Consumed metadata projections and geometry/content-change comparisons remain application policy beside references, not substitutes for them. The shared validator should own reference/membership semantics, not absorb the entire Spiral manifest schema or solver policy.

A direct public `manifestPins(document)` call must validate independently; within an already-validated read/partial-refresh operation, use an internal path accepting the decoded result so validation happens once. Validate outer-shell and dropped relationships even when not materialized in that refresh. Missing domain content metadata is a separate backward-content-comparison policy, not license to omit saved identity.

Required checks: complete valid manifest unchanged; stale but well-formed references comparable without retained payload; missing/malformed/wrong-object refs; duplicates (same/different ref and shell collision); mismatched optional versions; full read, direct manifestPins, partial refresh and Python ingress. Root's malformed-object probes demonstrate the gap but do not alone cover these siblings.

## FM-A02 — Relationship-key consolidation: approve with semantic constraints

No supported incorrect-answer collision was demonstrated; this remains an authorized simplification, not a newly proven bug. `relationshipsAtSelected` currently mixes archived revision UUIDs and JSON array strings for current-only relationships in membership/dedup state. A safe refactor resolves physical archive membership to selected RelationshipRevision values once, carries their full RelationshipVersion references, and uses one in-memory reference-keyed result/membership path.

Preserve these exact decisions:

1. Keep each endpoint's own archived membership; do not replace it globally with current relationship listings.
2. Overlay/shard current bindings are admitted only when recorded endpoints match every explicitly selected endpoint; missing off-screen endpoints retain existing behavior.
3. Keep root-first scope precedence and per-endpoint seenBindings behavior. History-off owner scope is still included only where needed to recover its current-only binding; do not fabricate archive rows or copy payloads.
4. A discovered current binding overrides that endpoint's selected membership for the same relationship UUID exactly as before. The final result remains the union of the resulting memberships, with complete-reference deduplication.
5. Preserve endpointApplies and the existing fallback when one endpoint lacks explicit membership but its recorded reference matches; do not simplify applicability to numeric versions or presence alone.
6. Resolve each distinct archive locator once into the owning selected relationship. Physical archive UUIDs remain legitimate storage locators, but do not retain a second synthetic JSON identity in the operational maps. Add ordering support for the shared RelationshipVersion value if needed, rather than serializing it to make a map key.

Check public output ordering deliberately. The old set order was an incidental lexical order over two encodings; a full-reference map produces another deterministic order. Preserve any documented ordering and do not claim byte-for-byte ordering equivalence without checking callers/tests. Current/exact graph callers often sort relationships themselves, but raw relationships_at_versions consumers also need inspection. If necessary define one shared deterministic result order instead of preserving obsolete identity strings solely for sorting.

Required matrix: history on/off source and root combinations, local/referenced/sharded scope where fixtures exist, metadata-only endpoint versions, current-only relationships, off-screen endpoints, competing full references with equal numeric counters, no-op membership and selectedInputs/graphReadViewAt/Around sibling behavior. Compare reference/applicability sets and documented order before/after; do not add a persistence migration.

## Overall

All three previously unreviewed correctness candidates are approved for small shared fixes. The optional relationship-key cleanup is approved under the constraints above. Older omissions remain attributed to pre-fusion code, and no fix may reintroduce legacy numeric guards, ownerless references, historical-current fallback, new persistent identity tables or implicit tool repinning. Final delta review and regression verification remain required after implementation.

## A02 raw-consumer ordering follow-up

Repository call search and source trace support canonical sorting by relationship UUID, then complete RelationshipVersion as a deterministic tie-breaker. No existing public raw-result ordering promise was found in `ObjectStore.hpp` or reading documentation. However, this is an intentional observable ordering change, not byte-for-byte equivalence:

- `geometry/SourceMetadata.cpp::nativeMetadata` sorts its emitted per-point links and winding-linked IDs after processing relationships; raw order does not survive.
- `formats/FiberJson.cpp::fiberNetworkJson` appends `branches` / `adjacent_branches` directly in raw result order. Export JSON array order can therefore change; relationship/content sets should remain identical. Its graph traversal finishes in sorted UUID sets/maps, so node inclusion does not depend on raw edge order.
- `SpiralStoreInputs.cpp::compareRelationships` inserts current relationships into a UUID-keyed map and emits fitted entries in manifest order; normal current selections do not depend on raw order.
- Python `object_store_inputs.filter_relationships` sorts consumed relationships by family/endpoints/UUID before manifest/materialization; raw order is not its final domain order.
- `LineAnnotationController` stored-fiber loader appends branch vectors in returned order. This can change UI/serialized branch-list order. No first-element semantic dependency was found in the bounded call trace, but the entire controller's branch algorithms were not re-audited.
- `Traversal.cpp::graphReadViewAt` sorts selected relationships by UUID; `graphReadViewAround` resolves endpoint choices by complete owner/lineage and maximum version, not by arrival order.
- Python/hosted/session adapters forward the returned order; existing singleton tests do not establish multi-edge parity.

One cache caveat needs implementation attention: `HostedAttachment::relationshipsAt` can return an already persisted graph-result payload under the unchanged schema21+commit key. Such a payload preserves old ordering even after native results adopt a canonical order. If canonical ordering is documented as a public guarantee, normalize through one shared selected-relationship ordering helper at the cache/result boundary too, or explicitly version only the graph-metadata cache representation. Do not duplicate sorting rules in each binding or invalidate raw geometry/artifact cache content. Test a multi-edge local/hosted/reopened-cache result and fiber export order under the chosen policy.
