# Shared store API contract

This contract applies to native store code, Python bindings, hosted transport and all consumers. API usage is documented in [reading](reading.md), [writing](writing.md), [editing](editing.md) and [collections](collections.md); persistent storage is specified in [store format](store-format.md).

## Resolution and identity

Physical placement belongs to the native store layer. Local, referenced, sharded and packed records use the same resolution APIs; consumers must not query physical SQLite tables or implement placement-specific fallbacks. Extend a shared API when it lacks a capability, then migrate its callers rather than adding partial resolvers.

Follow the [versioned-reference contract](versioned-reference-contract.md): resolve once and retain the complete selected `ObjectRevision`/`ObjectVersion` through metadata, geometry, transforms, artifacts, adapters and caches. The attached store is a lookup scope, not necessarily the selected record's owner. Compare complete identities, not equal numeric versions. Exact retained reads do not require a live current object and must not substitute current metadata.

Current lookup applies overlay precedence and stops at the winning tombstone. Exact lookup resolves the requested immutable identity. These are distinct operations, not alternative physical-placement implementations. History policy controls retention, not identity or routing. Do not add proxy records, per-object routing directories, copied referenced payloads or history-off reference formats.

Python bindings adapt language types; native code owns shared schema interpretation, validation and serialization. Saved/wire input is admitted once at its owning decoder, including declared collection shapes and numeric ranges. Typed downstream consumers do not repeat schema checks or repair obsolete formats. Application-owned metadata has its own schema, but embeds shared references unchanged.

## Selected geometry and evidence

Geometry, metadata, channels and coordinate mappings must describe the same selected state. Preserve explicit validity masks. Geometry-derived lengths, bounds and counts come from selected geometry, not ingestion summaries. A geometry payload UUID alone is not an exact metadata-version or edit-authority handle.

Persisted caches and solved evidence identify the source revisions actually consumed, including complete owner/lineage/version identities. Never stamp old results with freshly read current references. An immutable endpoint does not make relationship membership immutable: retain loaded membership and publication guards, and key membership-dependent caches by observed store freshness. Successful receipts advance observed freshness without consuming change-feed events; no-op receipts do not erase existing guards.

Geometry-fitting consumers use native `fittingRelationshipInput` (Python `fitting_relationship_inputs`) for relationship families, eligibility and endpoint ordering. Do not maintain separate policy tables in applications or bindings.

## Checked edits and publication

Edits of loaded data use `expectObject(request, selected.reference())`, with the loaded coordinate-mapping and membership guards. An unchanged object reference does not imply unchanged transform registrations. Deferred edits retain their loaded state until submission; they must not reconstruct a baseline from current heads.

Use shared identity, endpoint, tombstone and UUID canonicalization APIs for writes as well as reads. Local SQL uniqueness cannot enforce identity across a collection. Determine whether an edit changes anything before creating an override: no-op edits preserve ownership and versions, while required relationship revalidation still runs.

Repin from the receipt's complete object references, not its committing store plus numeric projections: unchanged inherited inputs keep their original owners. A receipt containing an unchanged endpoint does not authorize advancing another editor's relationship-membership baseline. Failed saves preserve pending edits; uncertain request outcomes retain the request identity for recovery rather than generating a replacement request.

Structural point IDs come from the shared retained-allocation API, not a restorable metadata counter or maximum surviving point ID. A foreign source slice does not establish ownership of an existing target point ID.

Public composite/batch writes roll back the complete operation even if the caller catches an error and commits its surrounding transaction. Trusted initial construction uses the dedicated batch APIs, not interactive edit bookkeeping.

## Queries, history and physical copies

Query optional history through the selected owner's policy, not the attachment's history flag. Each store has its own publication timeline; exact endpoint references alone do not reconstruct arbitrary historical collection membership.

Apply shadowing before admitting bounded query results, or refill pages while charging cumulative work budgets. Discarded candidates still count toward processing limits. Spatial queries require their documented accelerator; they do not silently construct one or switch to an unbounded scan.

Copy/export collections through shared dependency-closure APIs, not a backup of only the physical root. Physical copy, packing, conversion and compaction preserve external locator targets through the shared relocation helper, including retained historical records. Artifact bytes resolve through the pinned scope; metadata overrides do not copy borrowed sidecars. Exports and optional report files protect backing databases/artifacts and publish atomically.

## Validation requirements

Trace an invariant through all producer, binding, transport and consumer entry points, including read/write siblings. Distinguish legitimate boundary adaptation from duplicated policy. A passing unrelated suite or a file inspection does not establish that every caller obeys the contract.

Regression cases cover local and referenced data, sharded/packed placement, local/hosted transport, source/destination history-policy combinations, current and retained exact reads after override/deletion, no-op writes, and different owners with equal version numbers where applicable. For batch failure tests, catch the failure and commit the enclosing transaction to verify operation-wide rollback. Add mapping/membership races and retained point-ID checks where the edited path depends on them.

Review changed shared APIs together with their callers and document capability limits rather than introducing consumer workarounds. Keep transient findings, coverage ledgers, plans and run logs in ignored local storage; permanent docs describe supported behavior, contracts and usage, not review chronology or historical test totals.
