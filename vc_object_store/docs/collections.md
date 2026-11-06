# Linked and sharded v2 stores

`ObjectCollection` is the C++ composed-store API over independent `ObjectStore` connections. It does not rely on SQLite ATTACH limits or pretend that multiple tables permit independent writers. Every write is one ordinary transaction in the local overlay; dependencies and published partitions are sealed. Plain unsharded SQLite databases remain valid inputs. The optional packed VFS uses exactly the same manifest and routing.

## Manifest and identity

| Table | Purpose |
| --- | --- |
| `collection_members` | Ordered dependency/partition database UUID, owning store UUID, expected generation and locator; reservation bounds only for local partitions |
| `collection_routes` | Structured object/edge key-space partition rules and physical destinations |

These tables require the `collection-v1` feature. Database and table identities are registered in the common v2 directory. Locators are relative to the manifest file where possible; packed-internal members use `@packed:<database UUID>`, not staging or absolute paths. A dependency's UUID, committed generation, sealed policy and reservations must match. Missing dependencies, incompatible formats, cycles, conflicting pins or ambiguous ownership fail at open. Moving a complete directory tree or a self-contained packed file preserves identity.

Ordinary dependency links do not duplicate the parent's ranges: their bound fields are zero and ownership is loaded from the dependency manifest. Only physical partition entries carry local reservations, which are checked against their independently created database.

The object or edge UUID is its complete logical key. `ObjectKey::storeUuid` is a routing/provenance hint, not another identity component: the same entity UUID resolves to the same visible record regardless of that hint. Numeric row IDs remain scoped to their reserved store key space and are not edge endpoint identities. An override stores the original UUID, stamped in the overlay's own generation; no alias is generated. Creating a dependent store publishes above every parent's generation. All linked parents must be sealed before linking; opening a mutable store readonly is not sufficient.

`members()` reports the manifest identities and generations observed when the collection was opened. Dependency generations cannot change; the mutable root may advance afterward. Use `provisioning(transaction).generation` for the root's current snapshot generation, not the cached root entry in `members()`.

## Overlay reads and writes

Resolution checks the local record by UUID first, then linked stores in explicit manifest order. A local replacement wins; a local `deleted=1` record stops fallback. Only an absent key falls through. The same rules apply recursively, so a deeper deletion does not accidentally resurrect an ancestor record. Enumeration deduplicates by UUID under the same precedence. The ordered bases are deliberate precedence, not unordered filesystem discovery. Repeated references to one physical database are coalesced after validating the pin. Duplicate UUIDs in independent import shards are rejected as duplicate records in one initial dataset, not reinterpreted as different entities.

Unchanged records and geometry come directly from dependencies. Editing inherited metadata creates an ordinary local typed record with the same UUID; it does not copy geometry. A relation stores the original endpoint UUIDs directly in `edges.a_uuid` and `edges.b_uuid` as 16-byte BLOBs. Local, sibling-shard and linked endpoints use the identical row representation: no endpoint registration, negative handle, alias, proxy object or per-reference metadata. Native endpoint validation resolves the UUIDs through the collection and rejects missing or retired objects. This is not a SQLite foreign key across databases. Reverse adjacency queries every relevant edge partition: incident edges do not have to live beside the endpoint object. Results exclude deleted edges and deleted endpoints.

Endpoint references resolve the visible object's current lifecycle and are not separately addressable records. A same-content `setMeta` or `replaceObject` does not create an override or advance generation. Metadata keys explicitly anchored by inherited relationships cannot change until those relationships are removed/recreated locally; unanchored classification fields remain editable. Indexed anchors validate against visible geometry without acquiring local descriptors or chunk registrations. An actual geometry edit creates the local descriptor, retaining unchanged chunk hashes for copy-on-write geometry.

Deleting an inherited object requires first removing its visible incident edges. The overlay stores the deletion on an `objects` record at that original UUID; removing an inherited edge similarly stores `deleted=1` in `edges` at the original edge UUID. There are no separate object-reference or edge-tombstone tables. History-on records local edits using the existing history machinery; it does not copy the parent's history. External relationship endpoints retain sealed-store/generation provenance in historical version references, not fabricated local endpoint object histories. Global historical snapshots across arbitrary collection generations are not part of this version.

Initial geometry can be created with `ObjectCollection::putGeometry`. Sparse edits use `editGeometry` and the observed geometry token. On the first inherited edit, the local object descriptor directly references unchanged parent chunk hashes; no chunk registration or NULL payload row is created. Resolution searches pinned readonly dependencies, including deeper links. Only chunks that need materializing or changing acquire local bytes. Metadata, verification, normal geometry editing and dependency validation use the common reader. Inherited incident relationships must be explicitly removed/recreated locally before editing their geometry, so the API does not silently leave inherited anchors applicable after a change. New geometry tokens are local publications; the parent's data remains unchanged. Native low-level access without the collection resolver fails when it reaches external payloads.

```cpp
using namespace vc::objects;
OpenOptions options;
options.author = "classification";
options.history = HistoryMode::Off;
options.spatialIndex = false;
auto derived = ObjectCollection::create("classification.vcobj", {"fibers.vcobj"}, options);
auto write = derived.write();
derived.setMeta(write, {baseStoreUuid, fiberUuid}, "family", R"("H")");
derived.seal(write); // a later normals overlay may now depend on this layer
write.commit();
```

Collection-wide spatial queries are not exposed without a collection accelerator. Opening the physical root explicitly does not make its local spatial index cover its dependencies.

Use `ObjectCollection`'s object/edge lookup and adjacency APIs for the logical cross-store graph. A native `ObjectStore::graphReadView` selects local objects in that physical database and returns their induced subgraph; it is not a collection-wide graph view. Low-level key-only traversal can follow stored endpoint UUIDs with lifecycle filtering disabled, but nonlocal metadata/kind/lifecycle resolution requires a pinned collection context and otherwise fails explicitly.

## Independent import shards

The [initial batch API](initial-import.md) owns the shared logical store UUID, generation and disjoint object/edge reservations. Each worker seals its shard and returns its descriptor through the existing writer. The coordinator directly writes a history-off, sealed routing manifest at that same logical UUID/generation; physical database and table UUIDs remain distinct. It does not reopen the shards or invoke `publishShards`. Partitions are not parent dependencies, so their common final generation is intentional.

The default routing uses disjoint inclusive positive-ID ranges. `ObjectStore::insertObjectWithId` and `insertEdgeWithId` support predetermined import IDs within their respective reservations; occupied IDs/UUIDs cannot be overwritten. Explicit edge insertion uses ordinary current-endpoint dependency validation, history and generation handling, without persistent duplicate current-binding rows. `ObjectCollection::insertEdgeWithId` accepts original endpoint UUIDs with store routing hints, including linked objects. Failed insertions and transaction rollback restore the allocation state; out-of-order explicit insertion never moves the ordinary cursor backward or reopens an exhausted range. Unpopulated IDs inside a reservation remain owned, never become free allocation space in another shard. Manifest-only databases have exhausted local allocators.

Explicit `KeyRoute` declarations additionally support multiple ranges per destination, modulo, selected bits, and a versioned hash. For each key space, one authoritative rule kind/divisor/shift is allowed; overlapping rules are rejected. The publisher checks custom routes against all stored IDs once. Opening uses the sealed declaration without rescanning every key. Numeric object lookup selects the declared destination; UUID lookup uses indexed queries in relevant partitions. Edge identity routing and adjacency scattering are separate operations. UUID identity is independent of these numeric routing rules; an endpoint reference never needs to be translated into the destination's local row ID.

For positive ID `x`, modulo routes on `x % divisor`; bit routing uses `(x >> shift) % divisor` with a power-of-two divisor. HashV1 applies unsigned 64-bit overflow arithmetic: `x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9; x = (x ^ (x >> 27)) * 0x94d049bb133111eb; x ^= x >> 31;` then modulo. Never substitute implementation-dependent `std::hash`.

`publishShards` is the separate discovery/validation operation for caller-supplied existing databases, not a batch-finalization step. It checks those inputs before constructing their manifest. Ordinary file completion uses a same-directory stage and atomic no-overwrite exposure with directory sync. A failed manifest is not a usable partially completed collection. For packed builds, `PackedStoreBuilder::finalize` is the single outer completion boundary; all independent SQLite handles must close first. This flushes retained data and finalizes the container, without reopening its SQLite databases.

`packObjectCollection(source, destination)` packages a sealed dependency closure into one packed file, preserving logical/database/table UUIDs and generations while replacing physical locator strings. A writable derived overlay can subsequently link the packed source.

## Limits and safety

There are no distributed transactions, mutable published shards, automatic parent-following, implicit ID renumbering or default spatial fallbacks. Sealing is an API contract and change-detection boundary, not cryptographic protection against malicious raw file edits. Keep a collection and its dependency closure alive for the lifetime of its transactions. Transactions are thread-confined, as with `ObjectStore`. One transaction holds read snapshots for the immutable dependencies it actually touches; it does not reopen databases for every lookup.
