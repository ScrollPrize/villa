# Linked and sharded v2 stores

`ObjectCollection` is the C++ composed-store API over independent `ObjectStore` connections. It does not rely on SQLite ATTACH limits or pretend that multiple tables permit independent writers. Every write is one ordinary transaction in the local overlay; dependencies and published partitions are sealed. Plain unsharded SQLite databases remain valid inputs. The optional packed VFS uses exactly the same manifest and routing.

## Manifest and identity

| Table | Purpose |
| --- | --- |
| `collection_members` | Ordered dependency/partition database UUID, owning store UUID, expected generation and locator; reservation bounds only for local partitions |
| `collection_routes` | Structured object/edge key-space partition rules and physical destinations |
| `collection_refs` | Local endpoint/override handle → qualified inherited object, reference/override/deleted state, publication generation |
| `collection_edge_tombstones` | Qualified inherited edge deletions and generation |
| `collection_external_chunks` | Content hash → immutable source database UUID and exact byte length |

These tables require the `collection-v1` feature. Database and table identities are registered in the common v2 directory. Locators are relative to the manifest file where possible; packed-internal members use `@packed:<database UUID>`, not staging or absolute paths. A dependency's UUID, committed generation, sealed policy and reservations must match. Missing dependencies, incompatible formats, cycles, conflicting pins or ambiguous ownership fail at open. Moving a complete directory tree or a self-contained packed file preserves identity.

Ordinary dependency links do not duplicate the parent's ranges: their bound fields are zero and ownership is loaded from the dependency manifest. Only physical partition entries carry local reservations, which are checked against their independently created database.

Entity identity is `ObjectKey{ownerStoreUuid, objectUuid}`. Integer row IDs are only unique within the owner's reserved key space. An override preserves the inherited key, while its changed record is stamped in the overlay's own generation. Creating a dependent store publishes above every parent's generation. All linked parents must be sealed before linking; opening a mutable store readonly is not sufficient.

`members()` reports the manifest identities and generations observed when the collection was opened. Dependency generations cannot change; the mutable root may advance afterward. Use `provisioning(transaction).generation` for the root's current snapshot generation, not the cached root entry in `members()`.

## Overlay reads and writes

Resolution considers local overrides/tombstones first, then parents in explicit manifest order, then original records. The ordered bases are deliberate precedence, not unordered filesystem discovery. Repeated references to one physical database are coalesced only after validating the pin. Multiple original objects with the same qualified UUID are an error.

Unchanged records and geometry come directly from dependencies. Editing inherited metadata creates a local typed record and qualified identity mapping; it does not copy geometry. Local edges refer to lightweight local endpoint handles, so their ordinary foreign keys remain meaningful. The collection resolves those handles to inherited keys and validates anchors against inherited geometry. Reverse adjacency queries every relevant edge partition: incident edges do not have to live beside the endpoint object. Results exclude tombstones and deleted endpoints.

Endpoint handles preserve the original lifecycle and are not separately addressable as owned objects. A same-content `setMeta` or `replaceObject` does not create a proxy or advance generation. Metadata keys explicitly anchored by inherited relationships cannot change until those relationships are removed/recreated locally; unanchored classification fields remain editable. Indexed anchors acquire local geometry descriptors and pinned external chunk references, so native geometry-dependency validation remains active without copying the geometry bytes.

Deleting an inherited object requires first removing its visible incident edges; the overlay stores a tombstone without changing the parent. Removing an inherited edge similarly records a qualified tombstone. History-on records local edits using the existing history machinery; it does not copy the parent's history. Global historical snapshots across arbitrary collection generations are not part of this version.

Initial geometry can be created with `ObjectCollection::putGeometry`. Sparse edits use `editGeometry` and the observed geometry token. On the first inherited edit, the local descriptor references unchanged parent chunk hashes via pinned external references; only chunks that need materializing or changing acquire local bytes. Metadata, verification, normal geometry editing and dependency validation use the common reader. Inherited incident relationships must be explicitly removed/recreated locally before editing their geometry, so the API does not silently leave inherited anchors applicable after a change. New geometry tokens are local publications; the parent's data remains unchanged. Native low-level access without the collection resolver fails when it reaches external payloads.

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

## Independent import shards

Reserve one logical store UUID and one publication generation before starting workers. Each worker creates a separate database with that UUID, `initialGeneration = publicationGeneration - 1`, and its object/edge reservations. Import and seal in one transaction, so every shard commits at the reserved generation. The manifest created by `publishShards` is history-off, sealed and has the same logical UUID/generation; physical database and table UUIDs remain distinct. Partitions are not parent dependencies, so their common publication generation is intentional.

The default routing uses disjoint inclusive positive-ID ranges. `ObjectStore::insertObjectWithId` and `insertEdgeWithId` support predetermined import IDs within their respective reservations; occupied IDs/UUIDs cannot be overwritten. Explicit edge insertion uses the ordinary edge validation, geometry bindings, history and generation path. `ObjectCollection::insertEdgeWithId` accepts qualified endpoints, including linked objects. Failed insertions and transaction rollback restore the allocation state; out-of-order explicit insertion never moves the ordinary cursor backward or reopens an exhausted range. Unpopulated IDs inside a reservation remain owned, never become free allocation space in another shard. Manifest-only databases have exhausted local allocators.

Explicit `KeyRoute` declarations additionally support multiple ranges per destination, modulo, selected bits, and a versioned hash. For each key space, one authoritative rule kind/divisor/shift is allowed; overlapping rules are rejected. The publisher checks custom routes against all stored IDs once. Opening uses the sealed declaration without rescanning every key. Numeric object lookup selects the declared destination; UUID lookup uses indexed queries in relevant partitions. Edge identity routing and adjacency scattering are separate operations.

For positive ID `x`, modulo routes on `x % divisor`; bit routing uses `(x >> shift) % divisor` with a power-of-two divisor. HashV1 applies unsigned 64-bit overflow arithmetic: `x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9; x = (x ^ (x >> 27)) * 0x94d049bb133111eb; x ^= x >> 31;` then modulo. Never substitute implementation-dependent `std::hash`.

`publishShards` validates every shard before exposing the new destination. Ordinary file publication uses a same-directory stage and atomic no-overwrite publication with directory sync. A failed manifest is not a usable partially published collection. For packed builds, `PackedStoreBuilder::finalize` is the single outer publication boundary; all independent SQLite handles must close first.

`packObjectCollection(source, destination)` packages a sealed dependency closure into one packed file, preserving logical/database/table UUIDs and generations while replacing physical locator strings. A writable derived overlay can subsequently link the packed source.

## Limits and safety

There are no distributed transactions, mutable published shards, automatic parent-following, implicit ID renumbering or default spatial fallbacks. Sealing is an API contract and change-detection boundary, not cryptographic protection against malicious raw file edits. Keep a collection and its dependency closure alive for the lifetime of its transactions. Transactions are thread-confined, as with `ObjectStore`. One transaction holds read snapshots for the immutable dependencies it actually touches; it does not reopen databases for every lookup.
