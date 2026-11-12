# Offline schema conversion and history policy

Normal readers open schema 22 only. Use the separate converter for schema 21 or supported upstream schema 16–20 stores:

```sh
vc_object_convert source.vcobj converted.vcobj
vc_object_convert source.vcobj converted.vcobj mapping.json report.json
vc_object_convert --in-place store.vcobj
```

With separate source/destination paths, the source is never modified and the destination must not already exist. Conversion uses sibling staging storage, verifies table contents, foreign keys, geometry descriptors and SQLite integrity, and copies current and retained-history artifacts. Errors remove the incomplete destination and staging files. Keep the source until the converted output has been accepted by its consumer.

`--in-place` explicitly requests replacement. It requires exclusive access to a regular, non-symlink store, builds and validates a complete sibling conversion, and only then durably replaces the original database/container atomically. Referenced artifacts are verified and stay at their original location. A conversion/validation failure leaves the original logical store intact; an ordinary SQLite input may checkpoint its existing WAL while acquiring exclusive access. Stop all clients before conversion and reopen them afterwards.

## Schema 21 to 22

This conversion preserves every populated core field, logical/physical identity, extension table and payload, history policy, commit/lineage pin and retained revision. It does not automatically apply the optional slim-import layout or quantize existing extension data. Current edges become UUID-keyed `WITHOUT ROWID` rows, edge tags reference UUIDs, and built-in kind/approval strings become their schema-22 integer codes. Empty metadata/dependencies normalize to SQL NULL. Numeric edge allocators, routes and the obsolete kind/ID index are removed; table-key metadata is updated to the new keys. Unknown kind names fail explicitly until a registered conversion is available.

Packed inputs remain packed. All physical members, including members outside the root's reachable graph, are retained. Referenced sealed dependencies are converted into the same packed output and references use `@packed` locators. For an ordinary SQLite root, converted dependencies are ordinary databases under `DESTINATION.members/`; the root remains an ordinary SQLite database and its references are relative. In-place conversion uses a fresh persistent `STORE.members-UUID/` directory if a converted plain dependency closure is needed. Those files are part of the result, not disposable fixtures. Source dependencies are never modified, and the complete converted closure is reopened and checked before publishing/replacing its root.

Extension data is copied without reinterpretation. Extension foreign keys that depend on removed numeric edge IDs cause explicit rejection, not dropped data. The converter also uses SQLite's authorizer to reject extension programs reading the changed edge-kind/approval/ID representations or writing core edge/tag rows; merely preparing such SQL would not detect a formerly textual comparison that now silently matches nothing. Custom indexes on the changed edge/tag tables require explicit porting. Other compatible extension views, triggers and indexes are retained and prepared against the converted schema before publication.

The converter preserves store and lineage UUIDs, numbered object/relationship versions, historical revision references, commit receipts, timestamps, approvals, geometry bytes and artifact contents. Legacy mutation counters become numbered versions; they are not renamed as commit numbers. The report identifies incomplete upstream history and legacy journal entries whose individual mutation cannot be assigned to a commit: those journal entries retain attribution and approval provenance, with `commit_sequence=0` and no invented approval-version pin.

Upstream schema 17 receives the same deterministic numbered-history reconstruction and revision-to-version mapping as the upstream converter. Schema 16 retains its existing restrictions: surface/patch histories and unsettled synchronization baselines are rejected. Replicas with unpublished edits and stores with unfinished writes must be settled using their original tooling first. In the upstream schema-16–20 path, unknown tables, columns, views, triggers and indexes are rejected rather than silently dropped; extensions must be explicitly ported. The unpublished experimental schema-18/v2 branch layout is not a supported input.

## Changing history retention

History retention is a whole-store policy, not an ordinary open option:

```sh
vc_object_convert --remove-history-and-disable store.vcobj
vc_object_convert --enable-history store.vcobj
```

These commands require exclusive access to an unsealed ordinary SQLite store. They build and validate a sibling copy and atomically replace the original only after success. Reload open store and geometry handles afterwards. Sealed stores require a new export instead.

Removing history materializes the current geometry and keeps current record versions, publication lineages and commit stamps; it drops archived states and history-only storage. Enabling history archives the current state under its existing version identity, without claiming that unrecorded earlier versions or commits can be replayed. The policy change advances the store commit sequence once and invalidates earlier change-feed cursors. Extension triggers that become invalid under the requested policy cause rejection before replacement.

The deferred delta-history redesign is not part of schema 22.
