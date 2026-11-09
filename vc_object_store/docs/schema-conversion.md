# Offline schema conversion and history policy

Normal readers open schema 21 only. Use the separate converter for upstream schema 20 or supported upstream schema 16–19 stores:

```sh
vc_object_convert source.vcobj converted.vcobj
vc_object_convert source.vcobj converted.vcobj mapping.json report.json
vc_object_convert --in-place store.vcobj
```

With separate source/destination paths, the source is never modified and the destination must not already exist. Conversion uses a sibling staging database, verifies table counts, foreign keys, geometry descriptors and SQLite integrity, and copies artifacts referenced by retained history. Errors remove the incomplete destination and staging files. Keep the source until the converted output has been accepted by its consumer.

`--in-place` explicitly requests replacement. It requires exclusive access to a regular, non-symlink SQLite store, builds and validates a complete sibling conversion, and only then durably replaces the original database atomically. Referenced artifacts are verified and stay at their original location. A conversion/validation failure leaves the original logical store intact; SQLite may checkpoint its existing WAL while acquiring exclusive access. Stop all clients before conversion and reopen them afterwards.

The converter preserves store and lineage UUIDs, numbered object/relationship versions, historical revision references, commit receipts, timestamps, approvals, geometry bytes and artifact contents. Legacy mutation counters become numbered versions; they are not renamed as commit numbers. The report identifies incomplete upstream history and legacy journal entries whose individual mutation cannot be assigned to a commit: those journal entries retain attribution and approval provenance, with `commit_sequence=0` and no invented approval-version pin.

Upstream schema 17 receives the same deterministic numbered-history reconstruction and revision-to-version mapping as the upstream converter. Schema 16 retains its existing restrictions: surface/patch histories and unsettled synchronization baselines are rejected. Replicas with unpublished edits and stores with unfinished writes must be settled using their original tooling first. Unknown tables, columns, views, triggers and indexes are rejected rather than silently dropped; extensions must be explicitly ported. The unpublished experimental schema-18/v2 branch layout is not a supported input.

## Changing history retention

History retention is a whole-store policy, not an ordinary open option:

```sh
vc_object_convert --remove-history-and-disable store.vcobj
vc_object_convert --enable-history store.vcobj
```

These commands require exclusive access to an unsealed ordinary SQLite store. They build and validate a sibling copy and atomically replace the original only after success. Reload open store and geometry handles afterwards. Sealed stores require a new export instead.

Removing history materializes the current geometry and keeps current record versions, publication lineages and commit stamps; it drops archived states and history-only storage. Enabling history archives the current state under its existing version identity, without claiming that unrecorded earlier versions or commits can be replayed. The policy change advances the store commit sequence once and invalidates earlier change-feed cursors. Extension triggers that become invalid under the requested policy cause rejection before replacement.

The deferred delta-history redesign is not part of schema 21.
