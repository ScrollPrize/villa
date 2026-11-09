# Reference-resolution cleanup

## Goal and constraints

Geometry, history, extensions, relationships and artifacts must resolve references through shared storage infrastructure, regardless of whether data is local, sharded or in a pinned readonly dependency. History policy controls retention, not identity. Preserve the existing slim history-off representation, binary UUIDs, immutable dependencies and checked writes. Do not add proxy objects, per-object routing directories, copied payloads or a replacement history format. Existing uncommitted reference-resolution work is the starting point.

## Checklist

- [x] Replace `current:<object UUID>:<version>` geometry tokens with geometry UUIDs in both history modes. Store the current UUID and descriptor on the object; retain old states only with history enabled. Metadata-only edits retain the UUID. Remove the old encoding and its branches, including conversion, initial import and consumers.
- [x] Replace special `external:` relationship endpoint tokens and callbacks with ordinary object/version references and shared current/exact-version resolution. Apply the same dependency checks to local and referenced endpoints; preserve tombstone precedence and pinned source identity.
- [x] Resolve artifact hashes through the pinned reference scope, including validation and reads. Metadata overrides must not require copying artifact files into the overlay's sidecar.
- [x] Make graph traversal use shared visible-record lookup for existence, kind and lifecycle checks instead of scattered local/external branches.
- [x] Make structural point-ID allocation account for the relevant retained/reference history rather than the latest local row alone. Preserve non-reuse across restore and inherited edits without introducing allocation directories.
- [x] Update documentation and regressions for local/referenced/sharded storage, both history policies, independent branches, inherited edits, rollback and missing references.
- [x] Run `bash local/scripts/run_vcobj_pr_tests`, inspect the final diff and record outcomes below. No dependency installation or full-volume benchmark is part of this cleanup.

## Implementation notes

The shared exact-reference resolver and common geometry/extension readers already exist in the working tree. Extend those rather than creating separate consumer-specific lookup paths. Keep current-state lookup (overlay precedence, deletion stops fallback) distinct from exact immutable-version lookup (resolve the requested identity, never substitute another version).

Physical-store history enumeration is not a collection-wide timeline API. This cleanup must not silently pretend otherwise. Editable extension scans currently decode selected owner snapshots; immutable result-table scans retain indexed SQL. No new throughput claims are made here.

The cleanup adds no history-off tables or per-object directories. One partial index on the existing current `objects.storage_ref` supports geometry-UUID lookup. History-off retains only its current descriptor; enabling/disabling history preserves its UUID. The now-identity-only history-policy remapping table was removed. Exact object references use retained archive UUIDs where available and the ordinary `(store, lineage, object, version)` identity otherwise; both resolve through shared code, with no special external-endpoint token or dependency bypass.

The regression pass also found and fixed commit-time resolver lifetime (the transaction must remain active during publication), inherited-object deletion ownership checks, and artifact-manifest validation with history off. Read routing and write ownership are distinct: resolving a referenced object does not mean its row is writable locally.

## Validation results

Final validation on 2026-11-09: `bash local/scripts/run_vcobj_pr_tests` completed successfully (Linux amd64, GCC, RelWithDebInfo). Native tests: 51 passed, one optional SSH integration skipped. Object-store Python tests: 199 passed. Spiral object-store integration tests: 117 passed, one opt-in real-data smoke test skipped. `git diff --check` passed. The run covers both history policies, overlays/nested references, sharded/packed storage, exact-version reads, metadata-only overrides, sparse edits, rollback, independent geometry branches, artifact references and structural point-ID non-reuse after restores.

Final local log: `local/vcobj-pr-tests/logs/20261109T220514Z-700270.log`. Existing Spiral/Python deprecation and non-contiguous-tensor warnings, plus the pre-existing `PatchSamplingAtlas` nanobind shutdown-leak warning, remain. No macOS/arm64 run, full-volume import or new throughput measurement was performed. No commit or push was requested for this cleanup.
