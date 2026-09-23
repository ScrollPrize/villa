# Plan: manager-owned snapshot identity

1. Add a persistent manager model directory with a backwards-compatible
   default, include it in snapshot discovery, and expose it in `config show`.
2. Implement atomic snapshot import which preserves checkpoint bytes, verifies
   SHA-256, records the required external model identifier in an adjacent
   manager-owned identity sidecar, and rejects identity conflicts.
3. Add `snapshot import` to the CLI and shell completion.
4. Resolve manager identities on every index pass independently of the
   checkpoint metadata cache, rejecting conflicts with checkpoint/Atlas data.
5. Refuse inference before creating a run unless the selected checkpoint has a
   non-empty model identifier.
6. Add regression tests and run the complete manager test suite.

## Spec update

Document that external model identity is mandatory for inference and that a
manager sidecar binds it to immutable checkpoint bytes by SHA-256.

## Docs updates

Document configuration, import, storage layout, selectors, and error behavior
in `lasagna/docs/manager.md` and the manager README quick start.

## Changelog update

Record the new manager-owned snapshot import and identity requirement.
