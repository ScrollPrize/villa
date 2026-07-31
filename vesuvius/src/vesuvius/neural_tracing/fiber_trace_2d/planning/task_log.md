# Preserve Version-3 Fiber Spans During Sync

## Implementation

- Kept `vc_sync.py` file conflict detection, shadow bases, conflict copies,
  confirmation, and manual local/remote/skip resolution unchanged.
- Added a version-3-only anchored chunk merge in `scripts/fiber_merge.py`.
  Every chunk carries its complete dense line slice plus every CP descriptor
  for spans starting inside the chunk.
- Added ordered 1e-8 CP-to-line lookup matching VC3D's load invariant.
- Added conservative ownership resolution: one-sided and identical changes
  pass; different same-run changes fail; local-only and remote-only runs need
  at least one unchanged base span between them.
- Reconstructed clean results only by exact dense-slice concatenation. Version
  3 never uses the CP-polyline placeholder and never drops descriptors.
- Added base-aware merging and strict validation for `optimization_mode`.
- Left version-1/version-2 geometry merge behavior unchanged.

## Plan Review

- Included extrapolated prefix/suffix tails in the anchored partition so the
  entire persisted line, not only CP-to-CP interiors, is merged.
- Kept descriptors on their starting CP; a chunk's terminal CP contributes
  position only because its descriptor belongs to the next chunk.
- Required exact selected-chunk joins after tolerant base alignment. Any
  tolerance-only mismatch becomes a manual conflict rather than being snapped.

## Validation

- `python -m py_compile volume-cartographer/scripts/fiber_merge.py volume-cartographer/scripts/vc_sync.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q volume-cartographer/scripts/tests/test_fiber_merge.py volume-cartographer/scripts/tests/test_vc_sync_helpers.py`
  - 142 passed.

## Deviations

- The plan received a separate primary-agent review rather than the subagent
  review requested by the nested workflow because the active agent policy
  forbids spawning subagents unless the user explicitly requests delegation.
  There was no functional deviation.
