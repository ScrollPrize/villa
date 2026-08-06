# Task log: contextual help and argument completion

## Findings

- The open-data catalog identifies OME-Zarr origins but does not enumerate the
  source pyramid's dataset paths or number of levels.
- Exact scale completion can use a locally downloaded root `.zattrs`
  `multiscales[].datasets[].path` list and numeric groups with `.zarray`.
- The current Bash/Zsh scripts duplicate positional cases and match only full
  command names, so a shared contextual resolver is needed for abbreviation
  parity and broader argument coverage.
- The current live catalog contains four volumes with
  `properties.shape = null`; `index_volumes` defaults only a missing key, then
  attempts to iterate explicit `None`, causing `volume ls` to fail globally.

## Deviations

- None at planning time. Unknown remote scales will intentionally produce no
  proposals until local metadata exists; guessing would violate "available"
  semantics and network access during completion is prohibited.

## Independent review

- Clarified longest-valid-prefix help fallback and protected arguments after
  `--` from rewriting.
- Added separated/equals option-value coverage, abbreviation and adapter word
  transport, and completion termination after `--`.
- Kept the existing value-only shell presentation; annotations remain an
  internal tab-separated boundary rather than a newly promised UI feature.
- Required nullable shape rendering as `shape=-` and caught misplaced legacy
  assertions in the new regression test; both were incorporated.

## Validation

- The real cached catalog indexes 71 volumes, including four null-shape entries;
  `las_manager volume ls` prints all 71 and renders those four as `shape=-`.
- Final focused manager/open-data/provenance/packaging/direct-provenance suite:
  `43 passed, 59 deselected`, with six
  pre-existing Atlas Pydantic v2 deprecation warnings.
- Contextual subset: `13 passed`, covering help fallback/forwarding, full and
  abbreviated completion, separated/equals option values, cache-only scales,
  nullable shapes, multi-venv dispatch, and Bash transport.
- Installed-command Bash transport completes abbreviated roots, cached volumes,
  and separated option values. Generated Bash passes `bash -n`; Python modules
  compile and `git diff --check` passes.
- Zsh is not installed on this host, so its adapter transport is asserted in
  generated-script tests but could not be executed in a real Zsh process.
