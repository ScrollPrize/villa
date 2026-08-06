# Task log: grouped volume listing table

## Findings

- `index_volumes` already sorts by `(sample_id, long_id)`, so grouping does not
  need to alter discovery or selector behavior.
- The current tab-separated line repeats `id=`, `shape=`, `voxel=`, `format=`,
  `origins=`, and the full sample-prefixed selector on every record.
- `--json` already provides the appropriate machine-readable representation and
  must not be coupled to human table formatting.
- Root `.zattrs` may advertise remote levels that are not locally present, so
  the new prefetched column must require a numeric group with `.zarray` and an
  actual non-metadata chunk file.

## Deviations

- None at planning time.

## Independent review

- Required explicit header-only empty output, defensive grouping order, exact
  branch/spacing coverage, nullable/origin preservation, and separate JSON and
  filter tests.
- Required deliberate non-UTF behavior; the implementation will use ASCII
  branches when stdout cannot encode the Unicode tree glyphs.
- Confirmed no terminal-width truncation/color and no third-party dependency.
- All recommendations were incorporated before implementation.

## Validation

- Focused grouped-table tests pass: grouping, defensive sorting, exact branch
  semantics, alignment, null shapes, header-only empty output, ASCII fallback,
  JSON preservation, and chunk-backed prefetched scales.
- The real cached catalog renders 71 records plus header/separator in 0.21s;
  scroll IDs are grouped and the current empty manager volume store correctly
  shows `PREFETCHED=-`.
- Final focused manager/open-data/provenance/packaging/direct-provenance suite:
  `46 passed, 59 deselected`, with six pre-existing Atlas Pydantic v2
  deprecation warnings. Python compilation and `git diff --check` pass.
