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

## User review

- The initial table attached branch glyphs to the volume text on the same row
  as the scroll, which did not express the requested parent/child hierarchy.
- After user review, the final layout keeps the first volume on the scroll row
  and uses child branches in the `SCROLL` column only for additional volumes.
  The redundant `ID` column is removed because the long
  volume name already begins with that ID; JSON remains unchanged.
- Three-dimensional shapes will retain catalog depth/height/width order while
  using requested space-padded widths 6/5/5 for vertical component alignment.

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
- After user review, the same final suite remains `46 passed, 59 deselected`.
  The live 71-volume table confirms first-volume/scroll rows, branches only for
  additional volumes, no duplicate ID column, 6/5/5 shape padding, and no
  trailing whitespace on scroll rows.
