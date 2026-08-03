# Task log: robust auto-download cache and worker control

## Findings

- `_load_noremote` directly called `json.load`; an empty file therefore raised
  `JSONDecodeError` before downloading began.
- `_save_noremote` wrote directly to the final cache path, allowing interruption
  to expose a truncated file. It also skipped empty sets, retaining stale keys.
- The underlying downloader already supports `--workers` and programmatic
  `workers=` with a default of 64, but automatic inference downloads did not
  expose or forward it.

## Validation

- Focused `unittest` coverage passed for shared input/pred-dt worker forwarding
  and predict3d CLI forwarding.
- A direct regression smoke test verified recovery from an empty cache,
  preservation of the old cache and temporary-file cleanup on replace failure,
  atomic empty-cache writes, locked snapshot-by-value semantics, and rejection
  of zero downloader workers.
- Python compilation and `git diff --check` passed for the changed Python files.
- The active virtual environment does not contain `pytest`, so the new pytest
  regression module could not be run through pytest without an unrequested
  dependency installation.

## Plan review

- Cache save errors will warn and remain non-fatal, matching advisory read
  semantics; temporary files are still cleaned and old targets retained.
- Negative-key mappings will be snapshotted under the Stats lock before save.
- Tests will inject load and every atomic-save failure class and validate worker
  counts at downloader, wrapper, programmatic API, and CLI boundaries.
- Unique temporary names prevent malformed final files but do not merge
  concurrent independent downloader processes; last-writer-wins is acceptable
  for this advisory, revalidated cache and is recorded as a limitation.
