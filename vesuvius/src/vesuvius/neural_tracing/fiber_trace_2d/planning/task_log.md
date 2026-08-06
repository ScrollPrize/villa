# Task log: manager no-prefetch download behavior

## Findings

- Both manager backend command builders currently append `--no-download`
  unconditionally, even when `--no-prefetch` skips the manager downloader.

## Deviations

- None.

## Plan review

- Added empty-cache `_download` source initialization, backend worker
  forwarding, full backend/mode matrix, CLI dispatch, and explicit-argument
  precedence coverage.

## Validation

- Initial argv-only tests passed, but independent review found that a fresh
  cache lacked the `_download` source metadata required by backend fetching.
- Added shared metadata-only source initialization and expanded the validation
  matrix; final results are recorded after the complete run below.
- Focused manager/open-data/packaging/bootstrap suite: 54 passed.
- Python compilation and `git diff --check` completed cleanly.
