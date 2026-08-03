# Plan: robust auto-download cache and worker control

## Implementation

1. Treat `.noremote.json` as an advisory cache:
   - accept only a JSON list of string chunk keys;
   - on empty, truncated, malformed, unreadable, or schema-invalid data, print
     one actionable warning and continue with an empty set;
   - never suppress a real remote check based on corrupt cache data.
2. Save every per-level cache, including an empty set, using a uniquely named
   temporary file in the same directory followed by `os.replace`. Clean up the
   temporary file on failure and warn without failing the completed download.
   Snapshot mutable key sets under the Stats lock first. This prevents
   interruption from exposing a partially written cache and clears stale keys.
3. Add positive `download_workers` validation to downloader CLI/programmatic
   entrypoints, the shared auto-download helper, and both inference APIs, then
   Lasagna compatibility wrapper. Validate it is positive and pass it to the
   existing downloader `workers=` argument.
4. Add `--download-workers` to Fiber 3D and Lasagna predict3d. Keep default 64;
   `--no-download` makes it unused. Do not conflate it with
   `--prefetch-workers`, `--slots-per-gpu`, or `--pyramid-workers`.

## Tests

- Empty, malformed, and schema-invalid cache files load as empty with warning.
- Valid cache files retain their keys.
- Cache saves are atomic, leave no temporary file, and write empty lists.
  Inject write/dump/replace failures and verify the old target remains valid,
  temp cleanup occurs, and the advisory failure only warns.
- Both CLIs forward `--download-workers`; auto-download passes it to the
  programmatic downloader for both input and optional pred-dt. Defaults remain
  64, `--no-download` calls no downloader, and invalid values are always
  rejected.
- Run focused downloader/CLI tests, syntax compilation, and diff checks.

## Spec update

Add advisory/atomic negative-cache semantics and the distinct 64-worker
auto-download control to `planning/specs.md`.

## Docs updates

Document `--download-workers` in Fiber code structure, Lasagna README, and 3D
inference options.

## Changelog/task log

Add a dated changelog entry and record validation/deviations in task log.
