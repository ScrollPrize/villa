# Plan: Remote Lasagna Manifest Fetch Diagnostics

## Scope

- Improve diagnostics for failed remote Lasagna manifest fetches in
  `volume-cartographer/core/src/lasagna/Dataset.cpp`.
- Preserve the current fetch behavior, authentication behavior, retries, and
  remote-cache semantics.

## Implementation

1. Diagnostics helper
   - Build the remote `HttpClient::Config` explicitly in `Dataset.cpp` so fetch
     failure formatting can report S3 region and whether AWS credentials were
     loaded.
   - Add a bounded body-excerpt formatter that collapses control whitespace and
     avoids dumping large responses.
   - Redact query strings in displayed resolved URLs to avoid leaking presigned
     credentials.

2. Fetch failure message
   - Replace the current `Failed to fetch remote Lasagna manifest: <location>
     HTTP <status>` message with a richer message containing:
     original location, resolved URL, HTTP status or no-response marker,
     content type, content length, received bytes, body excerpt, and S3 auth
     status.
   - Keep empty successful response handling unchanged.

3. Tests
   - Add focused unit coverage for local formatting paths if cleanly reachable
     without network; otherwise rely on build/tests for the touched target and
     record the no-network limitation.

## Spec Update

- Document that remote Lasagna manifest fetch failures include resolved URL,
  HTTP metadata, S3 region/auth status, and response body excerpt.

## Docs Update

- Update `docs/code_structure.md` in the native metric/remote Lasagna section.

## Changelog

- Add a 2026-07-29 entry for richer remote Lasagna manifest diagnostics.

## Validation

- `cmake --build volume-cartographer/build --target test_lasagna_manifest`
- `cmake --build volume-cartographer/build --target vc_fiber_trace_metric`
- `volume-cartographer/build/bin/test_lasagna_manifest`
- `volume-cartographer/build/bin/vc_fiber_trace_metric --help`
- `git diff --check`

## Deferred Explicitly

- No live S3 fetch validation in this sandboxed run.
