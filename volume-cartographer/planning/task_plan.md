# Public S3 Lasagna authentication fallback plan

## Implementation

1. Extract the existing AWS credential-error recognition into one reusable
   core helper. Preserve explicit AWS token/signature error markers; treat 401
   and 403 responses as authentication rejection, but do not treat a bare 400
   or unrelated transport/server failure as authentication rejection.
2. Update the built-in `RemoteFileCache` transport to retry an S3 GET
   anonymously only when a credentialed request receives a recognized
   authentication failure. Preserve a useful failure if both attempts fail.
3. Update Lasagna's exact-byte remote Zarr store with the same policy for GET
   and HEAD requests. Coordinate concurrent authentication failures so one
   anonymous probe determines the transition while followers wait. Retain
   anonymous mode only after a 2xx or 404 response; 401/403, 5xx, and transport
   failures do not establish anonymous access. Keep both HTTP clients alive
   rather than replacing a client during concurrent requests.
4. Reuse the shared classification in the ordinary remote Zarr fallback so the
   three paths cannot drift on credential-error markers.

## Testing and validation

- Add deterministic unit coverage for recognized and unrelated responses and
  exact signed/anonymous request counts. Cover successful sticky fallback,
  concurrent probe coalescing, a valid signed private request, failed anonymous
  access to private data, unrelated 400/404/5xx responses, and non-S3 control.
- Cover URL classification for `s3://`, region-qualified S3, virtual-hosted S3
  HTTPS, and non-S3 HTTPS inputs.
- Run the relevant remote URL, HTTP fetch, remote file cache, Lasagna manifest,
  Lasagna project-volume, and remote Zarr tests from the existing build using
  all 32 build cores.
- Validate the reported public PHerc0139 manifest with deliberately invalid
  credentials as a real-data smoke test, without downloading Zarr chunks beyond
  descriptor validation.
- Run `git diff --check` on all changed files.

## Specification updates

Add a remote-authentication requirement to `planning/spec.md`: recognized S3
endpoints with rejected credentials must retry anonymously without changing
source identity, while private data must retain authenticated behavior.

## Documentation updates

Update `docs/remote_file_cache.md` and the remote Lasagna attachment section of
`docs/vc3d_project_files.md` to describe signed-first anonymous fallback and
sticky anonymous mode for public Lasagna objects.

## Changelog update

Add one dated line describing reliable public S3 Lasagna attachment in the
planning changelog.

## Independent review

Reviewed on 2026-08-27. The review required canonical error classification,
an explicit definition of authoritative anonymous responses, synchronized
fallback transition without client replacement, private-bucket diagnostics,
and mandatory exact request-count tests. The plan above incorporates those
requirements.
