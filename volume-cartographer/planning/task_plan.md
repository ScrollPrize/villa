# Public S3 Lasagna anonymous-first access plan

## Implementation

1. Attempt recognized S3 sources anonymously before using any available
   credentials. A successful anonymous response establishes sticky anonymous
   access for the owning remote store.
2. When anonymous access returns 401/403, retry with credentials and retain
   authenticated mode after a successful or not-found authenticated response.
   Permit a later anonymous denial to upgrade a mixed-access store.
3. Coordinate the initial Lasagna store probe so concurrent readers observe one
   selected mode, while retaining stable anonymous and authenticated clients.
4. Reverse the ordinary remote Zarr whole-open order so it also selects
   anonymous access before trying credentials.
5. Keep AWS error-body recognition diagnostic-only for successful responses;
   content containing strings such as `AccessDenied` must never change access
   mode.

## Testing and validation

- Add deterministic unit coverage for exact authenticated/anonymous request
  counts, successful response bodies containing AWS error words, empty-body
  HEAD success, concurrent probe coalescing, private fallback, mixed-access
  upgrade, unrelated failures, and non-S3 control.
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

Require recognized S3 endpoints to prefer anonymous access without changing
source identity, while private data retains authenticated behavior.

## Documentation updates

Update `docs/remote_file_cache.md` and the remote Lasagna attachment section of
`docs/vc3d_project_files.md` to describe anonymous-first selection and sticky
access mode.

## Changelog update

Add one dated line describing reliable public S3 Lasagna attachment in the
planning changelog.

## Independent review

The 2026-08-27 follow-up review found that signed-first handling scanned
successful payloads for error markers and could not classify an empty-body S3
HEAD 400. Anonymous-first selection removes both ambiguities while preserving
private and mixed-access behavior.
