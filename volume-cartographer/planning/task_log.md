# Public S3 Lasagna authentication fallback task log

## 2026-08-27 findings

- The public PHerc0139 manifest returns HTTP 200 to an unsigned request.
- `resolveRemoteUrl()` intentionally classifies both `s3://` locators and
  virtual-hosted `*.s3.amazonaws.com` HTTPS URLs as S3. Changing between those
  forms therefore does not bypass signing.
- Manual remote Lasagna attachment loads ambient AWS credentials before opening
  the manifest. The observed credentials contain a rejected session token.
- `RemoteFileCache` performs one credentialed request and reports
  `InvalidToken`; it has no anonymous retry.
- The ordinary remote Zarr opener already retries anonymously after a rejected
  credentialed open.
- Lasagna uses a separate exact-byte remote Zarr store. It also builds one
  credentialed client and currently has no anonymous fallback for descriptor or
  chunk reads.
- Open-data catalogue Lasagna preparation avoids this failure by explicitly
  disabling credential discovery, but manually attached manifests do not carry
  that public-data knowledge.
- A path-style `https://s3.amazonaws.com/<bucket>/<key>` locator works as an
  immediate workaround because the current resolver does not classify that
  hostname as S3, but it is not an acceptable permanent requirement.

## 2026-08-28 review follow-up

- Signed-first fallback scanned successful response bodies for AWS error words,
  so private content containing a marker could be discarded.
- S3 HEAD responses carry no error body. A stale session token therefore
  produced an unclassified empty HTTP 400 before a fresh remote Zarr could open.
- Anonymous-first access avoids both ambiguities: public data never uses ambient
  credentials, while anonymous 401/403 responses still fall back to private
  authenticated access.
- A not-found or unrelated failure does not select a sticky mode. Concurrent
  initial requests share one probe, and a later anonymous denial can upgrade a
  mixed-access store.

## Implementation result

- `S3AuthFallback` now performs one synchronized anonymous-first probe while
  retaining stable anonymous and authenticated HTTP clients.
- Anonymous 2xx responses make anonymous mode sticky. Anonymous 401/403
  responses retry with credentials, and authenticated 2xx/404 responses retain
  authenticated mode. Other failures leave the mode undecided.
- `RemoteFileCache` applies the policy to remote manifest GETs and preserves
  both anonymous and authenticated status in diagnostics when both fail.
- Lasagna's exact-byte remote Zarr store applies the policy to HEAD and GET
  requests, so descriptor validation and later chunk reads share the selected
  mode without changing cached bytes or source identity.
- Ordinary remote Zarr opening now attempts the complete anonymous open before
  retrying with credentials. Optional metadata 403s remain ignorable for
  least-privilege stores; if discovery then finds no required array metadata,
  the remembered denial triggers authenticated retry.
- Successful responses are never classified from their content.

## Validation

- Built `test_http_fetch_errors`, `test_remote_file_cache`,
  `test_lasagna_manifest`, `test_lasagna_project_volumes`, `test_remote_url`,
  `test_zarr_chunk_fetcher`, `test_volume_live_s3`, and `VC3D` with 32-way
  build parallelism.
- The six focused deterministic tests passed together. The policy test passed
  20 consecutive executions, including its coordinated eight-thread initial
  transition case.
- With `VC_TEST_REQUIRE_NETWORK=1`, `test_lasagna_manifest` passed all 17 cases
  with a malformed session token configured. The real-data case downloaded the
  reported PHerc0139 manifest anonymously and opened public `presence` metadata
  with shape `[9620, 3314, 3314]`.
- With required network access, `test_volume_live_s3` passed all 10 cases,
  including its invalid-session-credential anonymous-open case.
- The initial sandboxed live attempt could not resolve the public hostname;
  rerunning with network access succeeded. This was an execution-environment
  restriction, not a code failure.
