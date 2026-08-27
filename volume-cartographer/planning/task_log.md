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

## Independent review

- Use one explicit credential-error classifier. A bare HTTP 400 must not trigger
  fallback; explicit AWS error markers and HTTP 401/403 do.
- A 2xx or 404 anonymous response establishes anonymous accessibility. Auth,
  server, and transport failures do not make anonymous mode sticky.
- Coordinate concurrent fallback probes and retain separate stable signed and
  anonymous clients; do not replace a client while readers may use it.
- Deterministic tests must assert request counts and private-bucket behavior in
  addition to successful public fallback.

## Implementation result

- Added `S3AuthFallback`, a shared signed-to-anonymous request policy with one
  synchronized transition probe and stable signed/anonymous HTTP clients.
- Authentication failures are recognized from HTTP 401/403 or explicit AWS
  credential error codes. Bare HTTP 400, 404, 5xx, and transport failures do
  not trigger fallback.
- A 2xx or 404 anonymous response makes anonymous mode sticky. Anonymous auth
  rejection keeps private stores on authenticated access; transient anonymous
  failures may be retried by a future request.
- `RemoteFileCache` now applies the policy to remote manifest GETs and preserves
  both authenticated and anonymous status in diagnostics when both fail.
- Lasagna's exact-byte remote Zarr store applies the policy to HEAD and GET
  requests, so descriptor validation and later chunk reads share the selected
  mode without changing cached bytes or source identity.
- Ordinary remote Zarr fallback and generic HTTP error reporting now share the
  explicit AWS credential-error marker set.

## Validation

- Built `test_http_fetch_errors`, `test_remote_file_cache`,
  `test_lasagna_manifest`, `test_lasagna_project_volumes`, `test_remote_url`,
  and `VC3D` with `cmake --build volume-cartographer/build --parallel 32`.
- The five focused deterministic tests passed together. The fallback test also
  passed 20 consecutive executions, including its coordinated eight-thread
  transition case.
- With `VC_TEST_REQUIRE_NETWORK=1`, `test_lasagna_manifest` passed all 17 cases
  using syntactically valid but nonexistent AWS credentials. The real-data case
  downloaded the exact reported PHerc0139 HTTPS manifest anonymously after the
  signed request was rejected, then opened the public `presence` Zarr metadata
  with shape `[9620, 3314, 3314]`.
- The initial sandboxed live attempt could not resolve the public hostname;
  rerunning with network access succeeded. This was an execution-environment
  restriction, not a code failure.
