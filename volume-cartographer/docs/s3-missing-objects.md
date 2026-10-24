# S3 missing-object handling

A failed S3 bucket lookup can return HTTP 404 even when the requested object exists. Treating that response as an absent Zarr chunk substitutes fill values and can persist an empty cache marker. The marker can then hide the recovered object on later reads.

## Response policy

For S3 GET and ranged GET requests, absence requires HTTP 404 with a complete XML `Error` document containing exactly one direct `Code` equal to `NoSuchKey`. Empty, HTML, truncated, conflicting, nested-code and other S3 errors remain failures. DTDs are rejected. Unconfirmed responses use the configured retry count and backoff, opening a fresh connection for each retry.

S3 identity comes from an `s3://` or regional S3 URL, a parsed AWS S3 hostname, the effective redirect URL, or the final response's `Server: AmazonS3` or `x-amz-request-id` header. Identity is retained across retries of that request if a proxy drops those headers. Matching is against the URL authority, not path, query or userinfo text.

Ordinary HTTP origins keep their existing bare-404 missing-object behavior. A custom S3-compatible endpoint that provides none of the recognizable URL or header signals is still treated as generic HTTP. This change does not prove that an ambiguous error's object exists; it prevents unproven absence from becoming fill values.

A bodyless S3 HEAD 404 requires a fresh ranged GET of byte 0 for confirmation. A confirmed missing response remains absent. A successful probe causes HEAD to be retried for the full object size; its one-byte result is never used as that size. If the HEAD retry budget is exhausted, the response remains an error. Each confirmation GET uses the configured retry bound as well, so HEAD confirmation can require extra HTTP requests.

`HttpResponse::not_found()` excludes unconfirmed S3 404s, which carry `unconfirmed_404` and an `error_message`. Plain HTTP and SFTP 404s keep their meaning. Both Zarr store paths therefore distinguish missing chunks from failed reads. The generic `utils::HttpStore` optional APIs also propagate 403, 5xx and transport failures instead of silently returning an absent value. The text and binary HTTP wrappers throw for unconfirmed S3 404s. Confirmed missing chunks can still produce fill values and persistent empty markers.

Existing empty markers are not automatically repaired. This change protects future fetches; it does not identify which old markers were created by a bad response.

## Regression tests

`test_http_s3_missing` covers response parsing, poisoned connections, retry exhaustion, HEAD confirmation, full-size recovery, range preservation, generic HTTP compatibility, origin detection, Zarr fill values and store errors. `test_s3_chunk_cache` drives the production HTTP store, Zarr chunk fetcher and persistent chunk cache. It checks that a failed lookup writes no empty marker and that reopening the cache after origin recovery returns the original bytes. A genuine missing-key control verifies persistent sparse caching still works.

The fixture serves a synthetic 4 by 4 by 4 byte array on loopback TCP, including intentionally incorrect responses. The hostname tests use a loopback HTTP proxy that does not forward requests. No cloud credentials, bucket, scan, model or external network is needed.

On a machine with the normal VC dependencies installed, configure a test build from the villa root and run the two targets:

```sh
AGENTS_AGENT_MODE=1 cmake -S volume-cartographer -B build-http-tests -G Ninja \
  -DVC_TESTING=ON -DVC_USE_PCH=OFF
cmake --build build-http-tests --target test_http_s3_missing test_s3_chunk_cache --parallel 1
ctest --test-dir build-http-tests -R '^(test_http_s3_missing|test_s3_chunk_cache)$' \
  --output-on-failure --parallel 1
```

Use the platform compiler and dependency flags required by the normal build instructions. The XML classifier links `utils` to Qt Core, already used by VC. The loopback test fixture additionally links Qt Network. There are no new external parser or test-framework downloads.
