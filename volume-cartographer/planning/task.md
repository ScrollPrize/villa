# Public S3 Lasagna authentication fallback

Fix manual remote Lasagna attachment when ambient AWS credentials are stale or
malformed but the requested S3 manifest and Zarr objects are publicly readable.

The observed real-data failure uses the public PHerc0139 fiber-inference
manifest at:

```text
s3://vesuvius-challenge-open-data/PHerc0139/representations/predictions/fibers/20260102150214-fibers-20260801084232-L1/PHerc0139-20260102150214-las-sd1-92481a4c.lasagna.json
```

Both that locator and its virtual-hosted HTTPS equivalent are recognized as S3.
The previous implementation signed them whenever ambient credentials existed,
causing public access to fail with `InvalidToken` before anonymous access was
attempted.

Required behavior:

- Keep authenticated access for private S3 data.
- Attempt S3 data anonymously first and use credentials only when anonymous
  access is denied.
- Once Lasagna's remote Zarr store proves anonymous access works, keep using
  anonymous requests for that store.
- Apply the same behavior to `s3://`, region-qualified S3, and recognized S3
  HTTPS endpoints.
- Do not select or change a sticky mode after unrelated HTTP, transport,
  server, or not-found failures.
- Preserve remote source identity, cache paths, exact bytes, and project
  serialization.
