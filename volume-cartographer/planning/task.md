# Public S3 Lasagna authentication fallback

Fix manual remote Lasagna attachment when ambient AWS credentials are stale or
malformed but the requested S3 manifest and Zarr objects are publicly readable.

The observed real-data failure uses the public PHerc0139 fiber-inference
manifest at:

```text
s3://vesuvius-challenge-open-data/PHerc0139/representations/predictions/fibers/20260102150214-fibers-20260801084232-L1/PHerc0139-20260102150214-las-sd1-92481a4c.lasagna.json
```

Both that locator and its virtual-hosted HTTPS equivalent are recognized as S3,
signed with ambient credentials, and rejected with `InvalidToken`. The remote
manifest cache does not currently retry anonymously. Lasagna's exact-byte
remote Zarr store likewise retains the rejected credentials instead of falling
back as the ordinary remote-volume reader already does.

Required behavior:

- Keep authenticated access for private S3 data.
- When a signed S3 request fails specifically because authentication was
  rejected, retry the same request anonymously.
- Once Lasagna's remote Zarr store proves anonymous access works, keep using
  anonymous requests for that store instead of repeating the failed signed
  attempt for every object.
- Apply the same behavior to `s3://`, region-qualified S3, and recognized S3
  HTTPS endpoints.
- Do not retry unrelated HTTP, transport, server, or not-found failures.
- Preserve remote source identity, cache paths, exact bytes, and project
  serialization.
