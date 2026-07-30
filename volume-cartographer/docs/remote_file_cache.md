# Remote File Cache

`vc/core/util/RemoteFileCache.hpp` provides persistent exact-byte caching for
one arbitrary remote file or object. It is not a recursive directory cache and
does not replace the existing remote Zarr object cache.

## API

Call `vc::core::util::cacheRemoteFile(source, options)`. The caller supplies a
cache root and a relative destination below that root. The returned result
contains the local payload path, normalized endpoint, a cache-hit flag, and an
optional persistent-budget read pin.

Built-in fetching supports HTTP, HTTPS, S3, and region-qualified S3 locations.
`RemoteFileCacheOptions::fetcher` can provide another fetch-to-temporary-file
transport. Explicit `HttpAuth` is passed to the built-in transport; otherwise
the existing AWS credential loader is used where required.

`CacheFirst` reuses a valid entry. `Refresh` fetches and atomically replaces
it. `invalidateRemoteFileCacheEntry()` removes one known destination. There is
no automatic TTL, ETag check, or prefix traversal.

## Entry Contract

The payload is stored at the caller-selected destination. Its adjacent
`<payload>.vc-remote-file.json` sidecar records:

- sidecar format version;
- collision-free hexadecimal identity of the normalized source endpoint;
- exact payload size;
- `managed` or `unmanaged` accounting class.

A cache hit requires both files and matching identity, size, and accounting.
Downloads use unique temporary files and publish payload plus sidecar by
rename. A failed refresh restores the previous valid pair. Concurrent requests
for the same destination and identity are coalesced within one process.

The sidecar never stores the source URL as plaintext. Error and Lasagna
hit/download diagnostics redact query strings, which may contain signed
credentials.

## Disk Accounting

`Managed` payloads reserve space and hold read pins through
`PersistentZarrCacheBudget`. Startup scanning recognizes managed remote-file
payloads so accounting survives process restart. Budget eviction removes the
payload; a later lookup treats the remaining or missing sidecar as a miss.

`Unmanaged` payloads are small control files outside the evictable Zarr-byte
budget. Cached Lasagna manifests use this class. Callers must choose the class
deliberately and keep large data payloads managed.

## Lasagna Layout

Direct remote Lasagna manifests retain the existing collision-free layout:

```text
<remote-cache-root>/remote_lasagna/url_hex/<segmented-url-hex>/
  manifest.lasagna.json
  manifest.lasagna.json.vc-remote-file.json
  lasagna-remote.json
```

The marker stores the remote artifact parent used to resolve relative Zarr
group paths. Existing local manifests with an explicit adjacent
`lasagna-remote.json` remain supported and authoritative.
