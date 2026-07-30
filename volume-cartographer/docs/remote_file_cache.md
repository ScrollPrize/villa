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
- canonical query/fragment-free source location;
- exact payload size;
- `managed` or `unmanaged` accounting class.

A cache hit requires both files and matching identity, size, and accounting.
Downloads use unique temporary files and publish payload plus sidecar by
rename. A failed refresh restores the previous valid pair. Concurrent requests
for the same destination and source are coalesced within one process.

The readable source is stored in plaintext, but query strings and fragments
are removed before persistence because they may contain signed credentials.
The full caller-supplied locator is used only for the request. Error and
Lasagna hit/download diagnostics redact query strings.

## Disk Accounting

`Managed` payloads reserve space and hold read pins through
`PersistentZarrCacheBudget`. Startup scanning recognizes managed remote-file
payloads so accounting survives process restart. Budget eviction removes the
payload; a later lookup treats the remaining or missing sidecar as a miss.

`Unmanaged` payloads are small control files outside the evictable Zarr-byte
budget. Cached Lasagna manifests use this class. Callers must choose the class
deliberately and keep large data payloads managed.

## Lasagna Layout

`remoteFileCachePath()` maps a validated remote source to a readable path:

```text
<remote-cache-root>/remote_sources/https/example.org/run/file.json
<remote-cache-root>/remote_sources/s3/bucket/run/file.json
```

Scheme, authority, and object-path components are mirrored directly. Empty,
traversing, or platform-invalid components are rejected. Direct remote
Lasagna manifests use this path for the manifest and its sidecar; relative
Zarr groups share the readable parent cache directory. Existing local
manifests with an explicit adjacent `lasagna-remote.json` remain supported and
its `artifact_url` remains authoritative.

There is no lookup or migration for the earlier development-only cache
layout. Reopening a remote source populates the readable layout from a cold
cache.
