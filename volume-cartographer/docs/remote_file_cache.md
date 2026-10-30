# Remote File Cache

`vc/core/util/RemoteFileCache.hpp` provides persistent exact-byte caching for
one arbitrary remote file or object. It is not a recursive directory cache.

## API

Call `vc::core::util::cacheRemoteFile(source, options)`. The caller supplies a
cache root and a relative destination below that root. The returned result
contains the local payload path, normalized endpoint, a cache-hit flag, and an
optional persistent-budget read pin.

Built-in fetching supports HTTP, HTTPS, S3, and region-qualified S3 locations.
`RemoteFileCacheOptions::fetcher` can provide another fetch-to-temporary-file
transport. Explicit `HttpAuth` is passed to the built-in transport; otherwise
the existing AWS credential loader is used where required. S3 objects are
requested anonymously first. Credentials are used only when S3 denies
anonymous access. Ambient credential discovery is also deferred until that
denial, so opening public data does not execute credential-process helpers and
stale ambient credentials cannot shadow public objects.
Unrelated HTTP and transport failures do not trigger authenticated fallback.

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

Lasagna's exact-byte remote Zarr store applies the same anonymous-first policy
to descriptor and chunk requests. A successful anonymous response selects
anonymous access for subsequent requests from that store. Concurrent readers
share one initial probe; private stores retain authenticated access after
anonymous access is denied. A later anonymous denial can upgrade a public store
to authenticated access for mixed-access datasets.

There is no lookup or migration for the earlier development-only cache
layout. Reopening a remote source populates the readable layout from a cold
cache.

## Remote volumes

Remote Zarr chunks do not go through this cache. vc_reader caches them under
`<cache root>/<volume id>` (its `vc3d-mirror` / `vc3d-delta3d` layers, as
`Volume::sourceOpenOptions()` configures them), with one byte budget across the
cache root. Below the configured free disk space (`perf/remote_cache_min_free_gib`)
cache writes stop and VC3D warns; nothing already cached is removed for it. An
explicit `perf/remote_cache_max_gib` evicts the oldest chunks past that limit.
