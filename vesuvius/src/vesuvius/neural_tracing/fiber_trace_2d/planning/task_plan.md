# Plan: Remote Lasagna Manifests In Native Fiber Metric

## Scope

- Extend the reusable Lasagna dataset-open layer in VC3D, then consume it from
  `vc_fiber_trace_metric`.
- Do not add project/volpkg JSON persistence yet.
- Do not replace the existing local `lasagna-remote.json` marker behavior.
- Do not cache remote manifest JSON durably; only remote Zarr objects persist.

## Implementation

1. Add remote-manifest open options
   - Extend `LasagnaDatasetOpenOptions` with an optional `remoteCacheRoot`.
   - Add a public low-level opener, e.g. `LasagnaDataset::openLocation(...)`,
     accepting either a filesystem path or remote URL string.
   - Keep `LasagnaDataset::open(path, options)` as the local-only compatibility
     wrapper.
   - Detect remote locations with the existing VC3D remote URL helpers:
     `s3://`, `s3+REGION://`, `http://`, and `https://`.

2. Fetch and parse remote manifests
   - Resolve `s3://`/`s3+REGION://` through `vc::resolveRemoteUrl`.
   - Fetch the remote manifest body with existing HTTP/S3-capable code.
   - Parse via `LasagnaDatasetManifest::parseText(...)`; do not require a
     durable local manifest file.
   - Set a synthetic identity/path only for diagnostics and normal-source
     self-reference; do not write the manifest into the cache as durable state.
   - Fail clearly if the fetch returns empty/missing/invalid JSON.

3. Resolve group Zarr paths relative to the remote manifest
   - For a remote manifest URL ending in `.../fiber.lasagna.json`, set the
     remote artifact base URL to its parent URL.
   - Keep `LasagnaChannelGroup::relativeZarrKey` as the authoritative key.
   - Treat group Zarr paths as location strings:
     - relative paths resolve against the containing manifest location
       (`parent-url/path` for remote manifests, parent directory for local
       manifests);
     - absolute local paths starting at `/` are opened directly as local Zarr
       groups;
     - absolute remote paths with `s3://`, `s3+REGION://`, `http://`, or
       `https://` are opened as their own remote read-through volumes and use
       the supplied remote cache root.
   - Still reject malformed relative paths that escape their manifest parent
     where that would create an ambiguous location; absolute paths are explicit
     and are not rewritten.
   - Existing local manifests still use `zarrPath` and existing marker-backed
     remote manifests still use `lasagna-remote.json`.

4. Cache remote Zarr objects safely
   - For remote-manifest opens, require `remoteCacheRoot`.
   - Derive a deterministic collision-resistant cache subdirectory from the
     normalized remote manifest URL, e.g. under
     `<remoteCacheRoot>/remote_lasagna/<hash>/`.
   - Reuse `PersistentHttpStore` so fetched Zarr metadata/chunks are persisted
     object-for-object with atomic temp-file publish and the existing cache
     budget.
   - Normalize the remote base URL once, and make `PersistentHttpStore` use
     `vc::resolveRemoteUrl`/AWS auth support for `s3://` as well as already
     resolved HTTPS.

5. Wire the fiber metric CLI
   - Add `--remote-cache-dir PATH` to `vc_fiber_trace_metric`.
   - Open the fiber inference manifest with the new location-aware opener.
   - Open `--normal-manifest` through the same path.
   - If either manifest argument is remote and `--remote-cache-dir` is missing,
     fail before doing work with an explicit message.
   - Preserve current local invocation syntax unchanged.

6. Tests
   - Add unit tests for remote location parsing without network:
     `s3://bucket/path/fiber.lasagna.json` resolves group `pred.zarr` against
     the parent artifact URL and chooses a deterministic cache root.
   - Add manifest validation tests for:
     local absolute group paths are preserved;
     absolute remote group paths are preserved as independent remote locations;
     escaping malformed relative paths fail.
   - Add CLI argument tests or a lightweight runner smoke test for:
     remote manifest without `--remote-cache-dir` fails;
     local manifest remains accepted without `--remote-cache-dir`.
   - Keep existing `lasagna-remote.json` tests passing.

## Spec Update

- Document that Lasagna dataset opening supports:
  local manifests, local marker-backed remote manifests, and direct remote
  manifest URLs with explicit remote cache roots.
- Specify group path semantics: relative groups resolve against the manifest
  parent location, absolute local groups remain local, and absolute remote
  groups use read-through cached streaming.
- Specify that direct remote manifest JSON is transient and may be redownloaded
  on each run; only Zarr objects are persisted.

## Docs Update

- Update `docs/code_structure.md` with the new
  `vc_fiber_trace_metric --remote-cache-dir` usage and examples for `s3://`
  and HTTPS manifests.
- Update local development notes only if the command/build workflow changes.

## Changelog

- Add a short 2026-07-29 entry after implementation: native Fiber metric can
  stream precomputed fiber inference manifests directly from remote HTTP/S3
  locations with an explicit cache directory.

## Validation

- `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target test_lasagna_manifest`
- `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target vc_fiber_trace_metric`
- `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target test_fiber_trace3d`
- `volume-cartographer/build/ci-tests-clang-systemdeps/bin/test_lasagna_manifest`
- `volume-cartographer/build/ci-tests-clang-systemdeps/bin/test_fiber_trace3d`
- `volume-cartographer/build/ci-tests-clang-systemdeps/bin/vc_fiber_trace_metric --help`
- `git diff --check`

## Deferred Explicitly

- VC3D project JSON / volpkg persistence for remote fiber inference datasets.
- Open-data artifact-prefix discovery when the user supplies only a directory
  prefix rather than an explicit remote manifest URL.
- Durable manifest caching; this task intentionally redownloads the manifest
  and only persists Zarr objects.
