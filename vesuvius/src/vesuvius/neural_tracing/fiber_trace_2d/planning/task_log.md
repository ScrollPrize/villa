# Path-Based Lasagna Volumes And Base-Space Fiber Tracing Task Log

## Planning findings

- The reversible URL/path hex encoder originated in commit `458f19215` for a
  collision-free, filesystem-safe remote Lasagna cache directory.
- Commit `26e2d12b2` extracted that cache helper and incorrectly reused it for
  project provenance and derived-volume locations.
- Derived identities are currently double encoded because the code builds
  `hex(hex(manifest_location) + "|group|channel")`.
- `lasagna_datasets[].location` already retains the authoritative actual
  manifest source, so encoded project identities are redundant.
- The user confirmed this identity/cache representation was never shipped and
  requires complete removal. No decoder, project migration, old-cache lookup,
  or backward-compatible reader will be implemented.
- The replacement cache layout will mirror readable remote scheme, authority,
  and object-path components. Persistent sidecars will validate an actual
  canonical source path rather than `source_identity_hex`.
- Lasagna group resolution already distinguishes local paths, direct remote
  origins, explicit remote sidecars, and absolute remote groups, but it does
  not expose one authoritative human/project source-location field.
- The GUI currently rejects a valid base-line/trace-grid difference. It passes
  base-space line points directly into a prediction field configured for trace
  coordinates if that rejection is simply removed.
- Correct GUI tracing needs both point conversion and a separate normal sampler
  configured for trace coordinates. The ordinary base-space sampler is still
  needed to rebuild the stored line after converting results back.
- Endpoint physical conversion must use trace voxel size
  `base_voxel_um * trace_to_base`.

## Deviations

- Independent agent review required by the local planning process was not run
  because the active runtime policy prohibits delegation unless the user
  explicitly requests it. A direct code/spec consistency review was completed.
- Initial focused builds used the plan's conservative `-j2`. The user requested
  full machine parallelism, so subsequent builds use `-j32`.

## Validation

Implemented and validated:

- Remote arbitrary-file sidecars now store a readable, query-free `source` and
  Lasagna direct-manifest caches mirror
  `remote_sources/<scheme>/<authority>/<path>`.
- Lasagna groups retain an authoritative source locator independently of their
  runtime HTTP endpoint and local cache path.
- Project derived-volume locations are actual local/remote Zarr paths and use
  one `vc-lasagna-derived:<manifest location>` ownership tag per manifest.
- Shared-source reconciliation and detach preserve other manifest owners;
  independently attached primary volumes receive no automatic ownership tag
  and survive all manifest detaches.
- VC3D keeps line/control coordinates in base space, runs prediction and a
  dedicated normal sampler at the derived trace scale, converts accepted
  results back to base space, restores endpoints exactly, and uses trace-scale
  physical voxel size.

Build commands:

```bash
cmake --build volume-cartographer/build/ci-tests-clang-systemdeps \
  --target test_remote_file_cache test_lasagna_manifest \
  test_lasagna_project_volumes test_volume_pkg test_fiber_trace3d -j2
cmake --build volume-cartographer/build/ci-tests-clang-systemdeps \
  --target test_lasagna_manifest test_open_data_manifest VC3D \
  vc_fiber_trace_metric -j2
cmake --build volume-cartographer/build/ci-tests-clang-systemdeps \
  --target test_fiber_trace3d VC3D vc_fiber_trace_metric -j32
```

The final `-j32` build passed. Ninja reported a recoverable pre-existing
`premature end of file` warning and rebuilt more targets than expected. VC3D
also emitted existing Qt deprecation warnings in unrelated sources.

Test results:

- `test_remote_file_cache`: 8 passed.
- `test_lasagna_manifest`: 14 passed.
- `test_lasagna_project_volumes`: 4 passed.
- `test_volume_pkg`: 39 passed.
- `test_fiber_trace3d`: 26 passed.
- `test_open_data_manifest`: 32 passed.
- `vc_fiber_trace_metric --help`: passed.
- `git diff --check`: passed.
- Source/doc audit found no stale development-only encoded location or cache
  symbols in `volume-cartographer` implementation/docs or fiber docs/specs.

Two initial test failures were corrected during the loop: direct manifest
validation now expects malformed remote sources to fail before fetching, and
the authoritative group-source helper supports directly parsed local test
manifests through their resolved `zarrPath`.

## Limitations

- The interactive VC3D smoke test with the user's local project was not run in
  this non-GUI session. The VC3D target and all relevant noninteractive tests
  pass, but the attach/save/trace/reopen workflow still needs an interactive
  check against that dataset.
