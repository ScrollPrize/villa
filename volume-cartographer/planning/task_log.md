# Task log

## Findings

- Open Data prefill currently walks a private cache through
  `getChunkBlocking()`, so every downloaded chunk is decoded and enters decoded
  cache accounting before disk persistence.
- Settings redownload repeats the same pattern with a separate service, then
  writes the returned decoded bytes itself.
- Ordinary prefetch users already target a Volume's shared cache; the main
  architectural gap is a persistence-only consumer on the shared keyed source
  fetch.
- Generic compressed Zarr source payloads currently become decoded `.bin`
  persistent entries. A strict no-decode prefill therefore requires exact
  encoded source payloads to become a supported persistent representation.
- The previous task's default injection of the process decoded-byte budget into
  isolated services creates eviction coupling without cache/data sharing. This
  task removes that prefill/redownload special case.
- Independent plan review found that maintenance completion must remain separate
  from decoded entry state, that the existing fair background lane is not a
  strict lowest-priority lane, and that changing ordinary persistent writes to
  exact-source format would exceed the task. The revised plan uses a separate
  maintenance state, a third scheduler class, and an additional exact-source
  representation used only by maintenance.
- Exact no-decode refresh cannot preserve redownload's current compression and
  quantization behavior. Redownload will refresh exact source payloads only;
  existing-cache compression remains a separate explicit action.
- Implemented a third maintenance work class inside each existing scheduler;
  no additional scheduler, worker pool, or connection pool was introduced.
- Added exact `.source` persistence, normal-read decoding of that form,
  bidirectional source-transfer joining, and atomic replacement semantics.
- Migrated Open Data prefill and Settings redownload to the shared source cache,
  removed Volume-level private cache construction, and moved Lasagna corner
  samplers to stable path-keyed process-cache sources.
- Added focused tests for zero-decode persistence, source refresh outcomes,
  both transfer-join directions, view-demand removal, fetcher refresh, decoded
  RAM accounting, and maintenance priority.

## Deviations

- None.

## Validation

- Built the complete configured project, including VC3D and `vc_volume`.
- Focused cache/Volume/Open Data/Lasagna tests passed (6/6).
- Full CTest passed (150/150), including the configured live network tests.
- Final source scans found no remaining Volume-level private cache factory or
  in-process Lasagna corner-budget hybrid.
