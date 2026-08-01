# Plan: Vesuvius Python CI Compatibility

## Workflow

1. Retain the committed `lasagna/**` workflow trigger and repository-root
   `PYTHONPATH` test environment.
2. Classify every failure in the Zarr 3.2.1 job log by shared root cause.

## Zarr Compatibility

1. Add one test helper that creates explicit v2 arrays/groups with
   slash-separated chunk keys through the appropriate Zarr 2 or Zarr 3 API.
2. Port the affected Fiber, Fiber 2D, and Fiber 3D fixtures to the helper.
3. Add one runtime helper for store-relative chunk-key encoding and raw store
   reads across both Zarr APIs.
4. Make both Fiber prefetch implementations use the shared key encoder, and
   make the older prefetch reader use the shared raw-byte operation.

## Tests And Validation

1. Run focused regressions under Zarr 2.18.7 and 3.2.1.
2. Run all three modules named in the CI failures under each Zarr version.
3. Attempt the complete CI test selection and report local dependency blockers
   rather than treating them as product failures.
4. Run `git diff --check`.

## Spec Update

- Require cache/prefetch chunk keys and raw store reads to work under both
  supported Zarr library versions without changing the persisted v2 layout.

## Docs Updates

- Document the shared Zarr compatibility helper in `docs/code_structure.md`.
- Record matrix reproduction and this host's asyncio limitation in
  `planning/local_development.md`.

## Changelog

- Record the CI import and Zarr matrix compatibility fixes.
