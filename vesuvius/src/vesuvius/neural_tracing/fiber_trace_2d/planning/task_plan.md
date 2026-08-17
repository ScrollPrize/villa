# Plan: direct replay visualization manifests

## Implementation

1. Refactor replay loading into strict schema parsers plus one normalized
   display result. Accept a `vc_fiber_replay_visualization` manifest directly;
   independently validate its complete sources/binding/identity/failure/tube/
   crop/artifact contract without relying on an aggregate root. Restore a
   dedicated strict version-1 `vc_fiber_replay` parser which normalizes the old
   trace to one greedy segment and its optional graph route to one fiberlet
   segment, for both failed and nonfailure artifacts.
2. Remove viewer `--index` parsing and bundle-index resolution. A version-2
   aggregate root passed to `--replay` must fail with a precise message that
   points users to its listed per-failure manifests.
3. Publish one stable top-level alias per `{tracer, tracer_failure_index}` named
   `fiber_replay_visualization.<tracer>.<index>.json`. Each is a complete local
   manifest whose artifact descriptors point into the current immutable
   generation. Atomically replace aliases on publication, remove obsolete
   aliases only after successful publication, and make reload reread the same
   alias so it observes later runs. Version-1 reload rereads its original path.
4. Keep aggregate publication and immutable content-addressed generations for
   reproducibility and discovery. Its entries reference the stable aliases.
   Print each alias as an absolute directly usable manifest path after
   publication so it can be copied into a viewer command without parsing JSON.

## Tests

1. Load a generated local visualization manifest directly with the aggregate
   absent, including strict artifact/hash/geometry/path/symlink validation.
2. Verify the CLI has no `--index`, rejects a version-2 aggregate with a useful
   direct-manifest error, and accepts replay mode without an index.
3. Cover stable alias publication, printed paths, obsolete-alias cleanup, and a
   reload observing a newly published generation.
4. Restore focused coverage for strict version-1 failed/nonfailure replay,
   optional graph routes, reload, and `--no-anchor-stages`.
5. Build `vc_fiberlets` and `test_fiber_replay` with `-j32`; run the focused C++
   and Python viewer suites.

## Spec Update

- Replace indexed-root viewer semantics, identity-based root reload, and the
  no-compatibility statement with direct stable per-failure manifests,
  aggregate-root discovery only, and the restored version-1 reader.

## Documentation Updates

- Update `volume-cartographer/docs/fiberlets.md` with direct manifest commands
  and the distinction between aggregate and visualization manifests.
- Record review, implementation, validation, and any deviations in planning
  documents and the changelog.
