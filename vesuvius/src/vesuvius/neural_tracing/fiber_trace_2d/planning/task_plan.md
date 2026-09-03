# Plan

1. Recover the benchmark's authoritative crop, lookahead expansion, source
   models, source OME groups, generated artifact metadata, and current CLI
   defaults from stored provenance and repository documentation.
2. Add a standalone Volume Cartographer guide with prerequisites, Release
   build, managed full-volume Fiber and Lasagna prediction, Fiberlet generation,
   1024 crop tracing, oracle-pruning evaluation, expected outputs, and metric
   interpretation.
3. Distinguish the requested 1024 output crop from the 384-base-voxel expanded
   search range, state coordinate order and half-open semantics, and explain
   why no additional maximum-Fiberlet-length padding is required.
4. Link the new guide from the general crop-tracing documentation.
5. Validate paths, option names, model identities, arithmetic, Markdown links,
   and shell syntax against the implementation and stored artifact metadata.

## Spec Update

No behavior or format changes. Add no normative specification requirements.

## Docs Update

Create `volume-cartographer/docs/fiber_pruning_benchmark.md` and link it from
`volume-cartographer/docs/fiber_chunk_tracing.md`.

## Changelog

Record the new reproducible benchmark guide in the planning changelog.
