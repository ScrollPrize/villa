# Task log: fix large staged Fiberlet cache preparation

## Reproduction

- The exact 1024-region staged command fails during cache preparation before
  staged reduction begins.
- Observed schedule: 1,728 anchor chunks and 1,000 Fiberlet chunks.
- Reproduced failure after 15.2 seconds on a partially warm cache, at 918
  resolved anchors and 179 resolved Fiberlet prefixes.
- Current `prefetchScheduled(..., wait=true)` replaces the original terminal
  status and message with `scheduled fiberlet chunk did not resolve to data`.
- Precise diagnostics identify owner `0/427/139/187`, endpoint
  `1708/556/748:0`, and a normal-direction comparison. Stored and resampled
  normals print identically to six decimals and have absolute dot product 1.
- A fresh cache covering that owner completes successfully. The existing
  persisted anchor chunk and fresh anchor chunk share the same metadata
  namespace but have different serialized hashes/sizes.
- The mismatch is not limited to harmless compressed representation: the
  current producer generates 2,526 Fiberlets for the focused region while the
  stale anchor population generates 2,517, and both prefix and route payloads
  differ. Accepting a float tolerance would therefore combine incompatible
  producer outputs and hide an invalid cache identity.
- The stale chunk predates the staged-analysis implementation. The cache
  processing contract remained at version 2, allowing an older built producer
  and current generation to use the same namespace despite different output.

## Independent review

- The reviewer correctly required precise key/status/source-error propagation
  and warned against retrying or pinning an entire prefetch schedule.
- The reviewer hypothesized a decoded-cache handoff race. The actionable
  diagnostic disproved that for this failure: `ChunkCache` returned a stable
  terminal generator error whose source was the endpoint scoring check.
- Generic blocking-handoff changes and their broader tests are therefore out
  of scope for this focused fix; no retry or memory-pinning behavior will be
  added.

## Plan correction

- The initial tolerance proposal is rejected after byte and population
  comparison. Strict endpoint evidence validation remains in place.
- The corrected fix revises the unpublished generation contract so the current
  producer cannot reuse stale anchor or Fiberlet payloads, while preserving
  precise terminal error propagation.

## Implementation

- Added shared generation contract version 3 to anchor, Fiberlet, and combined
  algorithm identity and switched `vc_fiberlets` metadata generation to it.
- Kept endpoint scoring validation strict. Stale/current payload comparison
  showed that tolerance alone was invalid, so contract v3 separates those
  producers. A subsequent Clang-Debug consumer of GCC-Release v3 anchors proved
  that redundant Lasagna normal reconstruction varies by one to two float32
  ULPs across supported compilers. A finite componentwise eight-epsilon bound
  now covers only that arithmetic roundoff; validity, orientation, scoring,
  paths, costs, and serialized values are unchanged.
- The Clang-from-GCC-anchor focused generation passed with that validation
  bound but produced 2,528 incident Fiberlets versus GCC's 2,526 (both reduced
  to 2,275). Therefore compiler identity/version and build configuration are
  now part of v3 producer metadata. Mixed-toolchain cache roots are rejected
  instead of combining numerically different candidate populations.
- Scheduled resolution failures now retain the exact owner key, terminal cache
  status, and nested generator message. Direction mismatch diagnostics print
  all float32 digits rather than rounding both values to the same six-decimal
  text.
- Raised the chunk-route exact-search default guard from 1,000,000 to
  5,000,000 states per entry. The original guard was the next independent
  failure exposed after cold v3 preparation completed; five million was the
  smallest tested bound for the requested 1024 workload.

## Validation

- GCC Release focused storage tests: 34/34 passed.
- Clang Debug focused storage tests: 34/34 passed.
- A Clang Debug process explicitly opening the final GCC Release v3 anchor
  cache rejects it at metadata validation before generation, confirming that
  compiler/build-specific producers cannot mix persisted records.
- The requested 1024-region command cold-generated 1,728 anchor and 1,000
  Fiberlet chunks in the v3 namespace without the stale endpoint error. With
  the old one-million guard it then failed separately during stage-one exact
  analysis; the five-million guard completed all five stages.
- The unchanged requested command completed from a cold final
  toolchain-specific namespace with final populations of 19,963 anchors,
  165,561 incident Fiberlets, and 24,817 interior Fiberlets. A second hot run
  was not repeated because the managed workspace sandbox makes the external
  dataset cache read-only; the earlier contract-v3 hot validation completed
  with the same populations before toolchain identity was added.
- Three hot 512-region runs measured 2.724/2.748/2.786 seconds wall and each
  produced 3,373 anchors, 35,039 incident Fiberlets, and 4,477 interior
  Fiberlets. The previous documented median was 2.90 seconds.

## Deviations

- Final toolchain-specific hot timing was not repeated because the managed
  workspace sandbox cannot update cache bookkeeping in the external dataset
  directory without a new approval. Correctness was validated by the complete
  cold run, prior v3 hot run, focused cache reopen tests, and explicit
  cross-toolchain rejection.
