# Plan: fix large staged Fiberlet cache preparation

## Diagnosis and invariant

1. Reproduce the user command with its existing cache and capture the exact
   failing owner key, resolved status, and fetcher error instead of discarding
   them behind the generic message.
2. Trace the failed key through `prefetchScheduled`, `ChunkCache`, and the
   generated anchor/Fiberlet fetchers. The reproduced terminal cause is an
   endpoint-scoring mismatch between a previously persisted float anchor and a
   current canonical re-interpolation.
3. Compare the stale and fresh cache artifacts rather than treating the first
   near-equal normal as sufficient evidence. The anchor payloads, generated
   Fiberlet prefix/route payloads, and resulting Fiberlet population differ
   under the same declared algorithm namespace. Therefore the old and current
   producers are not cache-compatible.
4. Preserve the invariant that every valid scheduled Fiberlet owner is either
   generated/read successfully or fails with its original deterministic cause.
   Eviction must not turn a successful scheduled fetch into a false failure.

## Implementation

1. Increment the unpublished Fiberlet-generation contract in the structured
   algorithm identity. Current code must use a fresh anchor and Fiberlet cache
   namespace instead of mixing payloads produced by an incompatible binary.
   Include compiler identity/version and build configuration because candidate
   populations at hard thresholds are not record-identical across tested GCC
   Release and Clang Debug producers.
2. Retain strict endpoint-scoring validation across supported compilers.
   Validity and axis orientation remain exact; only finite componentwise
   float32 reconstruction differences within eight epsilons are accepted.
   Contract v3 prevents this narrow arithmetic allowance from mixing the known
   semantically different v2 producer. Do not change scoring, DP, path costs,
   or current serialization.
3. Keep preprocessing parallel and deterministic. Avoid retaining the full
   1024-region decoded population solely to make the wait loop succeed.
4. Include owner key, status, and underlying message in any remaining
   resolution failure so future failures are actionable.

## Tests and validation

1. Add focused regression coverage proving that the previous and current
   processing contracts produce different algorithm/dataset fingerprints for
   anchor, Fiberlet, and combined datasets.
   Verify that opening an explicit v2 root as v3 is rejected without deleting,
   repairing, or modifying the v2 dataset.
   Retain the exact scheduled owner/status/source-error formatter test without
   duplicating preprocessing infrastructure.
2. Run the relevant focused unit tests with GCC Release and Clang Debug.
   Verify the narrow endpoint comparator against the observed GCC/Clang
   reconstruction pair, and verify that explicitly mixing GCC and Clang cache
   roots is rejected by producer metadata before generation.
3. Run the exact user 1024-region command to completion and hot-reopen it.
   Confirm that it selects the new namespace, never reads the stale owner, and
   completes. Then hot-reopen it and rerun the established 512-region workload,
   verifying stable decoded populations/identities with no more than 10% warm-
   wall regression. Raw v2/v3 payload bytes are not compared because their
   fingerprint-bearing headers intentionally differ.

## Spec update

Clarify that generation-contract changes which can alter anchor or Fiberlet
payloads require a cache identity revision; mixed-producer endpoint evidence
remains a hard error. Require precise terminal error propagation.

## Docs updates

Document no new user-facing workflow unless the fix changes an observable
cache/progress contract. Record the actionable error format if retained.

## Changelog update

Record the large-region staged preprocessing correctness fix.
