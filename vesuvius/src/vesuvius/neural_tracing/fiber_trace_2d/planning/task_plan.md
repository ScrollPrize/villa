# Plan: write-back memory cache for temporary Fiberlet reduction layers

## Baseline and invariants

1. Use the committed Release binary and the established four-stage Paris4
   workload as the baseline: 4.37 s wall from the user run and 4.18 s locally.
2. Preserve canonical stage counts, retained-ID digests, serialized payload
   bytes, prefix/route pairing, monotone removal, and deterministic errors.

## Shared write-back store

1. Add a reusable temporary Fiberlet write-back store in the existing
   `FiberletDataset` module. Do not duplicate serializers, validators, atomic
   publication, or overlay fallback behavior in the CLI.
2. Key entries by a stable registered layer ordinal plus owner chunk. Store
   only canonical serialized bytes: either one anchor buffer or one atomic
   prefix/route pair. Decoded payload ownership and accounting remain exclusive
   to `ChunkCache`, avoiding duplicate residency and double accounting.
3. Use a per-owner generation and explicit `ResidentDirty -> Queued/Writing ->
   SpilledClean` state. A rewrite serializes behind an older pending generation;
   an old completion can never erase or shadow newer bytes. Reads use resident
   or still-owned pending bytes, otherwise wait only for that key before disk.
4. Maintain deterministic LRU victim choice by access epoch with stable
   layer/kind/ZYX tie-breaking. Logical output determinism is independent of
   concurrent touch order; writer completion order never selects errors or
   content.
5. Share one store across all temporary stages. Charge resident dirty buffers,
   queued/writing buffers until actual release, and conservative entry overhead.
   Reduce the existing graph `DecodedChunkCacheBudget` ceiling by exactly that
   live amount, preserving the existing evaluation-cache reservation. If live
   write-back bytes exceed the shared ceiling, queue oldest resident entries
   and apply bounded backpressure until pending buffers are actually released.
   One active return-value copy may transiently exceed the ceiling.
6. Use a bounded asynchronous writer queue. Prefix and route buffers share one
   job and remain logically visible as a pair until both files succeed. On
   failure remove partial new output, retain the deterministic lowest logical
   key error, and fail explicitly. Runtime visibility is pair-atomic; the
   existing two-file disk layout is not made crash-transactional.
7. `finish()` waits for already-started spills, surfaces errors, discards
   never-evicted dirty entries, restores the decoded-cache allowance, and runs
   before invocation-local tree cleanup. Destruction is nonthrowing and only a
   defensive drain.

## Staged reduction integration

1. Attach the shared write-back store only to invocation-local stage anchor and
   Fiberlet datasets. Authoritative on-demand anchor/Fiberlet caches retain
   their existing durable behavior.
2. Add distinct mutable `replaceAnchor` and `replacePair` backend operations
   after the existing strict subset/pair validation. Authoritative immutable
   publication keeps its existing conflict behavior.
3. Keep generated overlay cache reads unchanged at the call sites: dataset
   reads transparently resolve memory, pending spill, disk, then lower-layer
   fallback.
4. Compute detailed payload diagnostics over the canonical relative-path union
   of metadata files, spilled chunks, and the latest resident/pending logical
   chunks. Memory shadows stale disk, temporary writer files and lower fallback
   are excluded, and prefix/route pairs are snapshotted together. Hashing pins
   or streams entries without flushing them and must match a forced durable
   directory hash byte-for-byte.

## Tests and validation

1. Add focused tests for memory hits, deterministic LRU order,
   budget-triggered asynchronous spill, pending read, stale-generation rewrite,
   a budget smaller than one pair, shared-budget pressure/restoration, explicit
   empty fallback through multiple stages, pair visibility, teardown without
   unnecessary writes, and injected anchor/prefix/route write failures.
2. Re-run one-versus-multiple-thread staged equivalence and the focused
   Fiberlet storage/path tests.
3. Preserve at least three warm pre-change Release measurements, then build the
   ordinary Release binary and run the identical four-stage workload at least
   three times. Report min/median/max wall time, CPU use, peak RSS, write-back
   hit/spill/peak-live-byte statistics, and exact stage IDs/payloads.

## Spec update

Specify the invocation-local write-back LRU, shared memory accounting,
prefix/route atomicity, eviction/spill/read ordering, and teardown semantics.

## Docs updates

Document that staged temporary layers are memory-first, how `--cache-gib`
governs them, when asynchronous spill occurs, and the measured four-stage
performance.

## Changelog update

Record the asynchronous write-back overlay cache and its exact-output
performance improvement.
