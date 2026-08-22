# Plan: whole-volume fiberlet preprocessing

1. Add a reusable canonical-presence chunk scan to `FiberPredictionField`.
   Read each physical presence chunk once through the existing Zarr reader,
   classify missing and decoded-all-zero chunks as inactive, and return
   non-empty chunk bounds in deterministic Z/Y/X order. Keep channel binding
   and Zarr layout knowledge inside the existing prediction-field
   implementation.
2. Convert non-empty presence chunks into the conservative set of overlapping
   spatial output chunks. A non-empty input chunk activates every output chunk
   it overlaps; no direction channel participates in sparse eligibility.
3. Keep the combined `anchors/`, `prefix/`, and `routes/` arrays, but remove
   `active_chunks.bin`, `dataset.complete`, and per-chunk completion markers.
   Configure the expected active set in memory from the fresh input-presence
   scan. A combined chunk is found only when all three payload files exist,
   decode with the expected codecs, and form a valid prefix/route pair.
   Expected-inactive chunks resolve as canonical empty payloads; missing active
   chunks remain incomplete. Read-only combined graph facets require the
   expected set and verify every expected triple from the payload files. Every
   standalone combined reader must therefore reopen the source manifest, run
   the same canonical presence scan, and configure that expected set before it
   can construct stored graph facets.
4. Add a core whole-volume preprocessing driver around
   `FiberletOnDemandPreprocessor`. First generate the union of anchor chunks
   required by all active output chunks, including existing exact dependency
   halos. Then convert each owned float anchor chunk into the compact final
   dataset and generate its compact prefix/route pair from the intermediate
   anchors.
5. Add `vc_fiberlets preprocess-volume FIBER_MANIFEST OUTPUT_ZARR` with
   required `--normal-manifest` and optional `--anchor-cache`. Default the
   durable anchor output beside the final output as `<stem>.anchors.zarr`.
   Reuse the existing extraction, path, cache-budget, thread, remote-cache,
   storage-chunk, and threshold options.
6. Schedule both stages in stable Z/Y/X order and report presence scan,
   eligible/skipped chunks, anchor progress, fiberlet progress, resume hits,
   elapsed wall time, and output locations. A stage reporter runs independently
   of chunk completion, refreshes one terminal line about once per second,
   writes a persistent newline every minute and at stage completion, and shows
   completed/total chunks, percent, elapsed, rate, ETA, and current/projected
   payload bytes based on the visited-chunk mean. Re-scan input, anchor
   dependencies, and final triples on every invocation; do not require strict
   completion order from parallel workers.
7. Keep payload writes individually atomic and use the validated three-file
   final tuple as the logical completeness unit. An interrupted tuple may leave one
   or two canonical payload files; resume must accept matching existing files,
   generate missing files, and reject conflicts. Remove stale files matching
   the atomic writer's exact temporary suffix convention when opening/resuming
   and once workers have shut down. Put suffix recognition and recursive
   cleanup beside the atomic writer rather than duplicating its convention.
   Hold exclusive directory locks on both preprocessing roots throughout the
   scan/generation run so cleanup cannot remove another writer's live file.
8. Remove legacy `active_chunks.bin`, `dataset.complete`, and `complete/`
   artifacts while opening a local fiberlet dataset. Remove completion-marker
   gating from ordinary prefix/route datasets as well as combined datasets.
   Unexpected final payloads are ignored and inaccessible unless the current
   input scan expects their owner; anchor-cache extras remain reusable because
   dependency halos legitimately form a superset.
9. Add focused tests for presence-only sparse selection, missing/zero/nonzero
   source chunks, marker-free combined round trip, incomplete tuples, conflict
   rejection, empty tuples, corrupt and mismatched tuples, zero active chunks,
   unexpected final payloads, legacy-artifact removal, stale temporary cleanup,
   cleanup on exception, reconstructed-reader reopening, dependency-halo anchor
   generation, Z-priority schedules, resume behavior, and the existing
   presence-floor early exit. Build validation covers the CLI-local progress
   reporter; a bounded production smoke remains unavailable because the command
   has no bounded-region mode.
10. Build `vc_fiberlets` and focused tests, run the tests, then run a bounded
   representative preprocessing smoke test before attempting a full-volume
   production run.

## Spec Update

- Specify the whole-volume command, two durable outputs, presence-only sparse
  eligibility, compact final encoding, dependency halos, marker-free scan-based
  completion, atomic payload recovery, temporary cleanup, and Z-priority
  scheduling.
- Clarify that `--presence-floor` is an observation eligibility threshold and
  already prevents seed/refinement work for wholly sub-threshold cells.

## Docs Updates

- Add command syntax, output layout, scan-based sparse/resume behavior, atomic
  publication and cleanup, ordering, threshold semantics, and operational progress to
  `volume-cartographer/docs/fiberlets.md`.

## Changelog Update

- Record whole-volume anchor/fiberlet preprocessing and the combined compact
  dataset format.
