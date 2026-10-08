# Full unsigned AFV constraint extraction and reusable database

## Task
Extract unsigned local events on all Qual Paris4 AFV fibers, without cuts or
normal alignment. Persist constraints, source identity, exact base points/arcs,
Lasagna normals at both points, all scores/distances/support/diagnostic values.
Run full extraction and report constraint types and per-fiber degree statistics.

## Implementation
- Extend existing native adapter with full catalog selection (no spatial limit),
  one extraction repetition and optional SQLite export. Shared extractor unchanged.
- New versioned SQLite artifact (.afc.sqlite, NOT an AFV geometry database):
  metadata/provenance/settings, all source fiber IDs including isolated fibers,
  constraint endpoints and typed primary scores plus lossless JSON of every
  FiberTraceConstraint field. Normals sampled with existing double-position batch
  API, preserving validity; null invalid samples, never fabricated zeros.
- Native transactional inserts with bounded normal-sampling batches. Index both
  fiber endpoints. Output temporary sibling, complete only after counts/integrity
  validation, close then rename; refuse overwriting existing output.
- Source AFV UUID and hashes from approved runner input records; record metadata
  and settings inside DB. No copied geometry: foreign identity refers to source AFV.

## Tests
Rebuild with approved command. Small slab export first, SQLite integrity/fk checks,
check all field/count round trips and endpoints against source AFV, no signed
events/continuations. Then full run through approved runner, 128 workers, verify
all512698fibers and degree sum=2*events. Independent plan review before run.

## Spec/docs/changelog
Document schema, units, validity and source AFV requirement; unsigned extraction
does not yield global winding order. Record timings, memory, database size, mean/
median/p95/max degree, isolated fibers, types and distance distribution in report.
Update current task log/status; preserve prior benchmark results separately.
