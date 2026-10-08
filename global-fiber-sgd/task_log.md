# Shared BP integration — 2026-10-08

- User requested merge of the existing BP implementation, then an overview of
  Lasagna constraint extraction. No new extractor or SGD implementation requested.
- Committed initial AFV reconnaissance as 5303c59be; source download remains
  ignored and intact. Earlier reconnaissance details remain in README/metadata.
- Merged fiber-exp5 at d7deec459, which includes shared extractor/BP changes
  through be3b45d36 and committed tracing infrastructure. This is the full branch
  merge, not a selective port. Uncommitted skeleton work remains in its old stash.
- Only conflict: volume-cartographer/python/CMakeLists.txt. Retained the union
  of runtime/install libraries: vc_fiber_tracer plus main's volcomp libraries.
- Inspected auto-merges: main's AFV reader/SQLite and volcomp support retained;
  remote Lasagna store has both deferred-auth work and main's trailing-range API.
  Extractor, normal alignment and winding BP sources match fiber-exp5 exactly.
- Approved native build attempted; configuration blocked by missing SQLite3
  development headers/library. No installs, native reruns or old-binary claims.
  Minimal Ubuntu prerequisite command is `sudo apt-get install libsqlite3-dev`,
  subject to approval. Other build prerequisites remain unverified beyond this.
- Python benchmark support: 12 passed (plot and crop-fiber JSON tests).
- Whole imported history has existing whitespace warnings, especially generated
  SVGs; left untouched. Conflict-resolution/new-doc checks are separate.
- Overview in bp_constraints.md distinguishes legacy/local-event extraction,
  magnitude versus sign, source-supported certification and downstream policies.
- No AFV adapter, Lasagna volume download or global solve run in this task.
- Independent review approved CMake target union and inspected auto-merge
  overlaps. Added the requested closest-connector/local-mode clarification.
