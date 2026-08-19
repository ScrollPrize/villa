# Plan: merge current fiber-lets2 speed improvements

1. Commit the proposed fiberlet storage format separately.
2. Merge `fiber-lets2` and inventory conflicts.
3. Replace conflicting task-local planning files with this concise merge record.
4. Retain all cleanly merged C++ speed changes and durable documentation.
5. Build the affected targets with 32 jobs and run focused tests.
6. Commit the validated merge.

## Spec update

No new spec change. Retain the speed branch's already-reviewed specification
updates.

## Documentation update

Retain the speed branch's `volume-cartographer/docs/fiberlets.md` updates.

## Changelog update

No additional entry. The durable changelog already records the imported speed
checkpoints.
