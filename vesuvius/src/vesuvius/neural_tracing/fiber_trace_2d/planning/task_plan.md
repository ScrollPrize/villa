# Plan: re-merge fiber-lets2

1. Commit the completed benchmark document, scale-0 config, and associated
   task records separately from the merge.
2. Merge `fiber-lets2` and inventory every conflict.
3. Replace conflicting task-local planning files with this concise current
   task record.
4. Resolve implementation conflicts semantically: retain the current
   anisotropic replay threshold and tangential broad-phase while accepting the
   float-based fiberlet extraction changes.
5. Verify the merge index and whitespace, build the affected targets with 32
   jobs, and run focused anchor, path, and replay tests.
6. Commit the completed merge.

## Spec update

No new spec change. The merge imports the already-reviewed `fiber-lets2`
specification updates without modification.

## Documentation update

No new documentation change. The merge imports the existing `fiber-lets2`
fiberlet documentation updates.

## Changelog update

No new entry. The source branch already records its retained performance
checkpoints in the durable changelog.
