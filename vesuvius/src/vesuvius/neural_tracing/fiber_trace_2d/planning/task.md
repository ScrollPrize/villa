# Preserve Version-3 Fiber Spans During Sync

- Extend the fiber-aware three-way sync merge for `vc3d_fiber` version 3.
- Treat each CP-to-CP dense line slice and its `segment_to_next` descriptor as
  one atomic stored result.
- Merge independent changes only when unchanged base spans separate them.
- Never choose one side's per-span goal or result merely from generation.
- Reject ambiguous changes through the existing manual conflict workflow.
- Merge the top-level `optimization_mode` with base-aware scalar semantics.
- Preserve version-1 and version-2 merge behavior.
