# Plan

1. Checkpoint existing work (8ed2c7911).
2. Shared bounded process-pool audit of trace JSON batches; coordinator validates unique keys/counts; bounded threaded writers publish audits. Preserve audit math. Recover completed batches before any new tracing, validated against candidate geometry and scientific identity.
3. Parallelize independent union annotation export, corridor geometry, chunk links and coverage checks. Preserve ordered output and inexpensive interval merging/selection.
4. Test equivalence, invalid/missing/duplicate artifacts and recovery. Measure saved artifacts, restart via approved runner and verify no retracing.

## Approved correctness correction
The review found the previous global worklist passed segment-local arcs to a full-fiber reader. User explicitly approved discarding those trials and rerunning. Replace ambiguous lo/hi API with named original-fiber arc fields and a versioned contract. Derive expected start/end XYZ independently from the historical eligible reference segment; validate native slice and saved trace within0.25 base voxel. Bump protocol version, remove old reproduction tree only (keep original evaluations and every inference cache), then rerun GPUs0–5,192 trace workers. Test both directions on nonzero run offsets and deliberate wrong endpoints in native CLI fixtures. This supersedes recovery of the incorrect current batch.

## Spec update
No spec.md exists. No scientific protocol changes.

## Docs updates
Record timings, resume evidence and limitations in task_log.md and benchmark history; update changelog.
