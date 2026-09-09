# Plan: integrate Fiberlets development into Fiber3D training

## History and safety

1. Fetch the exact remote integration ref and record both tips, their merge
   base, divergence counts, and worktree cleanliness. Lock the reviewed
   incoming tip to `b660837d9ef8c0d59eb7783b1dee588d6f382529`; immediately
   before merging, require `origin/fiberlets-development-integration` to still
   resolve to that commit or stop and repeat the preview/review.
2. Use a merge commit on `fiber-training`. Do not rebase the 17 local training
   commits across the 206-commit integration history: preserving both branch
   identities and avoiding unnecessary history rewriting is preferable here.
3. Preview the three-way merge before mutation and classify textual conflicts,
   files changed on both sides that Git auto-merges, and one-sided additions.
   Commit these four required integration-planning files first so the worktree
   is clean and their exact reviewed content is preserved. Record the resulting
   first-parent tip and updated divergence before starting the merge.

## Merge and semantic conflict resolution

4. Merge exact commit `b660837d9ef8c0d59eb7783b1dee588d6f382529`
   with `--no-commit` so the complete index/worktree can be audited before the
   merge commit is created.
5. Resolve documentation/specification conflicts by retaining both the
   Fiber3D training contracts and Fiberlet/native-tracing contracts. Preserve
   the chronological changelog entries from both sides.
6. Replace only the transient current-task documents (`task.md`,
   `task_plan.md`, `status.md`, and `task_log.md`) with this integration task,
   as required by the local workflow. Prior task history remains in Git and
   durable outcomes remain in `changelog.md` and normative docs.
7. Review every overlapping code/test auto-merge, especially the shared tiled
   inference scheduler and native chunk cache, against both side diffs and the
   combined result. Make the smallest semantic fix needed if an automatic
   merge drops or invalidates either feature set.
8. Inspect the full staged merge delta, conflict-marker absence, unexpected
   deletions, diff formatting, and merge-parent topology before committing.

## Validation

9. Run focused Python tests covering the shared tiled inference changes and
   Fiber3D training changes. Reuse an existing configured native build directory
   and toolchain only as configuration, incrementally rebuild the merged
   `vc_core`, `test_chunk_cache`, `test_chunk_cache_persist`,
   `test_chunked_plane_sampler`, and `test_chunked_plane_sampler_more` sources,
   plus `test_volume_extras` (decode statistics/concurrency) and
   `test_chunked_plane_sampler_fallback` (requested-level locality/worker
   limits), then execute those freshly linked tests. Do not install or
   bootstrap dependencies.
10. If no usable configured native build exists or a required test cannot run
    in the available environment, record the
   exact command and reason rather than silently skipping it.
11. Obtain an independent review of the conflict resolutions, automatic-merge
    audit, test evidence, and final staged tree before creating the merge
    commit.
12. Commit the reviewed merge on `fiber-training`, verify its two parents and
    clean worktree, and report the conflicts and feature-preserving resolutions.

## Spec update

No new product behavior is introduced by the merge itself. Reconcile the
existing `planning/specs.md` additions from both histories without deleting or
weakening either branch's requirements.

## Docs updates

Reconcile `docs/code_structure.md` by preserving both Fiber3D training and
Fiberlet/native-tracing documentation. No new standalone user guide is needed.

## Changelog update

Preserve both histories' durable entries and add a concise integration entry
only if conflict resolution changes their organization or adds new behavior.
