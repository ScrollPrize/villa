# VC3D Lasagna Attachment And Generic Remote File Cache

## Planning

- [x] Read the current user request and repository-level instructions.
- [x] Read `AGENTS.md`, `planning/plan.md`, `planning/specs.md`, existing
  code-structure documentation, and local development notes.
- [x] Inspect current VC project schema, Lasagna remote loading/cache behavior,
  Open Data manifest caching, VC3D menus, and Line Annotation fiber hooks.
- [x] Replace `planning/task.md` with the current task.
- [x] Create `planning/task_plan.md` with implementation, testing, spec, docs,
  changelog, risk, and follow-up sections.
- [x] Incorporate user clarification that both explicit
  `lasagna-remote.json` origins and direct remote-manifest origins remain
  supported.
- [x] Incorporate automatic ordinary project-volume attachment for all
  manifest data while keeping VC3D and generic volume loading strictly 3D.
- [x] Require one actual ZYX array per attached channel and reject older flat
  CZYX preprocessing/fit intermediates in VC3D paths.
- [ ] Obtain an independent agent review of the plan against `task.md`,
  `specs.md`, and `plan.md` (blocked by the active no-delegation runtime
  policy; direct consistency audit completed instead).
- [x] Incorporate direct consistency-audit findings.
- [x] Obtain user feedback/approval before implementation.

## Implementation

- [x] Extract shared remote URL/fetch/cache identity primitives mechanically.
- [x] Implement and test the arbitrary remote-file cache.
- [x] Integrate persistent cached manifests into Lasagna `openLocation()`.
- [x] Preserve and test explicit-sidecar and direct-origin remote group
  resolution.
- [x] Port Open Data Lasagna manifest publication to the generic cache.
- [x] Canonicalize tagged project Lasagna entries and legacy migration.
- [x] Implement Lasagna-owned preparation of flat ordinary 3D project volumes.
- [x] Implement atomic manifest plus derived-volume project attachment.
- [x] Reuse VC3D remote auth/cache-root plumbing for manifest attachment.
- [x] Implement local/remote Lasagna project attachment service.
- [x] Add VC3D menu actions and Detach entries.
- [x] Update VC3D project Lasagna resolution, including tagged fiber entries.
- [x] Complete the compatibility call-site audit.

## Validation And Documentation

- [x] Build all focused core, CLI, and VC3D targets.
- [x] Run focused cache, Lasagna, project, and Open Data tests; validate the Qt
  attachment integration by compiling VC3D (no focused Qt interaction test was
  added).
- [x] Run the applicable broader regression suite.
- [x] Update `planning/specs.md`.
- [x] Add/update implementation documentation.
- [x] Update `planning/changelog.md`.
- [x] Record commands, results, findings, and deviations in `task_log.md`.
- [x] Report remaining risks and the deferred interactive Line Annotation usage
  test in the final response.
