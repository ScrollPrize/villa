# Native VC3D 3D Fiber Tracer Status

- [x] Read project workflow instructions.
- [x] Inspect relevant VC3D volume, Lasagna, and line annotation entry points.
- [x] Write current task into `planning/task.md`.
- [x] Write detailed implementation plan into `planning/task_plan.md`.
- [x] Review `task_plan.md` against `task.md`/`specs.md` while implementing.
- [x] Update `planning/specs.md` with native VC3D tracer requirements.
- [x] Update docs for project dataset storage and GUI workflow.
- [x] Extract shared C++ Lasagna compact-channel sampling helpers and port
  existing normal sampling to them.
- [x] Implement native core tracer library.
- [x] Implement project-level fiber inference dataset storage.
- [x] Implement Ctrl-right-click GUI action and edit blocking via the existing
  optimization busy state.
- [x] Add focused native tracer test and rebuild the VC3D target.
- [x] Run validation commands and record results in `planning/task_log.md`.
- [x] Add changelog entry.
- [ ] Implement persisted tracer-optimized segment metadata/invalidation.
- [ ] Protect unchanged tracer-optimized segments from regular Lasagna
  reoptimization.
- [ ] Add full Python-equivalent multi-step beam lookahead in the C++ core.
- [ ] Add UI progress text/percentage beyond the existing busy overlay and
  worker log messages.
- [ ] Add real-data parity validation against the Python reference command.
