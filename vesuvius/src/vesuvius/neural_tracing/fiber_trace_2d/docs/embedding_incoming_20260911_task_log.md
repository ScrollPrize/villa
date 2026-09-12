# Manager-only launch documentation

The prior embedding command bypassed the manager. The supplied embedding config has no training.cooperative request or measured calibration. The manager CLI is not installed in the venv; its source module can run using PYTHONPATH. This task updates guidance without inventing memory estimates, launching GPU jobs, or changing internal test launchers.

- User expanded scope: implemented managed-by-default run_training and explicit --unmanaged opt-out. Shared default socket helper preserves the existing path; config/CLI overrides remain. The explicit profiling harness now passes --unmanaged; internal cohort training primitives are unchanged.
- Independent review approved the design; updated the stale torchrun recommendation in errors, retained fatal missing-calibration errors, and tested auxiliary prefetch dispatch.
- CPU suite (managed launch, calibration, cooperative integration, embeddings): 47 passed, 1 GPU skip in 18.22s. After adding auxiliary dispatch coverage, focused launch/calibration tests: 15 passed in 1.69s (48 distinct passing tests total). CLI --help and git diff --check passed.
- No GPU execution or service startup. The real embedding config still requires a complete request and configuration-matched measured calibration; this is documented, not replaced with invented values. Previous seed fix remains in this uncommitted worktree.

- User refined socket policy: shared helper now prefers /run/villa-gpu-manager/manager.sock if a listener is reachable (100ms connect-only probe) or its creation location is writable/searchable; otherwise home path. No filesystem creation during selection, no task registration, no postselection fallback. Server group/UID/ownership checks unchanged.
- Review caught inaccessible-parent OSError handling and eager CLI probing despite explicit overrides; both fixed and covered. Final socket/launch/calibration tests: 22 passed. No server startup or permission changes performed.
