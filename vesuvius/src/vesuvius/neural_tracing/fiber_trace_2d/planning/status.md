# Status: anisotropic Gaussian fiber-presence supervision
The first managed half-sigma launch restored update 18,000 but incorrectly used
GPUs 0-6 and failed in the initial test because a partial final evaluation wave
violated DDP forward parity. The scheduler now pads only execution (not
metrics/refinement) on empty ranks. The first corrected six-GPU launch exposed
shared ownership of its padded broker sample; test leases are now
reference-counted. Gaussian ridge, distribution-negative, and background
losses are independently normalized and included in the single-backward path.
Relaunch verification is pending.
- [x] Preserve pre-task dirty source changes in focused commits.
- [x] Inspect current target, loss, native-normal, refinement, and checkpoint paths.
- [x] Write task and implementation plan.
- [x] Independently review the plan against repository guidance.
- [x] Implement configuration and compact normal metadata.
- [x] Implement Gaussian tube materialization and soft presence loss.
- [x] Add and run regression tests.
- [x] Add production config and documentation.
- [x] Extend direction supervision across Gaussian support with weighted normalization.
- [x] Restrict managed production launch to GPUs 0-5.
- [x] Launch managed training from best EMA-0.999 and verify progress.
- [x] Commit the completed implementation.
- [x] Stop the sigma-2 run cleanly at the user's request.
- [x] Implement persistent bounded Gaussian-orientation refinement.
- [x] Validate orientation refinement and legacy archive compatibility.
- [x] Preserve DDP parity and shared broker ownership in partial test waves.
- [x] Implement explicit 50/25/25 Gaussian presence normalization.
- [ ] Resume from its latest full checkpoint with half sigma on GPUs 0-5.
