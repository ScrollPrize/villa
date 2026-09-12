# Single-backward bounded training

Fix the incorrect repeated component-backward implementation: determine the
target-derived normalization counts before model forward, then use one combined
backward per microbatch, preserving the full effective batch and bounded GPU
activation memory. Verify gradients, managed lifecycle and measured speed.

User-approved scope: temporarily reject embedding and ordinary multi-branch
losses with explicit errors/TODOs; do not change their objectives or silently
fall back. Keep ordinary one-branch and conditioned losses target-normalized
with one backward per microbatch. Existing inference models remain usable.

Also fix reliable training termination: user stopped root1200528, but shutdown
hangs and its manager allocation remains held. Diagnose live wait and remove
unbounded shutdown/IPC waits; escalate only owned training processes, never the
manager or unrelated tasks. Report genuinely unkillable kernel-I/O survivors.

## Approved commands for upcoming reproduction experiments

Workspace: `/home/hendrik/villa`; branch: `fiber-train-reproduce`.
The upcoming autonomous experiment task is not yet specified. The following
reusable command prefixes are approved; approval does not itself start a study.

- `/home/hendrik/villa/vesuvius/scripts/run_fiber3d_experiment`
  — approved including future subcommands. Currently a placeholder; implement
  `start` and `stop` before use. Experiment settings can be edited behind this
  fixed launcher without requesting a new approval for each argument change.
- `/home/hendrik/villa/vesuvius/scripts/run_fiber3d_profile_benchmark`
- `/home/hendrik/villa/vesuvius/scripts/run_fiber3d_pipeline_benchmark`
- `/home/hendrik/villa/vesuvius/scripts/run_fiber3d_blur_benchmark`
- `/home/hendrik/villa/vesuvius/scripts/run_fiber3d_cooperative_tests`
  — existing fixed benchmark/test entrypoints; edit their control settings.
- `git stash`, `git switch`, `git merge`, `git add`, `git commit`
  — approved prefixes for preserving, isolating and recording experiment work.
  No blanket approval for `git checkout`, arbitrary Python, or `kill`.

Use `/home/hendrik/.venv_las`; experiment artifacts belong under
`/ephemeral/hendrik/fiber3d_train2`. All GPU work must obtain manager allocation:
adaptive production training or exclusive, nonpausable/nonadaptive benchmarking.
The experiment stop command must identify only its own processes, escalate
graceful shutdown to SIGKILL, and report survivors instead of claiming success.
Uninterruptible kernel I/O cannot be forcibly completed; zombies are already
dead and require parent reaping. Never kill unrelated training or the manager.
