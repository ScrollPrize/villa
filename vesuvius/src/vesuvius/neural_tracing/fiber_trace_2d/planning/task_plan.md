# Plan: render existing forward results

1. Shared bounded observer of loss forwards renders already-materialized samples and detached predictions with the existing contact-sheet helper; retain CPU sheets only.
2. Initial and periodic train/test use the same path. Train selects local samples instead of fetching canonical global samples. Keep test captures in pending-test RAM handoff across resize, with no reload fallback.
3. Remove train-visual broker IPC. Preserve losses, schedules, logging tags and model mode.
4. CPU tests for forward count, actual tensor reuse, bounds, unchanged gradients/mode, plus existing evaluation tests. No GPU runs/live job changes.

## Spec update
Replace canonical/refetched visualization with bounded actual-forward local captures. Preserve macro-batch, evaluation and snapshot semantics.

## Docs updates
Update cooperative training docs, specs, status, task log and changelog; record validation and limitations.
