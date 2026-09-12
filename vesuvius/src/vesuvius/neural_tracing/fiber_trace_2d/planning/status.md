# Single-backward status

- [x] Inspect current loss/count and queue ownership code.
- [x] Plan independently reviewed; preserve nonempty-group normalization.
- [x] Implement target prepass and single backward for ordinary one-branch and conditioned modes.
- [x] Reject embedding/ordinary multi-branch losses with explicit TODO errors, per user approval.
- [x] Expanded CPU/zero-work regression:38passed/15CUDA skips,30.94s.
- [x] Bounded loader cleanup and independent owned-process stop guard; CPU suite31passed/2CUDA skips,88.15s.
- [x] Final stop-guard review and shutdown suite:61passed/2CUDA skips,118.43s.
- [x] Manager-exclusive CUDA equivalence/speed:3passed11.78s; one-H100 B4 BF16 forward/loss/backward16.25→38.93samples/s.
- [x] Manager-exclusive two-GPU resize/resume regression:2passed43.96s.
- [x] Old root1200528 killed with explicit approval after compute workers gone;
      GDB attachment failed, exact original futex site remains unproven.
- [x] Docs, TODO limitations, measured model-only performance and test commands recorded.
