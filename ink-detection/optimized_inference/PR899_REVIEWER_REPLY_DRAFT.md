# PR #899 Reviewer Reply Draft

Thanks for the direct feedback. I agree with the concern: the original PR text
overstated the evaluation and did not separate generated scaffolding from
human-checked reasoning clearly enough.

I went back through the Primus optimized-inference path manually and narrowed
the claim to what I can defend:

- `MODEL_TYPE=primus` now goes through the same runtime contract gate as the
  existing model types.
- The entrypoint dispatches to a Primus-specific `load_model`.
- The wrapper preserves the optimized-inference tensor contract:
  `(B, 1, D, H, W)` stays in that shape for Villa `NetworkFromConfig`.
- The checkpoint loader normalizes the production checkpoint envelope and
  selects the `ink` target when present.
- The GPU image now has an explicit opt-in dependency path for Primus:
  `INSTALL_PRIMUS_DEPS=1`, with `VILLA_REPO` and `VILLA_REF` build args for
  fork/branch validation.

I also added a human evaluation note in
`ink-detection/optimized_inference/PR899_HUMAN_EVALUATION.md` so the reasoning
and remaining limitations are auditable in the branch rather than hidden in a
PR comment.

Validation I ran locally:

```bash
python -m unittest tests.test_runtime_contracts tests.test_model_primus tests.test_model_primus_integration -v
```

Result: 11 tests passed. The integration test installs the local
`vesuvius[models]` package, builds a minimal real `NetworkFromConfig` Primus-S
model, saves a production-style checkpoint envelope, reloads it through
`model_primus`, and checks a `(1, 1, 16, 16, 16)` optimized-inference tensor.

One limitation remains: I have not completed an end-to-end Docker container
smoke on my VM. Docker is part of the official Villa workflow here, and I agree
that the container path should be tested before claiming full runtime support.
I installed Docker 29.5.2 locally and got the daemon/containerd/DNS path far
enough to pull images and build metadata-only Dockerfiles, but real container
execution is blocked on this host by missing privileged rootless prerequisites
(`newuidmap`/`newgidmap` or system Docker/AppArmor policy). The observed
failures are `runc` cgroup/devpts setup errors and subordinate-ID layer
extraction errors.

So the current state is: the Python loader contract is tested, the Docker
dependency path is explicit, and I am not claiming the GPU container smoke has
passed until it has actually run on a host with working container execution.
