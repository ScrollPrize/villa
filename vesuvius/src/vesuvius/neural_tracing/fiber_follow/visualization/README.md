# Model interpretation

Generate a measured report for a coordinate regression or flow-matching checkpoint:

```bash
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.visualization.interpret \
  --checkpoint output/RUN/last.pt --output-dir /tmp/fiber-atlas \
  --fiber-index 0 --arc-position 100
```

Use `--replay CACHE --row N` instead of an annotation position to inspect a recorded decision. Use `--help` for volume and fiber-source overrides. Each run requires a fresh output directory.

The report records CT inputs, observed history, predicted paths, survival confidence, and generation/refinement steps. Shared-component capabilities provide optional patch features, attention, and history interventions; support does not depend on an architecture version string. Flow integration steps are not candidate-selection attempts.

Instrumentation retains original forward kernels and must produce exactly the same outputs as an ordinary forward. Interventions are restored before checking the baseline again. Attention measures read allocation; one-decision interventions are not tracing-accuracy estimates.

The directory contains local HTML, PNG/PDF figures, numerical arrays, provenance, and validation results. Re-render or validate with `python -m vesuvius.neural_tracing.fiber_follow.visualization.render DIRECTORY` or `.visualization.validate DIRECTORY`.
