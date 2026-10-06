# Raw loss reporting — 2026-10-06

- Existing work committed as345952e32(native baking/preparation) and82d78104e(annotation heads, supervision, views/tests/docs).
- Production training must not be stopped or restarted. This change affects subsequent launches only.
- Generated workdirs datasets and perf.data excluded from code commits.
- Independent plan review approved. Implementation review caught zero-weight refinement reporting: retained raw geometric numerators instead of dividing zero; pending vectors keep weighted prefix and explicitly upgrade. Disabled direction has no sampled map, and old zero-weight pending terms are unrecoverable: omit these unavailable metrics, not a fabricated zero.
- Initial regressions caught raw embedding zero retaining an unnecessary graph (fixed with detach), and the mocked collective's hardcoded layout (updated for raw statistics). Optimization sums and gradient path unchanged.
- Independent implementation re-review cleared zero-weight reporting and pending-state upgrades. The empty-refinement test was updated for the appended raw reporting suffix.

## Validation command (CPU only, approved runner)

```bash
/home/hendrik/.venv_las/bin/python -m pytest -q vesuvius/tests/neural_tracing/test_loss_reporting.py vesuvius/tests/neural_tracing/test_normal_tube.py vesuvius/tests/neural_tracing/test_fiber_annotations.py vesuvius/tests/neural_tracing/test_refinement.py vesuvius/tests/neural_tracing/test_fiber_embeddings.py vesuvius/tests/neural_tracing/test_local_competition.py vesuvius/tests/neural_tracing/test_paired_embeddings.py vesuvius/tests/neural_tracing/test_gradient_accumulation.py vesuvius/tests/neural_tracing/test_forward_visualization.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py vesuvius/tests/scripts/test_collect_reviewed_fibers.py
```

Synthetic annotated tubes, sidebands, paired/split embeddings, refined curves,
mocked distributed reductions, accumulation parity, loader/views and reviewed-data
preparation. No new GPU run for scalar-only logging; existing optional GPU tests
remain opt-in. Production was neither stopped nor restarted.

Final result:442 passed,28 optional skips,2 existing xfails in39.44s; two existing
Pillow deprecation warnings. Weighted objectives and gradient parity verified;
raw reporting tested at zero/non-unit weights and across microbatch aggregation.
git diff --check passed. No performance claim or training parameter change.
