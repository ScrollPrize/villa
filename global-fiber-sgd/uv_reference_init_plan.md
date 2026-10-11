# Reference-initialized full joint solve

User requests one full joint solve, every global variable optimized immediately.
Use CCW,151windings,10full passes,shared24GiB GPU reservation. No strong follow-up,
no rejection warmup,no entropy ramp,no coordinate/width/logit freezing.

Initialize winding distributions from cached source-median reference windings,
compensating for anchor phase and signed midpoint arc so MAP physical winding
matches reference rather than blindly treating bucket index as physical winding.
Keep Gaussian sigma3 and1%broken initial probability, zero coordinate offsets,
radial circumference width initialization. Unmatched fibers keep radial priors;
retain all input fibers/constraints. Reference is initialization/evaluation ONLY.

## Spec update
Optional reference initialization with validated projection provenance and no
UV checkpoint. Explicit entropy_ramp=false for constant entropy objective.
All shared anchors, per-winding offsets, widths and probabilities train from0.

## Docs/tests
Log exact matched/unmatched initial counts and baseline reference score; per-pass
L1,±2%,q10/q90,retention and sign metrics. Regression tests for corrected bucket
centers, unmatched fallback, all-positive support and1%broken probability.
Approved runner only. Persist completed results and no automatic extra stage.
