# Task: Mixed-fiber direction-label ablation

Extend the stored crop-trace direction diagnostic so it cumulatively admits
more fibers initially classified as mixed, reruns canonical constraint
extraction and the discrete H/V-only-plus-broken MILP at every checkpoint, and
reports how the optimized labels diverge from the initial direction/defect
assignment. Initially mixed fibers are defects and are expected to optimize to
Broken, not to receive a tentative H/V direction.

Use coarse cumulative checkpoints rather than solving after every individual
fiber, and compare the discrete MILP with the same model's LP relaxation after
thresholding active and H/V values at 0.5.

Allow a diagnostic admission limit so broken-cost sweeps can evaluate a small,
identical prefix of ranked mixed defects before running the complete ablation.

The ablation must preserve the original confident direction-1/direction-2
population as a separate diagnostic cohort so degradation cannot be hidden by
the growing uncertain population.
