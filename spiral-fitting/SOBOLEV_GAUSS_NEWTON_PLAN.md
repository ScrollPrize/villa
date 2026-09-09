# Sobolev-Damped Truncated Gauss-Newton Plan

## Goal

Add an opt-in flow-lattice optimizer that computes an approximate step `d` from

```text
(G + lambda A) d = -g,
```

where `g` is the current flow-parameter gradient, `G` is a matrix-free
generalized Gauss-Newton (GGN) operator, and `A` is a symmetric positive-definite
Sobolev metric. Solve the system approximately with preconditioned conjugate
gradients (PCG). Large `lambda` should approach a Sobolev-gradient step
`-(1/lambda) A^-1 g`; small `lambda` should approach a GGN step.

This changes only optimization geometry, not the forward model or loss. Keep
the existing AdamW path as the default and preserve checkpoint compatibility.

## Fast prototype on the current branch

Build the experiment on `spiral-flow-grad-conditioning`, reusing its flow
parameter grouping, physical smoothing-width conversion, nonfinite-gradient
handling, influence masks, diagnostics, and configuration backfill. The goal
of this phase is to determine whether curvature-aware, spatially conditioned
steps are promising—not yet to supply a rigorous production optimizer.

Implement the prototype in the following order:

1. Bypass `LazyMomentAdamW` for flow parameters while retaining the existing
   optimizer for every non-flow parameter.
2. Start with Cartesian flow lattices and a fixed batch/random state. Implement
   `A(v) = v - l^2 Laplacian(v)` and an approximate `A^-1(v)` using a fixed
   small number of CG iterations.
3. First run with `G = 0`, giving the reference Sobolev step
   `d = -(1/lambda) A^-1 g`. Compare it with the branch's Gaussian-smoothed
   AdamW and shared-denominator variants.
4. Add a matrix-free Hessian-vector product using PyTorch second derivatives
   and solve `(H + lambda A)d = -g` with 5–10 outer PCG iterations. Detect
   nonfinite values and non-positive curvature; fall back to the Sobolev step.
5. Use fixed damping and a conservative configurable step multiplier. Defer LM
   retries, line searches, persistent curvature history, and automatic damping.
6. For an early cylindrical experiment, permit the existing Gaussian smoother
   to act as an approximate `A^-1` preconditioner. Label this path explicitly
   as heuristic: its border normalization and unequal-ring radial interpolation
   have not been shown self-adjoint under the physical cell measure.

Log wall time, inner/outer iteration counts, residual reduction, termination
reason, step RMS/roughness, and loss before/after on the same batch. The minimum
ablation is plain AdamW, smoothed AdamW, smoothed shared-denominator AdamW,
Sobolev gradient, and damped Hessian-free. A useful prototype result is evidence
about convergence and update quality; it need not yet outperform AdamW in
wall-clock time.

Do not initially refactor every loss into residual form. The full Hessian may
be indefinite, so negative-curvature handling is mandatory. If the experiment
is encouraging, replace it with the positive-semidefinite GGN formulation and
rigorous metric below.

## Production follow-up: scope and staging

The remaining sections describe the reviewable production design, not
prerequisites for the fast prototype.

1. Implement and validate the method on Cartesian flow lattices first.
2. Add the cylindrical metric only after its mass weights, vector transport,
   and adjoint tests pass.
3. Optimize only flow parameters with this method; continue using AdamW for
   pitch, gap, shell, and other parameters.
4. Initially run on a deterministic accumulated batch. Do not reuse curvature
   information across batches or optimizer steps.

## Discrete Sobolev metric

Implement a matrix-free `SobolevMetric` with `apply_A(v)`, `solve_A(r)`, and
`quadratic_norm(v) = v.T @ A(v)`.

For Cartesian fields, use

```text
A = M + lz^2 Lz + ly^2 Ly + lx^2 Lx,
```

with `M = I`, periodicity disabled, and explicit Neumann boundaries unless a
cell is pinned or influence-masked. Eliminate constrained cells from the
operator (equivalently impose `d = 0` there); do not solve and mask afterward.

For cylindrical fields, construct `A` from an explicit quadratic edge energy:

```text
sum_i Vi |v_i|^2 + sum_(i,j) wij |v_i - Tji v_j|^2.
```

Use physical cell-volume weights `Vi`, cyclic angular edges, z edges, and
undirected radial edges based on angular-cell overlap between unequal rings.
`Tji` must rotate local `(r, phi)` components into a common basis. Pin ring 0.
Build `apply_A` from the energy/adjoint pair so that symmetry is guaranteed;
do not reuse the current radial blur directly.

Start `solve_A` with a fixed-tolerance inner CG or separable Cartesian Helmholtz
solver. A multigrid or spectral implementation is a later performance change,
not part of the correctness MVP.

## Matrix-free GGN

Refactor each differentiable loss contribution to expose its residual `r(p)`
and scalar weighting. Define

```text
G(v) = J.T @ W @ J @ v,
```

using JVP followed by VJP, without materializing `J` or `G`. Include only losses
with a valid positive-semidefinite residual-space curvature. Treat unsupported
terms as first-order contributions to `g` during the MVP, and report which
terms are omitted from `G`.

Verify every GGN-vector product on a tiny lattice against an explicitly formed
Jacobian. Check linearity, symmetry, and nonnegative curvature numerically.

## Truncated PCG step

Solve `(G + lambda A)d = -g` independently for each flow stage, or jointly if a
loss couples stages and the JVP/VJP path preserves that coupling. Use `A^-1` as
the PCG preconditioner. Stop at the first of:

- configured maximum iterations;
- relative residual tolerance;
- non-finite arithmetic;
- non-positive curvature.

On non-positive curvature, retain the current valid iterate; on failure before
one valid iterate, fall back to `-(1/lambda) A^-1 g`. Apply a configurable
Sobolev-norm trust-region radius by scaling `d` if `d.T @ A(d)` exceeds it.

Do not pass `d` through per-cell Adam moments. Optional momentum may be added
later using one scalar learning-rate/normalization per lattice.

## Damping and acceptance

For the first implementation, use a fixed `lambda` and step scale to isolate
operator correctness. Then add Levenberg-Marquardt adaptation using

```text
predicted reduction = -(g.T @ d + 0.5 d.T @ G(d))
rho = actual reduction / predicted reduction.
```

Evaluate actual reduction on the same frozen batch and random state. Reject the
step when reduction is non-finite or `rho <= 0`; increase `lambda` on rejection
or poor agreement and decrease it only after good agreement. A rejected trial
must restore parameters exactly and must not mutate optimizer state.

## Integration

Add a separate module, tentatively `sobolev_gauss_newton.py`, and configuration
keys for enablement, Sobolev length scales in scroll-voxel units, damping, PCG
limit/tolerance, trust radius, and LM thresholds. Backfill all keys so old
checkpoints select the existing AdamW behavior.

Insert the step after DDP gradient averaging and non-finite validation. Ensure
all ranks use identical operators, masks, stopping decisions, and accepted
steps. Save damping and any persistent momentum state in checkpoints; PCG
workspace and curvature graphs are transient.

Log per lattice/stage: `lambda`, PCG iterations, initial/final residual,
termination reason, Sobolev step norm, Euclidean component RMS, predicted and
actual reduction, `rho`, and acceptance.

## Verification

Add focused tests covering:

- `A` symmetry, positive definiteness, boundary conditions, and constrained
  cells;
- cylindrical volume weighting, unequal-ring adjoints, periodicity, and basis
  transport;
- `solve_A(apply_A(x)) ~= x` on small lattices;
- matrix-free GGN products versus explicit Jacobians;
- PCG versus a dense direct solve;
- the large-damping limit versus a Sobolev-gradient step;
- exact rollback after rejected LM trials;
- checkpoint backfill/resume and deterministic distributed behavior.

Run the focused suite with:

```bash
AGENTS_AGENT_MODE=1 pytest -q spiral-fitting/tests/test_sobolev_gauss_newton.py
AGENTS_AGENT_MODE=1 pytest -q spiral-fitting/tests/test_flow_grad_smoothing.py \
  spiral-fitting/tests/test_checkpoint_load.py
```

Before enabling it for production, compare AdamW, Gaussian-smoothed AdamW,
Sobolev gradient descent, and Sobolev-damped GGN on the same fixed batches and
full-scroll workload. Report loss/quality metrics, update roughness, invalid
Jacobian/folding metrics, peak VRAM, wall time per accepted step, total time to
a fixed loss, PCG iteration statistics, and sensitivity to lattice resolution.

## Initial acceptance criteria

- Existing behavior and checkpoints are unchanged when disabled.
- Tiny-problem steps agree with dense reference calculations within documented
  floating-point tolerances.
- Constrained cells remain bitwise unchanged.
- The large-`lambda` limit demonstrably matches the implemented Sobolev step.
- No claim of quality or speed improvement is made until the controlled
  before/after workload comparison is complete.
