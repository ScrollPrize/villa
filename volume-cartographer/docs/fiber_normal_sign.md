# Native CPU normal-sign solve

`vc/fiber_tracer/FiberNormalSign.hpp` decides one normal sign per fiber from
perpendicular crossings, on the CPU, in the native layer. It is the same
objective as the `vc_wgraph` sign stage's wrapped IRLS (see
[the sign stage's weights](../../vesuvius/docs/wgraph.md#sign-stage-weights)),
without torch, CUDA or a GPU-manager grant, so the Fiber Map and other
interactive callers can run a sign solve in process.

## The problem

Every fiber carries one sign-free normal axis field. A perpendicular crossing
of two fibers witnesses whether their axes agree there (`dot > 0`) or oppose it
(`dot < 0`), never which way either points. Choosing a sign per fiber that
satisfies as much of that evidence as possible is a period-two L1 labelling,
relaxed to a continuous phase `x_v`: even phases are one class, odd the other,
and a crossing's residual is the wrapped difference

    r_e = wrap(x_b - x_a - t_e),   wrap(d) = fmod(d + 1, 2) - 1 in [-1, 1),

with `t_e = 1` for an opposing crossing and `0` for an agreeing one. With the
Huber-L1 cost `h(r)` (quadratic below `epsilon`, `|r| - epsilon/2` above) the
objective over the phases and the per-fiber keep variables `k in [0, 1]` is

    sum_e w_e k_a k_b h(r_e)                 crossing evidence
  + sum_v (p_v h(wrap(x_v - q_v)) + c_v)     per-fiber prior
  + sum_v z_v (1 - k_v)                      cost of rejecting a fiber.

A crossing's weight `w_e` is `|dot|^dotWeightPower` times the winding-distance
factor `min(1, 1/d^2)` (`InverseSquare`; an unmeasured distance counts as
within one winding), normalized to sum to one. The prior is the radial
agreement of each fiber scaled by `priorWeight`, or, for a fiber whose sign is
given, that sign at `fixedSignWeight`. `brokenPenalty` is the per-fiber cost of
rejection; `fixedKeep` holds every fiber retained and solves the signs alone.
The objective is not convex, so both backends descend it from the radial
initialization and report the value they reach, never a global optimum.

Signs come back in the inward convention, with the class score as
`signProbability`, matching the stored `normal_sign` column. The result also
carries the reference stage's metrics, including the strict internal metric
(disagreeing share of crossings whose axes are within 30 degrees and whose
winding distance is below one turn), so runs compare directly.

## Backends

`FiberSignBackend::Irls` reproduces the reference stage: each step reweights
the wrapped residuals into a quadratic, solves it with Jacobi-preconditioned
conjugate gradients (`cgIterations`, `cgTolerance`), takes one bounded keep
update and backtracks whenever the periodic objective fails to improve. Five
steps whose improvement stays below `tolerance` end the solve.

`FiberSignBackend::Ceres` states the same wrapped residuals as a Ceres problem
and lets a sparse trust-region solver move the phases: one residual per
crossing with an analytic Jacobian, under a Huber loss scaled per block so the
Ceres cost equals the pair and prior terms exactly (`ScaledLoss(Huber(eps),
w/eps)` on the unscaled residual). Keep is held during a solve; `keepRounds`
alternations of (phase solve, keep update) follow, and a round that raises the
periodic objective is rolled back. `iterations` caps the trust-region steps per
round.

Both run deterministically: loops are split at a fixed granularity, so the
solution does not depend on `threads`.

## Measured comparison

`vc_fiber_sign <config.json>` runs one backend over an extracted
constraint store (`database`), its radial priors (`priors`, a
`fiber-outward-priors` document) and optional `fixed_signs`, and writes the
summary, the per-fiber signs and the prepared events as JSON. Option keys
mirror the stage's (`backend`, `threads`, `epsilon`, `iterations`,
`cg_iterations`, `keep_rounds`, `prior_weight`, `broken_penalty`,
`fixed_sign_weight`, `winding_distance_weight`, `normal_dot_weight_power`,
`fixed_keep`).

Measured against the reference wrapped IRLS on identical inputs (the prepared
events agree except for the reference's float32 cast of `dot`, at most 3e-8),
on the PHerc0332 manual scaffold with fused normals (293 fibers, 1,893
perpendicular crossings, all with a measured winding distance). The reference
GPU run used one A100-SXM4-80GB, every CPU run an AMD EPYC 7J13 with eight
threads asked for.

The stage's default step cap of 100 stops short of convergence on this store
for every solver, so these runs raise it to 2,000. Solving the signs alone
(`fixed_keep`), all four reach the same 293 signs:

| solver | solve seconds | steps | objective |
| --- | --- | --- | --- |
| native Irls | 0.129 | 217 | 0.247549 |
| native Ceres | 0.297 | 1 round, 247 trust-region steps | 0.247547 |
| reference, CUDA | 15.114 | 220 | 0.247548 |
| reference, CPU torch | 3.350 | 229 | 0.247548 |

At the default cap of 100 steps the same runs take 0.058, 0.108, 6.740 and
1.463 seconds and stop at objective 0.2483, again with identical signs.

With rejection enabled (the stage default, `broken_penalty` 0.1), the native
IRLS path reproduces the reference exactly: 0.041 seconds over 66 steps to
objective 0.032944 and the same 293 signs as the reference's 4.581 seconds over
67 steps. Ceres over 100 keep rounds reaches 0.031155 in 0.480 seconds, but
gets there by rejecting 38% of the fibers where the others reject 30%, and 63
of its 293 signs differ. A lower value of this objective is not a better sign
field, which is why the scaffold workflow solves the signs with `fixedKeep`.

A note on both tunables this exposes. The Ceres backend used
`SPARSE_NORMAL_CHOLESKY` with SuiteSparse and honored its thread count, but a
scaffold is far too small to pay for threads: it took 0.108, 0.128 and 0.140
seconds at 1, 8 and 16 threads. The native IRLS backend splits its loops into
work units of 4,096 entries, so a problem of this size is one unit and runs
serially whatever `threads` says (0.058, 0.058 and 0.063 seconds at 1, 8 and
16): the parallel path only engages well above ten thousand fibers or
crossings, and the unit test covers it there.
