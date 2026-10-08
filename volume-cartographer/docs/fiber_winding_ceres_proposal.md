# Continuous Fiber Orientation and Winding Proposal

The fixed-scale subset of this design is implemented experimentally by
`vc_fiber_trace_chunk direction-ablation --winding-solver ceres`. The
phase/scale optimization and periodic integer-ladder residual below remain
deferred.

## Variables

The implemented parameterization for every fiber piece `i` is:

- `h_i` in `[0,1]`: horizontalness, with `1` for H and `0` for V.
- `a_i` in `[0,1]`: active confidence. `a_i=0` denotes Defect.
- `z_i` in `R`: continuous winding coordinate.

Optionally optimize the crop-global inter-ladder phase `phase` in `[0,0.5]`
and positive measurement scale `scale`. The first experiment should fix
`phase=0.5` and `scale=1`.

## Pair residuals

For parallel and perpendicular evidence weights `w_p` and `w_q`, use:

```text
r_parallel = sqrt(w_p) * a_i * a_j * (h_i - h_j)
r_perpendicular = sqrt(w_q) * a_i * a_j * (h_i + h_j - 1)
r_same_winding = sqrt(w_p) * a_i * a_j * (z_j - z_i)
r_measured_winding = sqrt(w_q) * a_i * a_j
    * ((z_j - z_i) / scale - signed_distance)
```

Ceres minimizes squared residuals, hence `sqrt(weight)`: squaring the residual
then contributes `weight * error^2`, rather than unintentionally squaring the
confidence weight.

The product `a_i*a_j` makes a Defect endpoint neutral for both orientation and
winding evidence. The current first experiment uses no robust loss.
Same-trace continuity uses a strong finite, non-robust factor.

## Unary and ladder residuals

Penalize inactive pieces and encourage active orientations to approach H/V:

```text
r_defect = sqrt(lambda_defect) * (1 - a_i)
r_binary = sqrt(lambda_binary * degree_i) * a_i * h_i * (1 - h_i)
```

The following integer-ladder residual remains deferred; current Ceres winding
coordinates are deliberately fractional:

```text
r_ladder = sqrt(lambda_ladder) * a_i
    * sin(pi * (z_i - phase * (1 - x_i) / 2))
```

At `x_i=+1`, the preferred ladder is integer `z`. At `x_i=-1`, it is
`integer+phase`. Defect pieces have no authoritative winding because the
ladder and pair residuals vanish with `a_i`.

## Gauge and initialization

Fix one reliable crop-central piece to H with winding zero, or apply equivalent
strong gauge residuals, to remove the global H/V swap and integer-offset
symmetries. Initialize `x` from the existing orientation estimate and `z` from
the existing continuous winding solve. Initialize `a` near one except for
known low-confidence pieces.

The periodic ladder and active/orientation coupling make the objective
non-convex. Use deterministic multi-starts or a homotopy that begins with weak
binary/ladder terms and strengthens them. Ceres produces a local MAP-like fit,
not discrete global optimality or calibrated posterior probabilities.

## Suggested first experiment

1. Fix phase and scale to `0.5` and `1.0`.
2. Optimize `x`, `a`, and `z` jointly with sparse normal Cholesky.
3. Report objective components, constraint residual quantiles, H/V/Defect
   counts, and deterministic repeatability.
4. Only after the fixed-calibration solve is useful, expose phase and log-scale
   as bounded global parameters.
