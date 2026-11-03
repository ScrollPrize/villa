import { SURFACE_NORMAL_OFFSET_LIMIT } from "../config.js";

const STEP_LIMIT = 24;
const SEED_REACH = 64;
const STEP_REACH = 8;
const DIFFERENCE_STEP = 0.01;
const CONVERGED_DISTANCE = 1e-4;

export function* surfaceNormalProjection(point_ZYX, seed, grid_HW) {
  const surface_UV = [...seed.surface_UV];
  let depth = seed.signedDepth;
  for (let iteration = 0; iteration < STEP_LIMIT; iteration++) {
    const frame = yield surface_UV;
    if (frame === null) return null;
    const position_ZYX = frame.point_ZYX.map((value, axis) => value + depth * frame.normal_ZYX[axis]);
    const residual_ZYX = point_ZYX.map((value, axis) => value - position_ZYX[axis]);
    if (Math.hypot(...residual_ZYX) <= CONVERGED_DISTANCE) {
      return { surface_UV, signedDepth: depth, distanceSquared: depth ** 2 };
    }
    const jacobian_33 = [];
    for (let axis = 0; axis < 2; axis++) {
      const next_UV = [...surface_UV];
      const step = next_UV[axis] + DIFFERENCE_STEP <= grid_HW[1 - axis] - 1 ? DIFFERENCE_STEP : -DIFFERENCE_STEP;
      next_UV[axis] += step;
      const sample = yield next_UV;
      if (sample === null) return null;
      jacobian_33.push(
        sample.point_ZYX.map((value, i) => (value + depth * sample.normal_ZYX[i] - position_ZYX[i]) / step),
      );
    }
    jacobian_33.push(frame.normal_ZYX);
    const determinant = (a, b, c) =>
      a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
    const divisor = determinant(...jacobian_33);
    if (Math.abs(divisor) < 1e-10) return null;
    const delta_UVD = [
      determinant(residual_ZYX, jacobian_33[1], jacobian_33[2]),
      determinant(jacobian_33[0], residual_ZYX, jacobian_33[2]),
      determinant(jacobian_33[0], jacobian_33[1], residual_ZYX),
    ].map((value) => value / divisor);
    const damping = Math.min(1, STEP_REACH / Math.max(Math.abs(delta_UVD[0]), Math.abs(delta_UVD[1]), 1e-12));
    surface_UV[0] += damping * delta_UVD[0];
    surface_UV[1] += damping * delta_UVD[1];
    depth += damping * delta_UVD[2];
    if (
      Math.abs(depth) > SURFACE_NORMAL_OFFSET_LIMIT ||
      surface_UV.some((value, axis) => Math.abs(value - seed.surface_UV[axis]) > SEED_REACH)
    )
      return null;
  }
  return null;
}
