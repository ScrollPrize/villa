import assert from "node:assert/strict";
import { test } from "node:test";
import { surfaceFrameAt } from "../src/surface/frame.js";
import { surfaceNormalProjection } from "../src/surface/normal-projection.js";

test("a point 3 voxels along the normal from UV [1.5, 2.25] projects back to that UV at depth 3", () => {
  const grid_HW = [4, 4];
  const pixel = ([u, v]) => [2 * v, v, 3 * u];
  const frame = surfaceFrameAt([1.5, 2.25], 1, grid_HW, pixel);
  const point_ZYX = frame.point_ZYX.map((value, axis) => value + 3 * frame.normal_ZYX[axis]);
  const projection = surfaceNormalProjection(point_ZYX, { surface_UV: [1, 2], signedDepth: 0 }, grid_HW);
  let step = projection.next();
  while (!step.done) step = projection.next(surfaceFrameAt(step.value, 1, grid_HW, pixel));
  const { surface_UV, signedDepth, distanceSquared } = step.value;
  assert.ok(Math.abs(surface_UV[0] - 1.5) < 1e-6);
  assert.ok(Math.abs(surface_UV[1] - 2.25) < 1e-6);
  assert.ok(Math.abs(signedDepth - 3) < 1e-6);
  assert.ok(Math.abs(distanceSquared - 9) < 1e-5);
});
