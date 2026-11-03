import assert from "node:assert/strict";
import { test } from "node:test";
import { quatFromAxisAngle } from "../src/math.js";
import { rotatedPlaneBasis } from "../src/planes.js";
import { PlanePrefetch } from "../src/render/plane-prefetch.js";

const SCALE = {
  chunks_ZYX: [32, 64, 128],
  downsample_ZYX: [2, 1, 1],
  offset_ZYX: [0, 0, 0],
  shape_ZYX: [2048, 4096, 4096],
};

function view(center_ZYX = [2048, 2048, 2048], degrees = 25, width = 1024) {
  const rotation = quatFromAxisAngle([1, 0, 0], degrees);
  return {
    plane: "xy",
    center_ZYX,
    rotation,
    basis: rotatedPlaneBasis("xy", rotation),
    viewWidthVoxels: width,
    viewAspect: 1,
  };
}

test("a pan leads by at most one chunk along its direction, and a reversal leads back", () => {
  const prefetch = new PlanePrefetch();
  prefetch.observe("a", [view()], 0);
  assert.equal(prefetch.prediction("xy", SCALE, 120, 0), null);
  prefetch.observe("a", [view([2100, 2074, 2061])], 100);
  assert.deepEqual(prefetch.prediction("xy", SCALE, 1000, 100).center_ZYX, [2164, 2106, 2077]);
  prefetch.observe("a", [view([2080, 2064, 2056])], 200);
  assert.deepEqual(prefetch.prediction("xy", SCALE, 100, 200).center_ZYX, [2050, 2049, 2048.5]);
});

test("a turn from 20 to 22 degrees in 100 ms with 100 ms reads leads to 25 degrees", () => {
  const prefetch = new PlanePrefetch();
  prefetch.observe("a", [view(undefined, 20)], 0);
  prefetch.observe("a", [view(undefined, 22)], 100);
  const { basis } = prefetch.prediction("xy", SCALE, 100, 100);
  const expected = view(undefined, 25).basis;
  for (const key of Object.keys(expected)) {
    for (const [axis, value] of expected[key].entries()) assert.ok(Math.abs(value - basis[key][axis]) < 1e-12, key);
  }
});

test("a fast turn leads by no more than moves the pane's corner one chunk", () => {
  const prefetch = new PlanePrefetch();
  prefetch.observe("a", [view(undefined, 22)], 100);
  prefetch.observe("a", [view(undefined, 90)], 200);
  const { basis } = prefetch.prediction("xy", SCALE, 1000, 200);
  const normal_XYZ = view(undefined, 90).basis.normal_XYZ;
  const angle = Math.acos(normal_XYZ.reduce((sum, value, axis) => sum + value * basis.normal_XYZ[axis], 0));
  assert.ok(angle > 0 && angle <= 64 / (1024 / Math.sqrt(2)) + 1e-12, `${angle}`);
});

test("a prediction ends after 250 ms at rest, on a source change, a hidden pane or a zoom", () => {
  for (const change of ["idle", "source", "hidden", "zoom"]) {
    const prefetch = new PlanePrefetch();
    prefetch.observe("a", [view()], 0);
    prefetch.observe("a", [view([2080, 2048, 2048])], 100);
    assert.notEqual(prefetch.prediction("xy", SCALE, null, 100), null);
    if (change === "source") prefetch.observe("b", [view()], 200);
    if (change === "hidden") prefetch.observe("a", [], 200);
    if (change === "zoom") prefetch.observe("a", [view(undefined, 25, 512)], 200);
    assert.equal(prefetch.prediction("xy", SCALE, null, change === "idle" ? 351 : 200), null, change);
  }
});
