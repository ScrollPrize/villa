import assert from "node:assert/strict";
import { test } from "node:test";
import {
  compileSurfaceReliefMask,
  defaultSurfaceReliefMask,
  removeNearestSurfaceReliefControl,
  setSurfaceReliefControl,
  updateSurfaceReliefControl,
  validateSurfaceReliefMask,
} from "../src/surface/relief-mask.js";

const flat = (v, depth) => ({
  v,
  points: [
    { u: 0, depth },
    { u: 1, depth },
  ],
});

test("a top control of depth 0.4 at the center compiles to 0.4 there and 0.2 halfway to the edge", () => {
  const mask = setSurfaceReliefControl(defaultSurfaceReliefMask(), "top", 0.5, 0.5, 0.4);
  assert.equal(mask.enabled, true);
  const texture_HW4 = compileSurfaceReliefMask(mask, 3);
  assert.equal(texture_HW4.length, 36);
  assert.ok(Math.abs(texture_HW4[(1 * 3 + 1) * 4] - 0.4) < 1e-6);
  assert.ok(Math.abs(compileSurfaceReliefMask(mask, 5)[(2 * 5 + 1) * 4] - 0.2) < 1e-6);
  const removed = removeNearestSurfaceReliefControl(mask, "top", 0.5, 0.5);
  assert.equal(compileSurfaceReliefMask(removed, 3)[(1 * 3 + 1) * 4], 0);
});

test("a top boundary at depth 0.8 below a bottom at 0.2 is rejected as crossing", () => {
  assert.throws(
    () =>
      validateSurfaceReliefMask({
        enabled: true,
        top: [flat(0, 0.8), flat(1, 0.8)],
        bottom: [flat(0, 0.2), flat(1, 0.2)],
      }),
    /must not cross/,
  );
});

test("moving a control changes its ridge, a boundary ridge keeps v 0 and u 0, and a move below the bottom is rejected", () => {
  let mask = setSurfaceReliefControl(defaultSurfaceReliefMask(), "top", 0.5, 0.5, 0.4);
  mask = setSurfaceReliefControl(mask, "bottom", 0.5, 0.5, 0.7);
  mask = updateSurfaceReliefControl(mask, "top", 1, 1, { u: 0.6, v: 0.55, depth: 0.5 });
  assert.deepEqual(mask.top[1], {
    v: 0.55,
    points: [
      { u: 0, depth: 0 },
      { u: 0.6, depth: 0.5 },
      { u: 1, depth: 0 },
    ],
  });
  const boundary = updateSurfaceReliefControl(mask, "top", 0, 0, { u: 0.4, v: 0.4 });
  assert.equal(boundary.top[0].v, 0);
  assert.equal(boundary.top[0].points[0].u, 0);
  assert.throws(() => updateSurfaceReliefControl(mask, "top", 1, 1, { depth: 0.9 }), /must not cross/);
});
