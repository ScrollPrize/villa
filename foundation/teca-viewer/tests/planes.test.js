import assert from "node:assert/strict";
import { test } from "node:test";
import { quatFromAxisAngle, quatMultiply } from "../src/math.js";
import { planeStepAxis, planeViewAxes, rotatedPlaneBasis } from "../src/planes.js";

test("xy shows ZYX axes 2 and 1 and steps 0, zy shows 0 and 1 and steps 2, xz shows 2 and 0 and steps 1", () => {
  assert.deepEqual(planeViewAxes("xy"), [2, 1]);
  assert.deepEqual(planeViewAxes("zy"), [0, 1]);
  assert.deepEqual(planeViewAxes("xz"), [2, 0]);
  assert.equal(planeStepAxis("xy"), 0);
  assert.equal(planeStepAxis("zy"), 2);
  assert.equal(planeStepAxis("xz"), 1);
  assert.throws(() => planeViewAxes("bad"), /Unknown plane/);
  assert.throws(() => planeStepAxis("bad"), /Unknown plane/);
  assert.throws(() => rotatedPlaneBasis("bad", [0, 0, 0, 1]), /Unknown plane/);
});

test("each pane basis under a rotation is three orthogonal unit vectors", () => {
  const rotation = quatMultiply(quatFromAxisAngle([1, 0, 0], 20), quatFromAxisAngle([0, 1, 0], 35));
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  for (const plane of ["xy", "zy", "xz"]) {
    const { normal_XYZ, right_XYZ, up_XYZ } = rotatedPlaneBasis(plane, rotation);
    for (const vector_XYZ of [normal_XYZ, right_XYZ, up_XYZ])
      assert.ok(Math.abs(Math.hypot(...vector_XYZ) - 1) < 1e-12);
    assert.ok(Math.abs(dot(normal_XYZ, right_XYZ)) < 1e-12);
    assert.ok(Math.abs(dot(normal_XYZ, up_XYZ)) < 1e-12);
    assert.ok(Math.abs(dot(right_XYZ, up_XYZ)) < 1e-12);
  }
});
