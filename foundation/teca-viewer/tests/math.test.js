import assert from "node:assert/strict";
import { test } from "node:test";
import {
  chordSteps,
  quatAlignVectors,
  quatFromAxisAngle,
  quatToEulerXYZDegrees,
  rotateVectorByQuat,
  rotationAngleDegrees,
} from "../src/math.js";

const near = (actual, expected, tolerance = 1e-12) => {
  for (const [index, value] of expected.entries()) {
    assert.ok(Math.abs(actual[index] - value) < tolerance, `${actual} ≠ ${expected}`);
  }
};

test("a quarter turn about Y takes +Z to +X", () => {
  near(rotateVectorByQuat([0, 0, 1], quatFromAxisAngle([0, 1, 0], 90)), [1, 0, 0]);
});

test("turns of 30 about X, -25 about Y and 40 about Z read back as XYZ degrees", () => {
  near(quatToEulerXYZDegrees(quatFromAxisAngle([1, 0, 0], 30)), [30, 0, 0], 1e-10);
  near(quatToEulerXYZDegrees(quatFromAxisAngle([0, 1, 0], -25)), [0, -25, 0], 1e-10);
  near(quatToEulerXYZDegrees(quatFromAxisAngle([0, 0, 1], 40)), [0, 0, 40], 1e-10);
  assert.throws(() => quatToEulerXYZDegrees([0, 0, 0, 0]), /nonzero/);
});

test("aligning +Z to a vector takes +Z onto it, and aligning it to -Z turns about the fallback axis", () => {
  const length = Math.hypot(2, -3, 4);
  const target_XYZ = [2 / length, -3 / length, 4 / length];
  near(rotateVectorByQuat([0, 0, 1], quatAlignVectors([0, 0, 1], target_XYZ)), target_XYZ);
  const opposite = quatAlignVectors([0, 0, 1], [0, 0, -1], [0, 1, 0]);
  assert.deepEqual(rotateVectorByQuat([0, 1, 0], opposite), [0, 1, 0]);
  near(rotateVectorByQuat([0, 0, 1], opposite), [0, 0, -1]);
  assert.throws(() => quatAlignVectors([0, 0, 1], [0, 0, -1], [0, 0, 2]), /fallback axis not parallel/);
  assert.throws(() => quatAlignVectors([0, 0, 0], [1, 0, 0]), /finite nonzero/);
});

test("a 30-degree turn about Y has a rotation angle of 30 degrees", () => {
  assert.ok(Math.abs(rotationAngleDegrees(quatFromAxisAngle([0, 1, 0], 30)) - 30) < 1e-9);
});

test("a quarter circle of radius 100 is cut into 8 chords within a voxel of it, and a line into 1", () => {
  const circle = (t) => [0, 100 * Math.sin((t * Math.PI) / 2), 100 * Math.cos((t * Math.PI) / 2)];
  const steps = chordSteps(circle, circle(0), circle(1), 1, 1);
  assert.deepEqual(
    steps.map(({ t }) => t),
    [1, 2, 3, 4, 5, 6, 7, 8].map((eighth) => eighth / 8),
  );
  for (const { t, point_ZYX } of steps) near(point_ZYX, circle(t));
  // Cut into 4, a chord's middle would be 100 (1 - cos(π / 16)) = 1.92 off the circle; into 8, 0.48.
  const line = (t) => [t, 2 * t, 3 * t];
  assert.deepEqual(chordSteps(line, line(0), line(1), 1, 1), [{ t: 1, point_ZYX: [1, 2, 3] }]);
});

test("a curve 40 cells long whose heights at five equal steps are 20, 30, 20, 10 and 20 is cut at its bends", () => {
  const heights = [20, 30, 20, 10, 20];
  const bent = (t) => {
    const piece = Math.min(3, Math.floor(t * 4));
    return [heights[piece] + (heights[piece + 1] - heights[piece]) * (t * 4 - piece), 0, 40 * t];
  };
  assert.deepEqual(
    chordSteps(bent, bent(0), bent(1), 1, 40),
    [0.25, 0.75, 1].map((t) => ({ t, point_ZYX: bent(t) })),
  );
});

test("a chord whose middle has no point is not cut", () => {
  const holed = (t) => (t === 0.5 ? null : [0, 100 * Math.sin(t * Math.PI), 0]);
  assert.deepEqual(chordSteps(holed, [0, 0, 0], [0, 0, 0], 1, 1), [{ t: 1, point_ZYX: [0, 0, 0] }]);
});
