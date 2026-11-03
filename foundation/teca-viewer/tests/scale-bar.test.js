import assert from "node:assert/strict";
import { test } from "node:test";
import { IDENTITY_ROTATION, quatFromAxisAngle } from "../src/math.js";
import { formatScaleLengthUm, scaleBarMeasurement } from "../src/ui/scale-bar.js";

const MICROMETERS = ["micrometer", "micrometer", "micrometer"];

test("a bar aimed at 22% of the view rounds down to 1, 2 or 5 times a power of ten", () => {
  for (const [unitsPerVoxel, length] of [
    [1, 20],
    [3, 50],
    [0.0001, 0.002],
    [500, 10000],
  ]) {
    const { lengthUnits } = scaleBarMeasurement(
      "xy",
      IDENTITY_ROTATION,
      [1, 1, unitsPerVoxel],
      MICROMETERS,
      100,
      200,
      1,
    );
    assert.equal(lengthUnits, length, `${unitsPerVoxel} per voxel`);
  }
});

test("an XY pane turned 45 degrees about Y over 4 by 2 by 2 micrometer voxels measures √10 per voxel", () => {
  const rotation = quatFromAxisAngle([0, 1, 0], 45);
  const measured = scaleBarMeasurement("xy", rotation, [4, 2, 2], MICROMETERS, 100, 200, 1);
  assert.ok(Math.abs(measured.unitsPerDisplayedVoxel - Math.sqrt(10)) < 1e-12);
  assert.equal(measured.unit, "micrometer");
  assert.equal(measured.lengthUnits, 50);
  assert.ok(Math.abs(measured.widthPixels - (50 / Math.sqrt(10)) * 2) < 1e-12);
});

test("mixed units along the pane's width measure voxels, and the 3D pane has no bar", () => {
  const rotation = quatFromAxisAngle([0, 1, 0], 45);
  const mixed = scaleBarMeasurement("xy", rotation, [4, 2, 2], ["nanometer", "micrometer", "micrometer"], 100, 200, 1);
  assert.equal(mixed.unit, null);
  assert.equal(mixed.unitsPerDisplayedVoxel, 1);
  assert.equal(mixed.lengthUnits, 20);
  assert.equal(scaleBarMeasurement("3d", IDENTITY_ROTATION, [1, 1, 1], MICROMETERS, 100, 200, 1), null);
});

test("a length prints with six significant digits", () => {
  assert.equal(formatScaleLengthUm(1 / 3), "0.333333 um");
});
