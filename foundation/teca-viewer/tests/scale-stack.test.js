import assert from "node:assert/strict";
import { test } from "node:test";
import { surfaceDisplayScaleIndex, surfaceVolumeScaleStack } from "../src/surface/scale-stack.js";

const pyramid = (count) =>
  Array.from({ length: count }, (_, index) => ({
    path: String(index),
    chunks_ZYX: [64, 64, 64],
    downsample_ZYX: [2 ** index, 2 ** index, 2 ** index],
  }));
const paths = (scales) => scales.map((scale) => scale.path);

test("a Surface reads its display level and the coarsest level, within the finest and coarsest bounds", () => {
  const scales = pyramid(7);
  assert.deepEqual(paths(surfaceVolumeScaleStack(scales, null, null)), ["0", "6"]);
  assert.deepEqual(paths(surfaceVolumeScaleStack(scales, 2, null)), ["2", "6"]);
  assert.deepEqual(paths(surfaceVolumeScaleStack(scales, null, 2)), ["0", "2"]);
  assert.deepEqual(paths(surfaceVolumeScaleStack(scales, 3, 3)), ["3"]);
  assert.deepEqual(paths(surfaceVolumeScaleStack(scales, null, null, 3)), ["3", "6"]);
  assert.throws(() => surfaceVolumeScaleStack(scales, 4, 3), /finest level/);
  assert.throws(() => surfaceVolumeScaleStack(scales, 7, null), /inside 0\.\.6/);
});

test("the display level is the coarsest whose downsample is within 1.35 times the full-resolution voxels per pixel", () => {
  const scales = pyramid(6);
  assert.equal(surfaceDisplayScaleIndex(scales, [286.68, 334.58], [0.05, 0.05], 790, null, null), 3);
  assert.equal(surfaceDisplayScaleIndex(scales, [64, 74], [0.05, 0.05], 790, null, null), 1);
  assert.equal(surfaceDisplayScaleIndex(scales, [286.68, 334.58], [0.05, 0.05], 790, 4, null), 4);
});
