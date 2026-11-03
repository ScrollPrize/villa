import assert from "node:assert/strict";
import { test } from "node:test";
import { demandScale, resolveScaleIndex, scalePickPixels } from "../src/volume/levels.js";

test("a level is picked for the view width in voxels, at least 32 and at most the pane's width or 640", () => {
  assert.equal(scalePickPixels(192, 192), 192);
  assert.equal(scalePickPixels(591, 192), 192);
  assert.equal(scalePickPixels(1536, 640), 640);
  assert.equal(scalePickPixels(1536, 4096), 640);
  assert.equal(scalePickPixels(10.9, 640), 32);
});

test("manual finest and coarsest levels clamp the automatic level, and equal bounds pin it", () => {
  assert.equal(resolveScaleIndex(2, null, null, 6), 2);
  assert.equal(resolveScaleIndex(2, undefined, undefined, 6), 2);
  assert.equal(resolveScaleIndex(2, Number.NaN, Number.NaN, 6), 2);
  assert.equal(resolveScaleIndex(2, 4, null, 6), 4);
  assert.equal(resolveScaleIndex(4, 2, null, 6), 4);
  assert.equal(resolveScaleIndex(4, null, 2, 6), 2);
  assert.equal(resolveScaleIndex(0, 2, 2, 6), 2);
  assert.equal(resolveScaleIndex(5, 2, 2, 6), 2);
  assert.equal(resolveScaleIndex(2, 99, null, 6), 5);
  assert.throws(() => resolveScaleIndex(2, 4, 2, 6), /Finest scale index/);
});

test("a demand scale keeps the path, shape, chunks, downsample and offset and drops the rest", () => {
  const scale = {
    path: "1",
    shape_ZYX: [8, 8, 8],
    chunks_ZYX: [4, 4, 4],
    downsample_ZYX: [2, 2, 2],
    offset_ZYX: [0.5, 0.5, 0.5],
    numericType: { dtype: "|u1" },
    store: {},
  };
  assert.deepEqual(demandScale(scale), {
    path: "1",
    shape_ZYX: [8, 8, 8],
    chunks_ZYX: [4, 4, 4],
    downsample_ZYX: [2, 2, 2],
    offset_ZYX: [0.5, 0.5, 0.5],
  });
});
