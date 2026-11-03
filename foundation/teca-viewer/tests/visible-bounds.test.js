import assert from "node:assert/strict";
import { test } from "node:test";
import { packSurfaceDemandBounds, planSurfaceDemand } from "../src/surface/planning.js";
import { surfaceStripBounds } from "../src/surface/strip-atlas.js";
import { visibleSurfaceBounds } from "../src/surface/visible-bounds.js";

const IDENTITY = { scale_ZYX: [1, 1, 1], offset_ZYX: [0, 0, 0] };
const SCALE = {
  shape_ZYX: [4096, 4096, 4096],
  chunks_ZYX: [32, 32, 32],
  downsample_ZYX: [1, 1, 1],
  offset_ZYX: [0, 0, 0],
};

function geometry(grid_HW, stripRows, pointAt_ZYX) {
  const [height, width] = grid_HW;
  const stripEntries = [];
  for (let stripIndex = 0; stripIndex < Math.ceil(height / stripRows); stripIndex++) {
    const data_HWRGBA = new Float32Array(width * stripRows * 4);
    for (let v = 0; v < stripRows && stripIndex * stripRows + v < height; v++)
      for (let u = 0; u < width; u++)
        data_HWRGBA.set([...pointAt_ZYX(u, stripIndex * stripRows + v), 1], (v * width + u) * 4);
    stripEntries.push({
      stripIndex,
      data_HWRGBA,
      boundsGrid_HW: [Math.ceil(stripRows / 8), Math.ceil(width / 8)],
      bounds_B6: surfaceStripBounds(data_HWRGBA, width, stripRows),
    });
  }
  return { stripEntries, stripRows, grid_HW };
}

test("a window of 4096 cells is bounded per cell, and one of 65 by 64 cells per 8×8 block", () => {
  const source = geometry([66, 66], 16, (u, v) => [10, v, u]);
  const cells = [...visibleSurfaceBounds({ ...source, center_UV: [32, 32], visibleSpan_UV: [64, 64] })];
  assert.equal(cells.length, 4096);
  assert.deepEqual([...cells[0].bounds_ZYX6], [10, 0, 0, 10, 1, 1]);
  assert.deepEqual([...cells[4095].bounds_ZYX6], [10, 63, 63, 10, 64, 64]);
  const blocks = [...visibleSurfaceBounds({ ...source, center_UV: [32.5, 32], visibleSpan_UV: [65, 64] })];
  assert.equal(blocks.length, 81);
  assert.deepEqual([...blocks[0].bounds_ZYX6], [10, 0, 0, 10, 8, 8]);
});

test("a 512 by 16 window gives 195 block bounds, the one across the strip seam reaching Z 300", () => {
  const source = geometry([17, 513], 16, (u, v) => [v === 16 ? 300 : 10, v, u]);
  const records = [...visibleSurfaceBounds({ ...source, center_UV: [256, 8], visibleSpan_UV: [512, 16] })];
  assert.equal(records.length, 195);
  assert.deepEqual([...records[0].bounds_ZYX6], [10, 0, 0, 10, 8, 8]);
  assert.deepEqual([...records[65].bounds_ZYX6], [10, 8, 0, 300, 16, 8]);
});

test("a diagonal sheet across 17 chunks of Z and X requires the 49 chunks of its cells, not the 161 of its two blocks", () => {
  const source = geometry([9, 17], 16, (u, v) => [u * 32 + 8, v + 8, u * 32 + 8]);
  const plan = (bounds) =>
    planSurfaceDemand({
      options: { surfaceBounds_B7: packSurfaceDemandBounds(bounds), registration: IDENTITY, normalRadius: 0 },
      scales: [SCALE],
      capacities: [4096],
    })[0];
  const result = plan(visibleSurfaceBounds({ ...source, center_UV: [8, 4], visibleSpan_UV: [16, 8] }));
  assert.equal(result.complete, true);
  assert.equal(result.requiredCount, 49);
  assert.deepEqual(
    result.addresses.map(({ chunk_ZYX }) => chunk_ZYX.join("/")).sort(),
    Array.from({ length: 17 }, (_, u) => [`${u}/0/${u}`, ...(u < 16 ? [`${u}/0/${u + 1}`, `${u + 1}/0/${u}`] : [])])
      .flat()
      .sort(),
  );
  const { bounds_B6 } = source.stripEntries[0];
  const blocks = [0, 1].map((block) => ({
    bounds_ZYX6: bounds_B6.subarray(block * 6, block * 6 + 6),
    surfacePriority: 0,
  }));
  assert.equal(plan(blocks).requiredCount, 161);
});

test("a grid shape that is not two positive integers throws", () => {
  const source = geometry([8, 8], 8, () => [1, 2, 3]);
  for (const grid_HW of [null, [], [1], [0, 1], [1.5, 1]]) {
    assert.throws(
      () => [...visibleSurfaceBounds({ ...source, grid_HW, center_UV: [4, 4], visibleSpan_UV: [8, 8] })],
      /grid dimensions are malformed/,
    );
  }
});
