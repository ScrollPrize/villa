import assert from "node:assert/strict";
import { test } from "node:test";
import { packSurfaceDemandBounds, planSurfaceDemand, visibleSurfaceStrips } from "../src/surface/planning.js";
import { surfaceStripBounds } from "../src/surface/strip-atlas.js";
import { visibleSurfaceBounds } from "../src/surface/visible-bounds.js";

const IDENTITY = { scale_ZYX: [1, 1, 1], offset_ZYX: [0, 0, 0] };
const cube = (size, chunk) => ({
  shape_ZYX: [size, size, size],
  chunks_ZYX: [chunk, chunk, chunk],
  downsample_ZYX: [1, 1, 1],
  offset_ZYX: [0, 0, 0],
});
const plan = (bounds, scale, capacity, normalRadius = 0) =>
  planSurfaceDemand({
    options: { surfaceBounds_B7: packSurfaceDemandBounds(bounds), registration: IDENTITY, normalRadius },
    scales: [scale],
    capacities: [capacity],
  })[0];
const keys = (addresses) => addresses.map(({ chunk_ZYX }) => chunk_ZYX.join("/"));

test("a view centered on V 130 spanning 180 rows needs strips 2, 3, 1 and 0 of 64 rows, nearest first", () => {
  assert.deepEqual(
    visibleSurfaceStrips({ center_UV: [100, 130], visibleSpan_UV: [80, 180], grid_HW: [512, 256], stripRows: 64 }),
    [2, 3, 1, 0],
  );
});

test("a two-row strip at Z 127.25 with a normal radius of 1 needs Z chunks 0 and 1", () => {
  const data_HWRGBA = new Float32Array(2 * 256 * 4);
  for (let y = 0; y < 2; y++) for (let x = 0; x < 256; x++) data_HWRGBA.set([127.25, 10 + y, x, 1], (y * 256 + x) * 4);
  data_HWRGBA[3] = 0;
  const bounds_B6 = surfaceStripBounds(data_HWRGBA, 256, 2);
  const surfaceBounds = visibleSurfaceBounds({
    stripEntries: [{ stripIndex: 0, bounds_B6, boundsGrid_HW: [1, 32], data_HWRGBA }],
    center_UV: [64, 0.5],
    visibleSpan_UV: [126, 2],
    stripRows: 2,
    grid_HW: [2, 256],
  });
  const scale = {
    shape_ZYX: [256, 64, 256],
    chunks_ZYX: [128, 32, 128],
    downsample_ZYX: [1, 1, 1],
    offset_ZYX: [0, 0, 0],
  };
  const found = new Set(keys(plan(surfaceBounds, scale, 16, 1).addresses));
  assert.ok(found.has("0/0/0"));
  assert.ok(found.has("1/0/0"));
});

test("bounds over a 200³-chunk volume at capacity 3 give the 3 central chunks and a lower bound of 260", () => {
  const bounds = [{ bounds_ZYX6: [0, 0, 0, 25_599, 25_599, 25_599], surfacePriority: 0 }];
  const result = plan(bounds, cube(25_600, 128), 3);
  assert.equal(result.addresses.length, 3);
  assert.equal(result.complete, false);
  assert.equal(result.requiredCount, null);
  assert.equal(result.requiredCountLowerBound, 260);
  assert.ok(result.addresses.every(({ chunk_ZYX }) => chunk_ZYX.every((chunk) => chunk === 99 || chunk === 100)));
});

test("bounds of 0..255 in 128³ chunks require exactly 27 chunks, over a capacity of 3", () => {
  const result = plan([{ bounds_ZYX6: [0, 0, 0, 255, 255, 255], surfacePriority: 0 }], cube(32768, 128), 3);
  assert.equal(result.requiredCount, 27);
  assert.equal(result.requiredCountLowerBound, 27);
});

test("bounds of two chunks at capacity 4 are complete with 2 addresses", () => {
  const scale = {
    shape_ZYX: [128, 128, 256],
    chunks_ZYX: [128, 128, 128],
    downsample_ZYX: [1, 1, 1],
    offset_ZYX: [0, 0, 0],
  };
  const result = plan([{ bounds_ZYX6: [0, 0, 0, 127, 127, 255], surfacePriority: 0 }], scale, 4);
  assert.equal(result.complete, true);
  assert.equal(result.requiredCount, 2);
  assert.equal(result.requiredCountLowerBound, 2);
  assert.equal(result.addresses.length, 2);
});

test("three bounds of one chunk at priorities 1, 0 and 1 give that chunk once at priority 0", () => {
  const bounds = [1, 0, 1].map((surfacePriority) => ({ bounds_ZYX6: [0, 0, 0, 31, 31, 31], surfacePriority }));
  const result = plan(bounds, cube(32, 32), 1);
  assert.equal(result.complete, true);
  assert.equal(result.requiredCount, 1);
  assert.deepEqual(result.addresses, [{ chunk_ZYX: [0, 0, 0], priority: 0 }]);
});

test("chunks tied in priority at a capacity of 8 are chosen in ZYX order: 5/0/0 to 5/0/6, then 5/5/5", () => {
  const priority = Math.hypot(0.5, 0.5, 0.5) / 1024;
  const single = (chunk_ZYX) => ({
    bounds_ZYX6: [...chunk_ZYX, ...chunk_ZYX].map((value) => value * 128),
    surfacePriority: priority,
  });
  const bounds = [
    ...Array.from({ length: 7 }, (_, x) => single([5, 0, x])),
    single([6, 6, 5]),
    { bounds_ZYX6: [640, 640, 640, 768, 768, 768], surfacePriority: 0 },
  ];
  const result = plan(bounds, cube(32768, 128), 8);
  assert.deepEqual(keys(result.addresses), [...Array.from({ length: 7 }, (_, x) => `5/0/${x}`), "5/5/5"]);
  assert.equal(result.complete, false);
  assert.equal(result.requiredCount, 15);
});

test("a capacity of 0, 1.5 or 65537 throws, and no bounds or a bound with NaN or ±Infinity requires 0 chunks", () => {
  for (const capacity of [0, 1.5, 65537]) {
    assert.throws(() => plan([], cube(32768, 128), capacity), /capacity must be inside 1\.\.65536/);
  }
  assert.equal(plan([], cube(32768, 128), 96).requiredCount, 0);
  for (const value of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
    for (let axis = 0; axis < 6; axis++) {
      const bounds_ZYX6 = [0, 0, 0, 100, 100, 100];
      bounds_ZYX6[axis] = value;
      assert.equal(plan([{ bounds_ZYX6, surfacePriority: 0 }], cube(32768, 128), 96).requiredCount, 0);
    }
  }
});

test("5000 packed bounds are 35000 float64 values that keep their coordinates and priority exactly", () => {
  const bounds = Array.from({ length: 5000 }, (_, index) => ({
    bounds_ZYX6: [0.1 + index, 1e-9, 1e15 + 1, 2, 3, 4],
    surfacePriority: 1 / 3,
  }));
  const packed_B7 = packSurfaceDemandBounds(bounds);
  assert.equal(packed_B7.length, 35000);
  assert.equal(packed_B7[0], 0.1);
  assert.equal(packed_B7[2], 1e15 + 1);
  assert.equal(packed_B7[6], 1 / 3);
  assert.equal(packSurfaceDemandBounds([]).length, 0);
});
