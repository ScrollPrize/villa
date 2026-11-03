import assert from "node:assert/strict";
import { test } from "node:test";
import { rayPick } from "../src/extensions/ct-box/pick.js";
import { boxLevel, boxRegion, copyBlock, displayBytes, regionChunks } from "../src/extensions/ct-box/region.js";
import { savedLook } from "../src/extensions/ct-box.js";

const LEVELS = [1, 2, 4].map((d) => ({
  shape_ZYX: [1000, 1000, 1000].map((size) => Math.ceil(size / d)),
  chunks_ZYX: [4, 4, 4],
  downsample_ZYX: [d, d, d],
  offset_ZYX: [0, 0, 0],
}));

test("a box of 512 finest voxels in 256 texels is read from the level of voxel 2, and of 2048 from none", () => {
  assert.equal(boxLevel(LEVELS, 256, 256), 0);
  assert.equal(boxLevel(LEVELS, 512, 256), 1);
  assert.throws(() => boxLevel(LEVELS, 2048, 256), { message: "No level of the CT holds 2048 voxels in 256 texels." });
});

test("a box takes the voxels whose centres lie within its side of the focus, cut to the level", () => {
  assert.deepEqual(boxRegion(LEVELS[0], [100.3, 100, 2], 8), {
    start_ZYX: [97, 96, 0],
    size_ZYX: [8, 8, 6],
    min_ZYX: [96.5, 95.5, -0.5],
    extent_ZYX: [8, 8, 6],
  });
  assert.deepEqual(boxRegion(LEVELS[1], [58.5, 58.5, 58.5], 16), {
    start_ZYX: [25, 25, 25],
    size_ZYX: [8, 8, 8],
    min_ZYX: [49.5, 49.5, 49.5],
    extent_ZYX: [16, 16, 16],
  });
});

test("the chunks of a region are those its first and last voxels lie in, and each copies its part in", () => {
  const scale = { ...LEVELS[0], shape_ZYX: [8, 8, 8] };
  const region = { start_ZYX: [3, 3, 3], size_ZYX: [2, 2, 2] };
  const chunks = regionChunks(scale, region, null);
  assert.deepEqual(chunks, [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
  ]);
  const box_ZYX = new Uint8Array(8);
  for (const chunk_ZYX of chunks) {
    const bytes = new Uint8Array(64).map((_, index) => {
      const [z, y, x] = [index >> 4, (index >> 2) & 3, index & 3].map((v, axis) => v + 4 * chunk_ZYX[axis]);
      return (100 * z + 10 * y + x) & 255;
    });
    copyBlock(
      box_ZYX,
      region,
      bytes,
      chunk_ZYX.map((c) => 4 * c),
      [4, 4, 4],
    );
  }
  assert.deepEqual(
    [...box_ZYX],
    [333, 334, 343, 344, 433, 434, 443, 444].map((value) => value & 255),
  );
});

test("a box moved one voxel along z reads the chunks of its new face and copies the rest from the box it held", () => {
  const scale = { ...LEVELS[0], shape_ZYX: [8, 8, 8] };
  const held = { start_ZYX: [3, 3, 3], size_ZYX: [2, 2, 2] };
  const region = { start_ZYX: [4, 3, 3], size_ZYX: [2, 2, 2] };
  assert.deepEqual(regionChunks(scale, region, held), [
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
  ]);
  const box_ZYX = new Uint8Array(8);
  copyBlock(box_ZYX, region, Uint8Array.of(1, 2, 3, 4, 5, 6, 7, 8), held.start_ZYX, held.size_ZYX);
  assert.deepEqual([...box_ZYX], [5, 6, 7, 8, 0, 0, 0, 0]);
  copyBlock(box_ZYX, { start_ZYX: [3, 3, 9], size_ZYX: [2, 2, 2] }, new Uint8Array(8), held.start_ZYX, held.size_ZYX);
  assert.deepEqual([...box_ZYX], [5, 6, 7, 8, 0, 0, 0, 0]);
});

test("a ray picks the brightest texel it meets, or where half its light is absorbed, and enters at the box's face", () => {
  const box_ZYX = new Uint8Array(64);
  box_ZYX[(3 * 4 + 1) * 4 + 2] = 200;
  const origin_XYZ = [0.125, -0.125, -2];
  const look = { black: 100, white: 200, density: 0.5, maximum: true };
  // Steps of an eighth from 1.5625; texel z 3 covers t 2.25 to 2.5.
  assert.deepEqual(rayPick(box_ZYX, [4, 4, 4], origin_XYZ, [0, 0, 1], look), { enter: 1.5, picked: 2.3125 });
  assert.equal(rayPick(box_ZYX, [4, 4, 4], [0.375, -0.125, -2], [0, 0, 1], look), null);
  assert.equal(rayPick(box_ZYX, [4, 4, 4], [2, 0, -2], [0, 0, 1], look), null);
  // White from texel z 1 on: each step keeps exp(-0.25) of the light, so the third keeps less than half.
  box_ZYX.fill(255, 16);
  const absorbed = { black: 0, white: 255, density: 0.5, maximum: false };
  assert.deepEqual(rayPick(box_ZYX, [4, 4, 4], origin_XYZ, [0, 0, 1], absorbed), { enter: 1.5, picked: 2.0625 });
});

test("a linear window maps its minimum to 0 and its maximum to 255, clamping past them", () => {
  const encoding = { kind: "linear_window", minimum: 1000, maximum: 2020 };
  assert.deepEqual(
    [...displayBytes(Float32Array.of(0, 1000, 1510, 2020, 9000, Number.NaN), encoding)],
    [0, 0, 128, 255, 255, 0],
  );
  assert.throws(() => displayBytes(new Uint8Array(1), { kind: "log" }), {
    message: "The 3D CT does not draw display encoding log.",
  });
});

test("a saved look is taken whole or refused", () => {
  const look = { side: 512, maximum: false, black: 10, white: 200, density: 0.2, follow: true };
  assert.equal(savedLook(look), look);
  assert.throws(() => savedLook({ ...look, side: 300 }), /malformed/);
  assert.throws(() => savedLook({ ...look, white: 5 }), /malformed/);
});
