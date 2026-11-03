import assert from "node:assert/strict";
import { test } from "node:test";
import { SurfacePyramid } from "../src/surface/pyramid.js";
import { TifxyzSurfaceSource } from "../src/surface/source.js";
import { SurfaceStripCache } from "../src/surface/strip-cache.js";

const GRID_HW = [130, 75];

function sheet() {
  const [height, width] = GRID_HW;
  const points_zyx_HW3 = new Float32Array(3 * height * width);
  for (let v = 0; v < height; v++)
    for (let u = 0; u < width; u++) {
      const cell = 3 * (v * width + u);
      if (u === 0 || (v * 7 + u * 3) % 29 === 0) points_zyx_HW3.fill(Number.NaN, cell, cell + 3);
      else if ((v + u) % 31 === 0) points_zyx_HW3.fill(-1, cell, cell + 3);
      else points_zyx_HW3.set([100 + Math.sin(u / 9) * 5, 200 + v, 300 + u], cell);
    }
  return { grid_HW: GRID_HW, scale_UV: [1, 1], points_zyx_HW3 };
}

async function built(surface) {
  const strips = new SurfaceStripCache(2 ** 24);
  const source = await TifxyzSurfaceSource.open("memory", surface, new AbortController().signal, strips, null);
  const pyramid = new SurfacePyramid(source.grid_HW);
  await source.forEachStrip((index, strip) => pyramid.addStrip(source, index, strip), null);
  pyramid.finish();
  assert.equal(strips.bytes, 0, "strips of points in memory are held");
  return { source, pyramid };
}

const bits = (values) => new Uint32Array(values.buffer, values.byteOffset, values.length);

for (const [name, start_HW, stop_HW] of [
  ["inside", [40, 20], [70, 33]],
  ["across the strips", [60, 0], [68, 75]],
  ["whose bound block ends in the next strip", [60, 10], [62, 20]],
  ["at the last row", [126, 30], [130, 40]],
  ["at the last column", [10, 70], [20, 75]],
  ["at the first cell", [0, 0], [1, 1]],
]) {
  test(`a patch of the cells ${name} writes what a build of every strip writes`, async () => {
    const surface = sheet();
    const { source, pyramid } = await built(surface);
    const [, width] = GRID_HW;
    for (let v = start_HW[0]; v < stop_HW[0]; v++)
      for (let u = start_HW[1]; u < stop_HW[1]; u++) {
        const cell = 3 * (v * width + u);
        if ((v + u) % 5 === 0) surface.points_zyx_HW3.fill(Number.NaN, cell, cell + 3);
        else surface.points_zyx_HW3.set([150, 200 + v / 2, 300 + u], cell);
      }
    await pyramid.patch(source, start_HW, stop_HW, null);
    const expected = (await built(surface)).pyramid;
    for (const [level, { data_HWRGBA }] of pyramid.levels.entries())
      assert.deepEqual(bits(data_HWRGBA), bits(expected.levels[level].data_HWRGBA), `level ${level}`);
    assert.deepEqual(pyramid.intersectionBounds_B6, expected.intersectionBounds_B6);
    assert.equal(pyramid.revision, 1);
  });
}

test("a 4 by 4 grid missing its last point marks the vertices beside it invalid, built or patched", async () => {
  const points_zyx_HW3 = new Float32Array(48).map((_, index) => 10 + index);
  const surface = { grid_HW: [4, 4], scale_UV: [1, 1], points_zyx_HW3 };
  const validity = (pyramid) => Array.from({ length: 9 }, (_, vertex) => pyramid.levels[0].data_HWRGBA[vertex * 4 + 3]);
  const { source, pyramid } = await built(surface);
  points_zyx_HW3.fill(Number.NaN, 45);
  await pyramid.patch(source, [3, 3], [4, 4], null);
  assert.deepEqual(validity(pyramid), [1, 1, 1, 1, 0, 0, 1, 0, 0]);
  assert.deepEqual(validity((await built(surface)).pyramid), [1, 1, 1, 1, 0, 0, 1, 0, 0]);
});
