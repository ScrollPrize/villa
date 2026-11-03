import assert from "node:assert/strict";
import { test } from "node:test";
import { surfaceFiles, tiff } from "../src/extensions/mesh-edit/file.js";
import { extent, gathered, stepped } from "../src/extensions/mesh-edit/log.js";
import {
  gridEdges,
  grown,
  nearestVertex,
  onRim,
  onSurface,
  reach,
  smoothed,
} from "../src/extensions/mesh-edit/stroke.js";
import { unzipped, zipped } from "../src/zip.js";

function flatGrid(height, width) {
  const points_zyx_N3 = new Float32Array(3 * height * width);
  for (let index = 0; index < height * width; index++)
    points_zyx_N3.set([0, 10 * Math.floor(index / width), 10 * (index % width)], 3 * index);
  return points_zyx_N3;
}

test("a drag reaches the vertices inside its radius across the grid, weighted (1 − r²)², and none off the surface", () => {
  const points_zyx_N3 = flatGrid(5, 5);
  points_zyx_N3.fill(-1, 3 * 6, 3 * 7);
  const { indices, weights } = reach(points_zyx_N3, [5, 5], [0.5, 0.5], 12, 3);
  assert.deepEqual([...indices], [7, 8, 11, 12, 13, 16, 17, 18]);
  const side = (1 - 4 / 9) ** 2;
  const corner = (1 - 8 / 9) ** 2;
  assert.deepEqual(
    [...weights].map((weight) => weight.toFixed(6)),
    [side, corner, side, 1, side, corner, side, corner].map((weight) => weight.toFixed(6)),
  );
});

test("a drag's reach takes columns by the grid's first scale and rows by its second", () => {
  const { indices, weights } = reach(flatGrid(5, 5), [5, 5], [1, 0.5], 12, 2);
  assert.deepEqual([...indices], [11, 12, 13]);
  assert.deepEqual([...weights], [(1 - 1 / 4) ** 2, 1, (1 - 1 / 4) ** 2]);
});

test("the vertex grabbed is the nearest on the surface within reach, and none when all are further", () => {
  const points_zyx_N3 = flatGrid(3, 3);
  const on = (index) => onSurface(points_zyx_N3, index);
  assert.equal(nearestVertex(points_zyx_N3, [0, 11, 9], 5, on), 4);
  points_zyx_N3.fill(-1, 3 * 4, 3 * 5);
  assert.equal(nearestVertex(points_zyx_N3, [0, 11, 9], 5, on), -1);
  assert.equal(nearestVertex(points_zyx_N3, [0, 11, 9], 20, on), 3);
});

test("a growing grabs the nearest vertex beside a hole in its row or column, not one beside the grid's edge", () => {
  const points_zyx_N3 = flatGrid(4, 4);
  points_zyx_N3.fill(-1, 3 * 5, 3 * 6);
  const rim = (index) => onRim(points_zyx_N3, [4, 4], index);
  assert.equal(
    nearestVertex(points_zyx_N3, [0, 28, 20], 15, (index) => onSurface(points_zyx_N3, index)),
    14,
  );
  assert.equal(nearestVertex(points_zyx_N3, [0, 28, 20], 15, rim), 9);
  assert.equal(nearestVertex(points_zyx_N3, [0, 28, 20], 12, rim), -1);
});

function tiltedGrid(height, width) {
  const points_zyx_N3 = new Float32Array(3 * height * width);
  for (let r = 0; r < height; r++)
    for (let c = 0; c < width; c++) points_zyx_N3.set([3 * r + 2 * c + 1, r, c], 3 * (r * width + c));
  return points_zyx_N3;
}

test("smoothing lowers a bump toward its neighbours, and leaves a plane as it is, at the grid's edge too", () => {
  const plane = tiltedGrid(6, 5);
  const indices = Int32Array.of(0, 2, 4, 12, 13, 14, 29);
  const weights = new Float32Array(indices.length).fill(1);
  const current_N3 = gathered(plane, indices);
  assert.deepEqual(smoothed(plane, [6, 5], indices, weights, current_N3, 0.5), current_N3);
  const bumped = plane.slice();
  bumped[3 * 12] += 8;
  const after_N3 = smoothed(bumped, [6, 5], indices, weights, gathered(bumped, indices), 0.5);
  assert.equal(after_N3[3 * 3], plane[3 * 12] + 4);
  assert.equal(after_N3[3 * 4], plane[3 * 13] + 1);
});

test("growing fills a hole in a plane as the plane has it, and stops at the grid's edge", () => {
  const plane = tiltedGrid(8, 6);
  const holed = plane.slice();
  for (const cell of [20, 21, 26, 27, 32]) holed.fill(-1, 3 * cell, 3 * cell + 3);
  const hole = grown(holed, [8, 6], [1, 1], 14, 4);
  assert.deepEqual(
    [...hole.indices].sort((a, b) => a - b),
    [20, 21, 26, 27, 32],
  );
  assert.deepEqual(hole.after_N3, gathered(plane, hole.indices));
  const cut = plane.slice();
  for (let r = 0; r < 8; r++) cut.fill(-1, 3 * (6 * r + 4), 3 * (6 * r + 6));
  const edge = grown(cut, [8, 6], [1, 1], 6 * 4 + 3, 20);
  assert.deepEqual(
    [...edge.indices].sort((a, b) => a - b),
    [4, 5, 10, 11, 16, 17, 22, 23, 28, 29, 34, 35, 40, 41, 46, 47],
  );
  assert.deepEqual(edge.after_N3, gathered(plane, edge.indices));
  assert.equal(grown(cut, [8, 6], [1, 1], 6 * 4, 2).indices.length, 0);
});

test("growing closes a hole in a bowl as the bowl curves", () => {
  const bowl = new Float32Array(3 * 12 * 12);
  const holed = new Float32Array(3 * 12 * 12);
  let hole = 0;
  for (let index = 0; index < 12 * 12; index++) {
    const [r, c] = [Math.floor(index / 12), index % 12];
    bowl.set([((r - 6) ** 2 + (c - 6) ** 2) / 4, 10 * r, 10 * c], 3 * index);
    const off = (r - 6) ** 2 + (c - 6) ** 2 < 9;
    holed.set(off ? [-1, -1, -1] : bowl.subarray(3 * index, 3 * index + 3), 3 * index);
    hole += off ? 1 : 0;
  }
  const { indices, after_N3 } = grown(holed, [12, 12], [1, 1], 6 * 12 + 6, 4);
  assert.equal(indices.length, hole);
  const expected_N3 = gathered(bowl, indices);
  const worst = Math.max(...after_N3.map((value, at) => Math.abs(value - expected_N3[at])));
  assert.ok(worst < 1e-3, `a grown vertex is ${worst} off the bowl`);
});

test("the grid's edges join neighbours in a row and a column, and none across a row's end", () => {
  assert.deepEqual(gridEdges(Int32Array.from([0, 1, 3, 4]), 3), [0, 1, 0, 2, 1, 3, 2, 3]);
  assert.deepEqual(gridEdges(Int32Array.from([2, 3]), 3), []);
});

test("an edit's extent is the grid cells from its first row and column to past its last", () => {
  assert.deepEqual(extent(Int32Array.of(9, 15, 12), 5), [
    [1, 0],
    [4, 5],
  ]);
  assert.deepEqual(extent(Int32Array.of(0), 5), [
    [0, 0],
    [1, 1],
  ]);
});

test("an edit done and undone gives the points back bit for bit, NaN included, and is refused on other points", () => {
  const points_zyx_HW3 = flatGrid(2, 3);
  points_zyx_HW3.fill(Number.NaN, 3 * 5);
  const files = points_zyx_HW3.slice();
  const indices = Int32Array.from([1, 5]);
  const edit = { indices, before_N3: gathered(points_zyx_HW3, indices), after_N3: Float32Array.of(7, 8, 9, 1, 2, 3) };
  stepped(points_zyx_HW3, edit, true);
  assert.deepEqual([...gathered(points_zyx_HW3, indices)], [7, 8, 9, 1, 2, 3]);
  assert.deepEqual([...gathered(points_zyx_HW3, Int32Array.of(0, 2))], [0, 0, 0, 0, 0, 20]);
  stepped(points_zyx_HW3, edit, false);
  assert.deepEqual(new Uint32Array(points_zyx_HW3.buffer), new Uint32Array(files.buffer));
  points_zyx_HW3[3] = 0.5;
  assert.throws(() => stepped(points_zyx_HW3, edit, true), {
    message: "Vertex 1 is at 0.5, 0, 10, not where the edit found it: the surface changed since the edit was kept.",
  });
  assert.deepEqual([...gathered(points_zyx_HW3, indices)].slice(0, 3), [0.5, 0, 10]);
});

test("a TIFF page is little-endian, one uncompressed float32 strip, its values where the strip offset says", () => {
  const values_HW = Float32Array.from([1.5, -1, 2.25, 1e6, Number.NaN, 0]);
  const { bytes, pixels_HW } = tiff(2, 3);
  pixels_HW.set(values_HW);
  const view = new DataView(bytes.buffer);
  assert.equal(view.getUint16(0, true), 0x4949);
  assert.equal(view.getUint16(2, true), 42);
  const ifd = view.getUint32(4, true);
  const tags = new Map();
  for (let entry = 0; entry < view.getUint16(ifd, true); entry++) {
    const at = ifd + 2 + 12 * entry;
    const short = view.getUint16(at + 2, true) === 3;
    tags.set(view.getUint16(at, true), short ? view.getUint16(at + 8, true) : view.getUint32(at + 8, true));
  }
  assert.deepEqual(
    [256, 257, 258, 259, 277, 278, 279, 339].map((tag) => tags.get(tag)),
    [3, 2, 32, 1, 1, 2, 24, 3],
  );
  const values = Array.from({ length: 6 }, (_, cell) => view.getFloat32(tags.get(273) + 4 * cell, true));
  assert.deepEqual(values, [...values_HW]);
});

test("a saved surface is the tifxyz directory: its scale, the box of its points on the surface, and x, y, z", async () => {
  const points_zyx_HW3 = Float32Array.from([3, 2, 1, -1, -1, -1, 6, Number.NaN, 4, 30, 20, 10]);
  const files = surfaceFiles({ grid_HW: [2, 2], scale_UV: [0.05, 0.05], points_zyx_HW3 }, "edited");
  const zip = new Uint8Array(await (await zipped(files, async () => {}, false)).arrayBuffer());
  const entries = new Map((await unzipped(zip)).map(({ name, bytes }) => [name, bytes]));
  assert.deepEqual([...entries.keys()], ["edited/meta.json", "edited/x.tif", "edited/y.tif", "edited/z.tif"]);
  assert.deepEqual(JSON.parse(new TextDecoder().decode(entries.get("edited/meta.json"))), {
    format: "tifxyz",
    type: "seg",
    uuid: "edited",
    scale: [0.05, 0.05],
    bbox: [
      [1, 2, 3],
      [10, 20, 30],
    ],
  });
  for (const [axis, letter] of ["x", "y", "z"].entries()) {
    const bytes = entries.get(`edited/${letter}.tif`).slice(160);
    const values = Float32Array.from([0, 1, 2, 3], (cell) => points_zyx_HW3[3 * cell + 2 - axis]);
    assert.deepEqual(new Uint32Array(bytes.buffer), new Uint32Array(values.buffer), `${letter}.tif`);
  }
});
