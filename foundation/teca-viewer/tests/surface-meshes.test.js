import assert from "node:assert/strict";
import { test } from "node:test";
import { meshArrays, meshStep } from "../src/extensions/surface-meshes.js";

function grid(off) {
  const points_zyx_HW3 = new Float32Array(27);
  for (let row = 0; row < 3; row++)
    for (let column = 0; column < 3; column++)
      points_zyx_HW3.set([10 * row, column, row + column], 3 * (3 * row + column));
  for (const [cell, value] of off) points_zyx_HW3.fill(value, 3 * cell, 3 * cell + 3);
  return { grid_HW: [3, 3], scale_UV: [1, 1], points_zyx_HW3 };
}

test("Paris 4's 1598 × 2103 grid takes every second row and column within 2 M triangles, and a small one every one", () => {
  assert.equal(meshStep([1598, 2103], 2_000_000), 2);
  assert.equal(meshStep([1598, 2103], 1_000_000), 3);
  assert.equal(meshStep([3, 3], 8), 1);
  assert.equal(meshStep([3, 3], 7), 2);
});

test("each quad whose four corners lie on the surface is two triangles of the cells' x, y, z", () => {
  const { positions_xyz_N3, triangles_T3 } = meshArrays(grid([]), 1);
  assert.deepEqual([...positions_xyz_N3.subarray(0, 6)], [0, 0, 0, 1, 1, 0]);
  assert.deepEqual([...positions_xyz_N3.subarray(24)], [4, 2, 20]);
  assert.deepEqual([...triangles_T3.subarray(0, 6)], [0, 3, 1, 1, 3, 4]);
  assert.equal(triangles_T3.length, 4 * 6);
});

test("a cell all -1 or not finite is no vertex, and takes the quads it is a corner of", () => {
  const { positions_xyz_N3, triangles_T3 } = meshArrays(grid([[4, -1]]), 1);
  assert.equal(positions_xyz_N3.length, 8 * 3);
  assert.equal(triangles_T3.length, 0);
  const nan = grid([]);
  nan.points_zyx_HW3[3 * 4] = Number.NaN;
  assert.equal(meshArrays(nan, 1).triangles_T3.length, 0);
  const corner = meshArrays(grid([[0, -1]]), 1);
  assert.deepEqual([...corner.positions_xyz_N3.subarray(0, 3)], [1, 1, 0]);
  assert.deepEqual([...corner.triangles_T3], [0, 3, 1, 1, 3, 4, 2, 5, 3, 3, 5, 6, 3, 6, 4, 4, 6, 7]);
});

test("a step takes every step-th row and column from the first", () => {
  const { positions_xyz_N3, triangles_T3 } = meshArrays(grid([]), 2);
  assert.deepEqual([...positions_xyz_N3], [0, 0, 0, 2, 2, 0, 2, 0, 20, 4, 2, 20]);
  assert.deepEqual([...triangles_T3], [0, 2, 1, 1, 2, 3]);
});
