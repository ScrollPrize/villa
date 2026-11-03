import assert from "node:assert/strict";
import { test } from "node:test";
import { SurfaceCells } from "../src/extensions/points/surface.js";

function flatSurface(H, W, off = []) {
  const points_zyx_HW3 = new Float32Array(H * W * 3);
  for (let row = 0; row < H; row++)
    for (let column = 0; column < W; column++) points_zyx_HW3.set([10, row * 10, column * 10], 3 * (row * W + column));
  for (const [row, column] of off) points_zyx_HW3.fill(-1, 3 * (row * W + column), 3 * (row * W + column) + 3);
  return new SurfaceCells({ grid_HW: [H, W], points_zyx_HW3 });
}

test("a point 2 above a flat surface's cell 1.5, 1.5 is found there within 3, and not 10 above it", () => {
  const cells = flatSurface(4, 4);
  assert.deepEqual(cells.nearest([12, 15, 15], 3), [1.5, 1.5]);
  assert.equal(cells.nearest([20, 15, 15], 3), null);
});

test("a point over the cell between two patches is found in it", () => {
  assert.deepEqual(flatSurface(2, 40).nearest([10, 5, 165], 1), [0.5, 16.5]);
});

test("a point over a cell off the surface is not found on the cells 10 away", () => {
  assert.equal(flatSurface(3, 3, [[1, 1]]).nearest([10, 10, 10], 1), null);
});

test("a point on one winding nearer another winding's vertex than its own is found on its own", () => {
  const points_zyx_HW3 = new Float32Array(6 * 4 * 3).fill(-1);
  for (let row = 0; row < 6; row++)
    for (let column = 0; column < 4; column++) {
      if (row < 3) points_zyx_HW3.set([10, row * 10, column * 10], 3 * (row * 4 + column));
      if (row > 3) points_zyx_HW3.set([16, (row - 4) * 10 + 15, column * 10 + 5], 3 * (row * 4 + column));
    }
  const cells = new SurfaceCells({ grid_HW: [6, 4], points_zyx_HW3 });
  assert.deepEqual(cells.nearest([10, 15, 15], 1), [1.5, 1.5]);
  assert.deepEqual(cells.nearest([16, 15, 15], 1), [4, 1]);
});

test("on a surface 25 patches long a point 2 above cell 1.5, 390.5 is found there, looked for in patches 23 and 24 alone", () => {
  const cells = flatSurface(4, 400);
  assert.deepEqual(cells.nearest([12, 15, 3905], 3), [1.5, 390.5]);
  const looked = [...cells.looked.keys()].filter((patch) => cells.looked[patch] === cells.looks);
  assert.deepEqual(looked, [23, 24]);
  assert.equal(cells.nearest([10, 15, 4100], 3), null);
});
