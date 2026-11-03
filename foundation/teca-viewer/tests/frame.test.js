import assert from "node:assert/strict";
import { test } from "node:test";
import { surfaceFrameAt } from "../src/surface/frame.js";

const dot = (first, second) => first.reduce((sum, value, axis) => sum + value * second[axis], 0);

test("a frame's normal, V and U rows are orthonormal ZYX vectors, U along the U tangent and the normal across V", () => {
  const pixel = ([u, v]) => [10 + 0.5 * u + 0.25 * v, 20 + v - 0.1 * u, 30 + u + 0.2 * v];
  for (const normalSign of [1, -1]) {
    const frame = surfaceFrameAt([4.5, 5.25], normalSign, [16, 16], pixel);
    const [normal_ZYX, v_ZYX, u_ZYX] = frame.axes_ZYX3;
    assert.deepEqual(normal_ZYX, frame.normal_ZYX);
    for (const axis_ZYX of frame.axes_ZYX3) assert.ok(Math.abs(Math.hypot(...axis_ZYX) - 1) < 1e-12);
    assert.ok(Math.abs(dot(normal_ZYX, v_ZYX)) < 1e-12);
    assert.ok(Math.abs(dot(normal_ZYX, u_ZYX)) < 1e-12);
    assert.ok(Math.abs(dot(v_ZYX, u_ZYX)) < 1e-12);
    const tangentU_ZYX = [0.5, -0.1, 1];
    assert.ok(Math.abs(Math.abs(dot(u_ZYX, tangentU_ZYX)) - Math.hypot(...tangentU_ZYX)) < 1e-12);
    assert.ok(Math.abs(dot(normal_ZYX, [0.25, 1, 0.2])) < 1e-12);
  }
});

test("on a grid of [2v, v, 3u] the frame at UV [1.5, 2.25] is at ZYX [4.5, 2.25, 4.5], and none lies outside the grid", () => {
  const pixel = ([u, v]) => [2 * v, v, 3 * u];
  assert.deepEqual(surfaceFrameAt([1.5, 2.25], 1, [4, 4], pixel).point_ZYX, [4.5, 2.25, 4.5]);
  assert.equal(surfaceFrameAt([-5, 9], 1, [4, 4], pixel), null);
});
