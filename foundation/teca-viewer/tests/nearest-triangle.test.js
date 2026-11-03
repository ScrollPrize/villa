import assert from "node:assert/strict";
import { test } from "node:test";
import { cellDistanceSquared, degenerateTriangle, nearerTriangle } from "../src/surface/nearest-triangle.js";

const POINTS = new Float64Array([0, 0, 0, 0, 0, 2, 0, 2, 0, 10, 0, 0, 10, 0, 2, 10, 2, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2]);
const FIRST = [0, 3, 6];
const SECOND = [9, 12, 15];
const COLLINEAR = [18, 21, 24];

function nearestOf(query_ZYX, triangles) {
  const nearest = {
    point_ZYX: [0, 0, 0],
    preview_UV: [0, 0],
    normal_ZYX: [0, 0, 0],
    distanceSquared: Infinity,
    ambiguous: false,
  };
  for (const [[a, b, c], [u, v]] of triangles)
    nearerTriangle(nearest, query_ZYX, POINTS, a, b, c, u, v, u + 1, v, u, v + 1);
  return nearest;
}

test("the nearest point of a triangle is on its face, an edge or a vertex", () => {
  const face = nearestOf([5, 0.5, 0.5], [[FIRST, [0, 0]]]);
  assert.deepEqual(face.point_ZYX, [0, 0.5, 0.5]);
  assert.deepEqual(face.preview_UV, [0.25, 0.25]);
  assert.deepEqual(face.normal_ZYX, [1, 0, 0]);
  assert.equal(face.distanceSquared, 25);
  const edge = nearestOf([0, -1, 1], [[FIRST, [0, 0]]]);
  assert.deepEqual(edge.point_ZYX, [0, 0, 1]);
  assert.deepEqual(edge.preview_UV, [0.5, 0]);
  assert.equal(edge.distanceSquared, 1);
  const vertex = nearestOf([0, -1, -1], [[FIRST, [0, 0]]]);
  assert.deepEqual(vertex.point_ZYX, [0, 0, 0]);
  assert.equal(vertex.distanceSquared, 2);
});

test("equal distances far apart on the grid are ambiguous, and a tie keeps the lower row", () => {
  const forward = nearestOf(
    [5, 0.5, 0.5],
    [
      [FIRST, [0, 0]],
      [SECOND, [5, 5]],
    ],
  );
  const backward = nearestOf(
    [5, 0.5, 0.5],
    [
      [SECOND, [5, 5]],
      [FIRST, [0, 0]],
    ],
  );
  for (const nearest of [forward, backward]) {
    assert.equal(nearest.ambiguous, true);
    assert.deepEqual(nearest.point_ZYX, [0, 0.5, 0.5]);
  }
  assert.equal(
    nearestOf(
      [4, 0.5, 0.5],
      [
        [FIRST, [0, 0]],
        [SECOND, [5, 5]],
      ],
    ).ambiguous,
    false,
  );
});

test("a degenerate triangle is skipped", () => {
  assert.equal(degenerateTriangle(POINTS, ...COLLINEAR), true);
  assert.equal(degenerateTriangle(POINTS, ...FIRST), false);
  assert.equal(nearestOf([0, 1, 1], [[COLLINEAR, [0, 0]]]).distanceSquared, Infinity);
});

test("a cell's box distance bounds its triangles from below", () => {
  assert.equal(cellDistanceSquared([3, 0.5, 4], POINTS, 0, 3, 6, 6), 13);
  assert.equal(cellDistanceSquared([0, 0.5, 0.5], POINTS, 0, 3, 6, 6), 0);
});
