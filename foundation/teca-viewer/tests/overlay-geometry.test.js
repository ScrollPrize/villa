import assert from "node:assert/strict";
import { test } from "node:test";
import { IDENTITY_ROTATION } from "../src/math.js";
import {
  buildSurfaceIntersectionIndex,
  paneBasis,
  surfaceIntersectionSegments,
} from "../src/surface/overlay-geometry.js";

const XY = paneBasis("xy", IDENTITY_ROTATION);

test("the unrotated XY pane has right_ZYX [0, 0, 1], up_ZYX [0, 1, 0] and normal_ZYX [1, 0, 0]", () => {
  assert.deepEqual(XY, { right_ZYX: [0, 0, 1], up_ZYX: [0, 1, 0], normal_ZYX: [1, 0, 0] });
});

test("a 3 by 3 grid rising in Z with its row crosses Z 5 in two segments along V 5 from U 0 to 20", () => {
  const points_zyx_HW3 = new Float32Array(3 * 3 * 3);
  for (let row = 0; row < 3; row++)
    for (let col = 0; col < 3; col++) points_zyx_HW3.set([row * 10, row * 10, col * 10], (row * 3 + col) * 3);
  const segments_uv = surfaceIntersectionSegments(points_zyx_HW3, 3, 3, XY, 5);
  assert.equal(segments_uv.length, 8);
  for (let i = 0; i < segments_uv.length; i += 2) assert.ok(Math.abs(segments_uv[i + 1] - 5) < 1e-6);
  assert.deepEqual(
    [segments_uv[0], segments_uv[2], segments_uv[4], segments_uv[6]].sort((a, b) => a - b),
    [0, 10, 10, 20],
  );
});

test("a 65 by 65 grid crossing Z 4.5 gives 64 segments from 512 tested cells in 8-cell blocks and 4096 unculled", () => {
  const points_zyx_HW3 = new Float32Array(65 * 65 * 3);
  for (let row = 0; row < 65; row++)
    for (let col = 0; col < 65; col++) points_zyx_HW3.set([row, row * 2, col * 3], (row * 65 + col) * 3);
  points_zyx_HW3.fill(Number.NaN, (20 * 65 + 20) * 3, (20 * 65 + 20) * 3 + 3);
  const indexedEvidence = {};
  const unculledEvidence = {};
  const indexed = surfaceIntersectionSegments(
    points_zyx_HW3,
    65,
    65,
    XY,
    4.5,
    buildSurfaceIntersectionIndex(points_zyx_HW3, 65, 65, 8),
    indexedEvidence,
  );
  const unculled = surfaceIntersectionSegments(
    points_zyx_HW3,
    65,
    65,
    XY,
    4.5,
    buildSurfaceIntersectionIndex(points_zyx_HW3, 65, 65, 65),
    unculledEvidence,
  );
  assert.deepEqual(indexed, unculled);
  assert.equal(indexed.length / 4, 64);
  assert.equal(indexedEvidence.testedCells, 512);
  assert.equal(unculledEvidence.testedCells, 4096);
});

test("a cell with a NaN corner gives no segment", () => {
  const points_zyx_HW3 = new Float32Array([0, 0, 0, 0, 0, 10, 10, 10, 0, Number.NaN, Number.NaN, Number.NaN]);
  assert.deepEqual(surfaceIntersectionSegments(points_zyx_HW3, 2, 2, XY, 5), []);
});
