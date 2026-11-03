import assert from "node:assert/strict";
import { test } from "node:test";
import { Brush } from "../src/extensions/paint/brush.js";
import { piecesOf } from "../src/extensions/paint/history.js";
import { Labels } from "../src/extensions/paint/labels.js";
import { TILE } from "../src/extensions/paint/stamp.js";
import { meshSwept, meshTriangles } from "../src/extensions/paint/surface-sweep.js";
import { sweptTiles, tilesOf } from "../src/extensions/paint/sweep.js";
import { MemoryVolume } from "../src/volume/memory-volume.js";

const pause = async () => {};
const CENTRE_ZYX = [40, 40, 40];
const SHAPE_ZYX = [80, 80, 80];

const dot = (a_ZYX, b_ZYX) => a_ZYX[0] * b_ZYX[0] + a_ZYX[1] * b_ZYX[1] + a_ZYX[2] * b_ZYX[2];

function tileVoxels(tiles, origin_ZYX) {
  const voxels = new Set();
  for (const { from_ZYX, axis, runs_N4 } of tiles)
    for (let run = 0; run < runs_N4.length; run += 4)
      for (let step = 0; step < runs_N4[run + 3]; step++) {
        const voxel_ZYX = [0, 1, 2].map((each) => from_ZYX[each] + runs_N4[run + each] + (each === axis ? step : 0));
        voxels.add(voxel_ZYX.map((voxel, each) => voxel - origin_ZYX[each]).join());
      }
  return voxels;
}

async function disc(radii_ZYX, normal_ZYX) {
  const plane = { normal_ZYX, level: dot(normal_ZYX, CENTRE_ZYX) };
  const { tiles } = await sweptTiles(radii_ZYX, plane, CENTRE_ZYX, CENTRE_ZYX, null, SHAPE_ZYX, pause);
  return tileVoxels(tiles, CENTRE_ZYX);
}

function planes(points) {
  const planes = [];
  for (const [index, { at_ZYX, normal_ZYX }] of points.entries()) {
    const last = planes[index - 1];
    planes.push(last?.normal_ZYX === normal_ZYX ? last : { normal_ZYX, level: dot(normal_ZYX, at_ZYX) });
  }
  return planes;
}

async function swept(radii_ZYX, points) {
  const voxels = new Set();
  let again = 0;
  const along = planes(points);
  for (const [index, { at_ZYX }] of points.entries()) {
    const from_ZYX = points[Math.max(0, index - 1)].at_ZYX;
    const before = index === 0 ? null : along[index - 1];
    const { tiles } = await sweptTiles(radii_ZYX, along[index], from_ZYX, at_ZYX, before, SHAPE_ZYX, pause);
    for (const voxel of tileVoxels(tiles, [0, 0, 0])) {
      if (voxels.has(voxel)) again += 1;
      voxels.add(voxel);
    }
  }
  return { voxels, again };
}

function sweptBy(radii_ZYX, points) {
  const weights_ZYX = radii_ZYX.map((radius) => 1 / radius ** 2);
  const segments = planes(points).map((plane, index) => {
    const to_ZYX = points[index].at_ZYX;
    const from_ZYX = points[Math.max(0, index - 1)].at_ZYX;
    const off = dot(plane.normal_ZYX, from_ZYX) - plane.level;
    return { plane, from_ZYX: from_ZYX.map((from, axis) => from - off * plane.normal_ZYX[axis]), to_ZYX };
  });
  const voxels = new Set();
  for (let z = 0; z < SHAPE_ZYX[0]; z++)
    for (let y = 0; y < SHAPE_ZYX[1]; y++)
      for (let x = 0; x < SHAPE_ZYX[2]; x++)
        for (const { plane, from_ZYX, to_ZYX } of segments) {
          const height = dot(plane.normal_ZYX, [z, y, x]) - plane.level;
          if (height < -0.5 || height >= 0.5) continue;
          const d_ZYX = to_ZYX.map((to, axis) => to - from_ZYX[axis]);
          const w_ZYX = [z, y, x].map((voxel, axis) => voxel - from_ZYX[axis]);
          const length = d_ZYX.reduce((total, d, axis) => total + d * d * weights_ZYX[axis], 0);
          const toward = d_ZYX.reduce((total, d, axis) => total + d * w_ZYX[axis] * weights_ZYX[axis], 0);
          const t = length === 0 ? 0 : Math.min(1, Math.max(0, toward / length));
          const reach = w_ZYX.reduce((total, w, axis) => total + (w - t * d_ZYX[axis]) ** 2 * weights_ZYX[axis], 0);
          if (reach > 1) continue;
          voxels.add(`${z},${y},${x}`);
          break;
        }
  return voxels;
}

test("a brush of radius 2 on the xy plane sets the 13 voxels of its disc, in one tile of the grid", async () => {
  const plane = { normal_ZYX: [1, 0, 0], level: CENTRE_ZYX[0] };
  const { tiles } = await sweptTiles([2, 2, 2], plane, CENTRE_ZYX, CENTRE_ZYX, null, SHAPE_ZYX, pause);
  assert.equal(tiles.length, 1);
  assert.equal(tiles[0].from_ZYX.join(), "32,32,32");
  assert.deepEqual(tiles[0].shape_ZYX, [TILE, TILE, TILE]);
  assert.deepEqual([...tiles[0].runs_N4], [8, 6, 8, 1, 8, 7, 7, 3, 8, 8, 6, 5, 8, 9, 7, 3, 8, 10, 8, 1]);
});

test("a brush of radius 20 on the xy plane sets the 1257 lattice points of its disc", async () => {
  assert.equal((await disc([20, 20, 20], [1, 0, 0])).size, 1257);
});

test("a brush of radius 20 on a tilted plane sets exactly the voxels within 20 and half a voxel of it", async () => {
  const normal_ZYX = [0.36, 0.48, 0.8];
  const expected = new Set();
  for (let z = -20; z <= 20; z++)
    for (let y = -20; y <= 20; y++)
      for (let x = -20; x <= 20; x++)
        if (Math.abs(z * normal_ZYX[0] + y * normal_ZYX[1] + x * normal_ZYX[2]) <= 0.5 && z * z + y * y + x * x <= 400)
          expected.add(`${z},${y},${x}`);
  assert.deepEqual(await disc([20, 20, 20], normal_ZYX), expected);
});

test("a brush of radii 3 along y and 5 along x on the xy plane sets the ellipse's 45 lattice points", async () => {
  const voxels = await disc([1, 3, 5], [1, 0, 0]);
  assert.equal(voxels.size, 45);
  assert.ok(voxels.has("0,0,5") && voxels.has("0,3,0") && voxels.has("0,2,3") && !voxels.has("0,2,4"));
});

test("a brush of radii 6, 12 and 20 on a tilted plane sets exactly the ellipsoid's voxels within half a voxel of it", async () => {
  const normal_ZYX = [0.36, 0.48, 0.8];
  const expected = new Set();
  for (let z = -20; z <= 20; z++)
    for (let y = -20; y <= 20; y++)
      for (let x = -20; x <= 20; x++)
        if (
          Math.abs(z * normal_ZYX[0] + y * normal_ZYX[1] + x * normal_ZYX[2]) <= 0.5 &&
          z * z * 144 * 400 + y * y * 36 * 400 + x * x * 36 * 144 <= 36 * 144 * 400
        )
          expected.add(`${z},${y},${x}`);
  assert.deepEqual(await disc([6, 12, 20], normal_ZYX), expected);
});

test("a brush of radii 4, 6 and 32 on a plane leaning mostly on z sets the 367 voxels of its ellipse and rim", async () => {
  const size = Math.hypot(1, 0.2, 0.16);
  const normal_ZYX = [1 / size, 0.2 / size, 0.16 / size];
  const expected = new Set();
  for (let z = -4; z <= 4; z++)
    for (let y = -6; y <= 6; y++)
      for (let x = -32; x <= 32; x++)
        if (
          Math.abs(z * normal_ZYX[0] + y * normal_ZYX[1] + x * normal_ZYX[2]) <= 0.5 &&
          z * z * 36 * 1024 + y * y * 16 * 1024 + x * x * 16 * 36 <= 16 * 36 * 1024
        )
          expected.add(`${z},${y},${x}`);
  assert.equal(expected.size, 367);
  assert.deepEqual(await disc([4, 6, 32], normal_ZYX), expected);
});

function cubeLabels(side) {
  const shape_ZYX = [side, side, side];
  const levels = [{ shape_ZYX, chunks_ZYX: shape_ZYX, downsample_ZYX: [1, 1, 1], offset_ZYX: [0, 0, 0] }];
  const viewer = {
    memoryVolume: (name, zeroed) => new MemoryVolume(name, levels, zeroed),
    addLayer: () => ({ changed() {} }),
  };
  const ct = {
    rootUrl: "memory://ct/",
    scales: [{ scale_ZYX: [1, 1, 1], translation_ZYX: [0, 0, 0] }],
    metadata: { physicalUnits_ZYX: ["micrometer", "micrometer", "micrometer"] },
  };
  return new Labels(viewer, ct, { journal: () => {}, compact: async () => {} }, (length) => new Uint8Array(length));
}

function labelled(labels) {
  const [side] = labels.volume.scales[0].shape_ZYX;
  const voxels = new Set();
  for (const [voxel, label] of labels.volume.chunk(0, [0, 0, 0]).entries())
    if (label === 1) voxels.add(`${Math.floor(voxel / side ** 2)},${Math.floor(voxel / side) % side},${voxel % side}`);
  return voxels;
}

async function stroked(pane, normal_ZYX, points_ZYX) {
  const labels = cubeLabels(40);
  const brush = new Brush([4, 4, 4], (error) => assert.fail(error), null);
  brush.begin(labels, null, 1);
  for (const [index, point_ZYX] of points_ZYX.entries())
    brush.to({ pane, point_ZYX, normal_ZYX }, index === points_ZYX.length - 1);
  await brush.end();
  return labels;
}

test("a stroke of radius 4 from 20,20,10 to 20,20,20 on xy sets the 139 voxels within 4 of it, however its points fall", async () => {
  for (const xs of [
    [10, 20],
    [10, 13, 16, 19, 20],
  ]) {
    const labels = await stroked(
      "xy",
      [1, 0, 0],
      xs.map((x) => [20, 20, x]),
    );
    assert.equal(labelled(labels).size, 139, `points at x ${xs.join(", ")}`);
  }
});

test("a stroke of radius 4 from 10,20,20 to 20,20,20 on zy sets the 139 voxels its sweeps hold, however its points fall", async () => {
  const normal_ZYX = [0, 0, 1];
  for (const zs of [
    [10, 20],
    [10, 13, 16, 19, 20],
  ]) {
    const points_ZYX = zs.map((z) => [z, 20, 20]);
    const labels = await stroked("zy", normal_ZYX, points_ZYX);
    const { voxels } = await swept(
      [4, 4, 4],
      points_ZYX.map((at_ZYX) => ({ at_ZYX, normal_ZYX })),
    );
    assert.equal(voxels.size, 139);
    assert.deepEqual(labelled(labels), voxels, `points at z ${zs.join(", ")}`);
  }
});

test("one sweep of radius 4 from 10,20,20 to 20,20,20 on zy is undone as its 19 rows along y, z 6 to 24", async () => {
  const labels = cubeLabels(40);
  const plane = { normal_ZYX: [0, 0, 1], level: 20 };
  const { tiles } = await sweptTiles([4, 4, 4], plane, [10, 20, 20], [20, 20, 20], null, [40, 40, 40], pause);
  labels.beginStroke();
  for (const tile of tiles) labels.stamp(tile, [0, 0, 0], 1, null);
  labels.endStroke(1);
  assert.deepEqual(labelled(labels), tileVoxels(tiles, [0, 0, 0]));
  const [[, runs]] = piecesOf(labels.history.done[0]);
  assert.equal(runs.axis, 1);
  assert.deepEqual(
    [...runs.offsets_N].map((offset) => Math.floor(offset / 1600)),
    Array.from({ length: 19 }, (_, index) => 6 + index),
  );
  // 9 bytes a run and 24 a piece: a run a voxel would be 139 of them.
  assert.equal(labels.history.done[0].bytes, 195);
  assert.equal(await labels.turn(true, pause), true);
  assert.equal(labelled(labels).size, 0);
});

test("a stroke of radius 12 across the plane z + y + x = 120 sets its capsule's voxels, none of them twice", async () => {
  const normal_ZYX = [1, 1, 1].map((component) => component / Math.sqrt(3));
  const direction_ZYX = [1, -1, 0].map((component) => component / Math.sqrt(2));
  const start_ZYX = [40.3, 39.6, 40.1];
  const points = Array.from({ length: 8 }, (_, index) => ({
    at_ZYX: start_ZYX.map((start, axis) => start + (index * 3 - 10) * direction_ZYX[axis]),
    normal_ZYX,
  }));
  const { voxels, again } = await swept([12, 12, 12], points);
  const expected = sweptBy([12, 12, 12], points);
  assert.equal(expected.size, 542);
  assert.deepEqual(voxels, expected);
  assert.equal(again, 0);
});

test("a stroke of radii 5, 8 and 6 over a surface turning 3.4° a point sets each sweep's voxels on its plane", async () => {
  const points = Array.from({ length: 8 }, (_, index) => {
    const angle = (index * 9 - 30) / 150;
    const normal_ZYX = [0.5, Math.cos(angle), Math.sin(angle)].map((component) => component / Math.hypot(0.5, 1));
    return {
      at_ZYX: [40 + 20 * Math.sin(angle), 40 + 150 * (Math.cos(angle) - 1), 40 + 150 * Math.sin(angle)],
      normal_ZYX,
    };
  });
  const { voxels } = await swept([5, 8, 6], points);
  const expected = sweptBy([5, 8, 6], points);
  assert.equal(expected.size, 1327);
  assert.deepEqual(voxels, expected);
});

function cylinder([h, w]) {
  if (h < 0 || h > 30 || w < 0 || w > 90) return null;
  const angle = (2 * w) / 30;
  return {
    point_ZYX: [10 + 2 * h, 40 + 30 * Math.cos(angle), 40 + 30 * Math.sin(angle)],
    normal_ZYX: [0, Math.cos(angle), Math.sin(angle)],
  };
}

async function surfaceStroke(labels, surface, radius, value, cells_HW) {
  const sweepers = [async (job) => meshSwept(job)];
  const brush = new Brush(
    [radius, radius, radius],
    (error) => assert.fail(error),
    null,
    () => surface,
    sweepers,
  );
  brush.begin(labels, null, value);
  for (const [index, cell_HW] of cells_HW.entries())
    brush.to({ pane: "surface", cell_HW, ...surface(cell_HW) }, index === cells_HW.length - 1);
  await brush.end();
}

test("a stroke of radius 4 from x 10 to x 20 over a flat surface's cells sets the 139 voxels it sets on its plane", async () => {
  const flat = ([h, w]) =>
    h < 0 || w < 0 || h > 30 || w > 30 ? null : { point_ZYX: [20, 1.3 * h, 1.3 * w], normal_ZYX: [1, 0, 0] };
  const labels = cubeLabels(40);
  await surfaceStroke(
    labels,
    flat,
    4,
    1,
    [10, 13, 20].map((x) => [20 / 1.3, x / 1.3]),
  );
  assert.equal(labelled(labels).size, 139);
  const vertexAt = (h, w) => flat([h, w])?.point_ZYX ?? null;
  const [from_ZYX, to_ZYX] = [
    [20, 20, 10],
    [20, 20, 20],
  ];
  const points_N9 = await meshTriangles([4, 4, 4], vertexAt, [20 / 1.3, 20 / 1.3], from_ZYX, to_ZYX, false, pause);
  const mesh = tilesOf(
    meshSwept({ radii_ZYX: [4, 4, 4], points_N9, from_ZYX, to_ZYX, before: false, shape_ZYX: SHAPE_ZYX }),
  );
  const plane = { normal_ZYX: [1, 0, 0], level: 20 };
  const swept = await sweptTiles([4, 4, 4], plane, from_ZYX, to_ZYX, null, SHAPE_ZYX, pause);
  assert.deepEqual(tileVoxels(mesh.tiles, [0, 0, 0]), tileVoxels(swept.tiles, [0, 0, 0]));
});

test("on a cylinder of radius 30 a stroke of radius 4 sets 640 voxels within half a voxel of it, and one of radius 8 over its cells, points 8 times as far apart, erases them all", async () => {
  const labels = cubeLabels(80);
  const along = (count, from, to) =>
    Array.from({ length: count + 1 }, (_, index) => [15, from + ((to - from) * index) / count]);
  await surfaceStroke(labels, cylinder, 4, 1, along(40, 10, 50));
  const painted = [...labelled(labels)].map((voxel) => voxel.split(",").map(Number));
  assert.equal(painted.length, 640);
  for (const [z, y, x] of painted) assert.ok(Math.abs(Math.hypot(y - 40, x - 40) - 30) < 0.5, `${z},${y},${x}`);
  await surfaceStroke(labels, cylinder, 8, 0, along(5, 8, 52));
  assert.equal(labelled(labels).size, 0);
});

const SHEET_Z = [20, 30, 20, 10, 20];
function sheet([h, w]) {
  if (h < 0 || h > 20 || w < 0 || w > 40) return null;
  const piece = Math.min(3, Math.floor(w / 10));
  const slope = (SHEET_Z[piece + 1] - SHEET_Z[piece]) / 10;
  return {
    point_ZYX: [SHEET_Z[piece] + slope * (w - 10 * piece), 10 + h, w],
    normal_ZYX: [1, 0, -slope].map((component) => component / Math.hypot(1, slope)),
  };
}

test("over a sheet whose heights at five equal steps are 20, 30, 20, 10 and 20, a stroke of radius 2 in one move sets the 205 voxels of one with a point at each step", async () => {
  const once = cubeLabels(48);
  await surfaceStroke(once, sheet, 2, 1, [
    [10, 0],
    [10, 40],
  ]);
  const stepped = cubeLabels(48);
  await surfaceStroke(
    stepped,
    sheet,
    2,
    1,
    [0, 10, 20, 30, 40].map((w) => [10, w]),
  );
  assert.equal(labelled(stepped).size, 205);
  assert.ok(labelled(stepped).has("30,20,10") && labelled(stepped).has("10,20,30"));
  assert.deepEqual(labelled(once), labelled(stepped));
});
