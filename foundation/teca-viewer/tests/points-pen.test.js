import assert from "node:assert/strict";
import { test } from "node:test";
import { Pen, spaced } from "../src/extensions/points/pen.js";
import { emptySkeleton, newTree } from "../src/extensions/points/skeleton.js";

const LEVEL = { shape_ZYX: [10, 20, 30], downsample_ZYX: [1, 1, 1], offset_ZYX: [0, 0, 0] };

function contents(skeleton) {
  const { groups, trees, nodes, links, branchpoints, selected } = skeleton;
  return structuredClone({ groups, trees, nodes, links, branchpoints, selected });
}

function at(fields) {
  return {
    winding: 0,
    side: "recto",
    sheet: 0,
    role: "fiber",
    fiber: 1,
    tree: null,
    mark: null,
    air: false,
    ...fields,
  };
}

function testPen(level = LEVEL, surface = assert.fail) {
  const skeleton = emptySkeleton();
  const told = { edits: 0, released: [], selected: [] };
  const mode = { editing: false };
  const pen = new Pen({
    skeleton: () => skeleton,
    level: () => level,
    target: () => at({}),
    spacing: () => 5,
    depth: () => 2,
    surfaceNode: () => null,
    joined: () => false,
    surfacePoints: () => surface,
    editing: () => mode.editing,
    shown: () => () => true,
    drawn: () => {},
    edited: () => {
      told.edits += 1;
    },
    selected: (id) => {
      skeleton.selected = id;
      told.selected.push(id);
    },
    clicked: assert.fail,
    said: assert.fail,
    putAway: () => {},
    released: (id, pane) => told.released.push([id, pane]),
  });
  return { skeleton, told, pen, mode };
}

function onPane(x) {
  return { pane: "xy", point_ZYX: [5, 5, x], normal_ZYX: [1, 0, 0], pixel: 1 };
}

test("a stroke of spacing 5 places points at 5 and 10 on its first 12 voxels, then at 15 and 20 on the next 8", () => {
  const first = spaced([0, 0, 0], [0, 0, 12], 5, 5);
  assert.deepEqual(first, {
    points_ZYX: [
      [0, 0, 5],
      [0, 0, 10],
    ],
    carried: 3,
  });
  const second = spaced([0, 0, 12], [0, 0, 20], first.carried, 5);
  assert.deepEqual(second, {
    points_ZYX: [
      [0, 0, 15],
      [0, 0, 20],
    ],
    carried: 5,
  });
});

test("New class during a stroke is refused, so two undos after its release leave no groups, trees or nodes selected", () => {
  const { skeleton, told, pen } = testPen();
  pen.down(onPane(2));
  pen.move(onPane(14));
  assert.equal(pen.commit(newTree(skeleton, [])), false);
  assert.equal(pen.undo(), false);
  pen.up(onPane(14));
  assert.deepEqual(
    [...skeleton.nodes.values()].map((node) => node.point_ZYX[2]),
    [2, 7, 12],
  );
  assert.equal(told.edits, 1);
  assert.deepEqual(told.released, [[skeleton.selected, "xy"]]);
  assert.equal(pen.undo(), true);
  assert.equal(pen.undo(), false);
  assert.deepEqual(contents(skeleton), contents(emptySkeleton()));
});

test("in Edit a press on node 1 selects it, one on empty space none, and no click or drag places or moves a node", () => {
  const { skeleton, told, pen, mode } = testPen();
  pen.down(onPane(2));
  pen.up(onPane(2));
  const nodes = contents(skeleton).nodes;
  mode.editing = true;
  for (const [from, to] of [
    [2, 14],
    [20, 20],
    [20, 28],
  ]) {
    pen.down(onPane(from));
    pen.move(onPane(to));
    pen.up(onPane(to));
  }
  assert.deepEqual(contents(skeleton).nodes, nodes);
  assert.deepEqual([told.edits, skeleton.done.length, told.selected], [1, 1, [1, null, null]]);
  assert.deepEqual(told.released, [
    [1, "xy"],
    [1, "xy"],
  ]);
});

test("a click and a stroke on an xy pane tilted to normal 0.6, 0.8, 0 keep that on each node, and a node dragged on zy too", () => {
  const { skeleton, pen } = testPen();
  const tilted = (x) => ({ ...onPane(x), normal_ZYX: [0.6, 0.8, 0] });
  for (const [from, to] of [
    [2, 2],
    [20, 28],
  ]) {
    pen.down(tilted(from));
    pen.move(tilted(to));
    pen.up(tilted(to));
  }
  const xy = { pane: "xy", normal_ZYX: [0.6, 0.8, 0] };
  assert.deepEqual(
    [...skeleton.nodes.values()].map(({ point_ZYX, view }) => [point_ZYX[2], view]),
    [
      [2, xy],
      [20, xy],
      [25, xy],
    ],
  );
  const zy = (x) => ({ pane: "zy", point_ZYX: [5, 5, x], normal_ZYX: [0, 0, 1], pixel: 1 });
  pen.down(zy(2));
  pen.move(zy(8));
  pen.up(zy(8));
  assert.deepEqual([skeleton.nodes.get(1).point_ZYX, skeleton.nodes.get(1).view], [[5, 5, 8], xy]);
});

test("a stroke under way when the pen is put away is an edit, which one undo takes back", () => {
  const { skeleton, told, pen } = testPen();
  pen.down(onPane(2));
  pen.move(onPane(9));
  pen.putAway();
  assert.deepEqual(
    [skeleton.nodes.size, skeleton.done.length, told.edits, told.released.length, pen.busy()],
    [2, 1, 1, 0, false],
  );
  pen.undo();
  assert.deepEqual(contents(skeleton), contents(emptySkeleton()));
});

test("a stroke on the Surface of a cylinder of radius 30 places its 16 nodes within a voxel of it, its points 76° apart", () => {
  const cylinder = ([h, w]) => {
    const angle = (2 * w) / 30;
    return { point_ZYX: [10 + 2 * h, 40 + 30 * Math.cos(angle), 40 + 30 * Math.sin(angle)], normal_ZYX: [0, 1, 0] };
  };
  const onSurface = (cell_HW) => ({ pane: "surface", cell_HW, pixel: 1, ...cylinder(cell_HW) });
  const level = { shape_ZYX: [80, 80, 80], downsample_ZYX: [1, 1, 1], offset_ZYX: [0, 0, 0] };
  const { skeleton, pen } = testPen(level, cylinder);
  pen.down(onSurface([15, 10]));
  pen.move(onSurface([15, 30]));
  pen.move(onSurface([15, 50]));
  pen.up(onSurface([15, 50]));
  const nodes_ZYX = [...skeleton.nodes.values()].map((node) => node.point_ZYX);
  for (const [z, y, x] of nodes_ZYX) assert.ok(Math.abs(Math.hypot(y - 40, x - 40) - 30) < 1, `${z},${y},${x}`);
  assert.equal(nodes_ZYX.length, 16);
});

test("a stroke in one move over a sheet whose heights at five equal steps are 20, 30, 20, 10 and 20 places its 12 nodes on it", () => {
  const heights_Z = [20, 30, 20, 10, 20];
  const height = (x) => {
    const piece = Math.min(3, Math.floor(x / 10));
    return heights_Z[piece] + ((heights_Z[piece + 1] - heights_Z[piece]) * (x - 10 * piece)) / 10;
  };
  const sheet = ([h, w]) => ({ point_ZYX: [height(w), 10 + h, w], normal_ZYX: [1, 0, 0] });
  const onSurface = (cell_HW) => ({ pane: "surface", cell_HW, pixel: 1, ...sheet(cell_HW) });
  const level = { shape_ZYX: [48, 48, 48], downsample_ZYX: [1, 1, 1], offset_ZYX: [0, 0, 0] };
  const { skeleton, pen } = testPen(level, sheet);
  pen.down(onSurface([10, 0]));
  pen.move(onSurface([10, 40]));
  pen.up(onSurface([10, 40]));
  const nodes_ZYX = [...skeleton.nodes.values()].map((node) => node.point_ZYX);
  for (const [z, y, x] of nodes_ZYX) assert.ok(Math.abs(z - height(x)) < 1, `${z},${y},${x}`);
  assert.equal(nodes_ZYX.length, 12);
});
