import assert from "node:assert/strict";
import { test } from "node:test";
import {
  addNodes,
  apply,
  commit,
  deleteNode,
  deleteStroke,
  deleteTree,
  emptyPatch,
  emptySkeleton,
  lastNode,
  moveNodes,
  nearestNode,
  newTree,
  redo,
  selectNode,
  stepStroke,
  strokeOf,
  strokePlace,
  undo,
} from "../src/extensions/points/skeleton.js";
import { makeTarget, treeTarget } from "../src/extensions/points/targets.js";
import { dot_ZYX } from "../src/math.js";

function chain() {
  const skeleton = emptySkeleton();
  commit(skeleton, newTree(skeleton, []));
  const points_ZYX = [0, 1, 2, 3, 4].map((x) => [0, 0, x]);
  commit(
    skeleton,
    addNodes(skeleton, 1, points_ZYX, { from: null, connected: true, stroke: null, comment: null, view: null }),
  );
  return skeleton;
}

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

function scribbles() {
  const skeleton = emptySkeleton();
  const made = makeTarget(skeleton, at({}));
  commit(skeleton, made.patch);
  for (const [x, stroke] of [
    [0, "a"],
    [4, "b"],
    [8, "c"],
  ]) {
    const points_ZYX = [x, x + 1].map((value) => [0, 0, value]);
    commit(
      skeleton,
      addNodes(skeleton, made.tree, points_ZYX, { from: null, connected: false, stroke, comment: null, view: null }),
    );
  }
  return { skeleton, tree: made.tree };
}

function stroked(comments = {}) {
  const skeleton = emptySkeleton();
  commit(skeleton, newTree(skeleton, []));
  commit(skeleton, newTree(skeleton, []));
  const add = (tree, xs, fields) =>
    commit(
      skeleton,
      addNodes(
        skeleton,
        tree,
        xs.map((x) => [0, 0, x]),
        { from: null, connected: false, stroke: null, comment: null, view: null, ...fields },
      ),
    );
  add(1, [0], {});
  add(1, [1, 2, 3], { from: 1, connected: true, stroke: "a" });
  add(1, [4, 5], { stroke: "b" });
  add(1, [6], {});
  add(2, [7], { stroke: "b" });
  for (const [id, comment] of Object.entries(comments)) skeleton.nodes.get(Number(id)).comment = comment;
  return skeleton;
}

function restroke(skeleton, id, stroke) {
  const node = skeleton.nodes.get(id);
  apply(skeleton, { ...emptyPatch(), nodes: [[id, node, { ...node, stroke }]] });
}

test("a replay that takes out the node a press selected leaves the selection its patch gives, or none", () => {
  const place = (skeleton, stroke) =>
    addNodes(
      skeleton,
      1,
      stroke === null
        ? [[0, 1, 0]]
        : [
            [0, 1, 0],
            [0, 1, 1],
          ],
      {
        from: null,
        connected: false,
        stroke,
        comment: null,
        view: null,
      },
    );
  const edits = [
    [(skeleton) => deleteNode(skeleton, 7), 7, redo],
    [(skeleton) => deleteStroke(skeleton, 5), 6, redo],
    [(skeleton) => deleteTree(skeleton, 2, 1), 8, redo],
    [(skeleton) => place(skeleton, null), 9, undo],
    [(skeleton) => place(skeleton, "c"), 10, undo],
  ];
  const left = edits.map(([edit, pressed, replay]) => {
    const skeleton = stroked();
    skeleton.selected = 1;
    commit(skeleton, edit(skeleton));
    if (replay === redo) undo(skeleton);
    skeleton.selected = pressed;
    replay(skeleton);
    return skeleton.selected;
  });
  assert.deepEqual(left, [null, null, 7, 1, 1]);
});

test("deleting the selected middle of a chain leaves 1–2 and 4–5 unjoined in its tree, selecting 2; undo joins it", () => {
  const skeleton = chain();
  skeleton.selected = 3;
  const before = contents(skeleton);
  commit(skeleton, deleteNode(skeleton, 3));
  assert.deepEqual(
    [...skeleton.nodes.values()].map((node) => node.tree),
    [1, 1, 1, 1],
  );
  assert.deepEqual(skeleton.links.get(2), new Set([1]));
  assert.deepEqual(skeleton.links.get(4), new Set([5]));
  assert.equal(skeleton.links.has(3), false);
  assert.equal(skeleton.selected, 2);
  assert.equal(undo(skeleton), true);
  assert.deepEqual(contents(skeleton), before);
  assert.equal(redo(skeleton), true);
  assert.equal(skeleton.nodes.has(3), false);
});

test("a stroke is its nodes in its tree, and a node a click placed is a stroke of one", () => {
  const skeleton = stroked();
  assert.deepEqual(
    [strokeOf(skeleton, 3), strokeOf(skeleton, 5), strokeOf(skeleton, 1), strokeOf(skeleton, 8)],
    [[2, 3, 4], [5, 6], [1], [8]],
  );
});

test("a stroke's place counts its tree's strokes begun up to it: a is 2 of 4, the click 7 4 of 4, and 8 1 of 1", () => {
  const skeleton = stroked();
  assert.deepEqual(
    [3, 7, 8].map((id) => strokePlace(skeleton, id)),
    [
      { place: 2, count: 4, nodes: 3 },
      { place: 4, count: 4, nodes: 1 },
      { place: 1, count: 1, nodes: 1 },
    ],
  );
});

test("a stroke's place follows edits: its first node taken out, a stroke moved to tree 2 and each undone", () => {
  const skeleton = stroked();
  const places = (...ids) => ids.map((id) => Object.values(strokePlace(skeleton, id)).join(" "));
  commit(skeleton, deleteNode(skeleton, 2));
  const cut = [...places(4), stepStroke(skeleton, 1, 1, 1)];
  commit(skeleton, moveNodes(skeleton, [5, 6], 2));
  const moved = places(7, 5, 8);
  undo(skeleton);
  const back = places(7, 8);
  undo(skeleton);
  assert.deepEqual(
    [cut, moved, back, places(3), lastNode(skeleton, 1)],
    [["2 4 2", [3, 4]], ["3 3 1", "1 2 2", "2 2 1"], ["4 4 1", "1 1 1"], ["2 4 3"], 7],
  );
});

test("a step from a stroke goes to the stroke begun next or last before, a click's node one of them, and none past an end", () => {
  const skeleton = stroked();
  const step = (id, by) => stepStroke(skeleton, 1, id, by);
  assert.deepEqual(
    [step(1, 1), step(3, 1), step(4, -1), step(6, 1), step(6, -1), step(7, -1)],
    [[2, 3, 4], [5, 6], [1], [7], [2, 3, 4], [5, 6]],
  );
  assert.deepEqual(
    [step(7, 1), step(1, -1), stepStroke(skeleton, 2, 8, 1), stepStroke(skeleton, 2, 8, -1)],
    [null, null, null, null],
  );
});

test("from no node a step goes to the tree's first stroke or its last, and in an empty tree to none", () => {
  const skeleton = stroked();
  commit(skeleton, newTree(skeleton, []));
  assert.deepEqual(
    [stepStroke(skeleton, 1, null, 1), stepStroke(skeleton, 1, null, -1), stepStroke(skeleton, 3, null, 1)],
    [[1], [7], null],
  );
});

test("deleting stroke a takes its nodes and edges and selects the click before it; one undo puts it back", () => {
  const skeleton = stroked();
  skeleton.selected = 3;
  skeleton.branchpoints = [3, 6];
  const before = contents(skeleton);
  commit(skeleton, deleteStroke(skeleton, 3));
  assert.deepEqual([[...skeleton.nodes.keys()], skeleton.links.size], [[1, 5, 6, 7, 8], 0]);
  assert.deepEqual([skeleton.selected, skeleton.branchpoints], [1, [6]]);
  undo(skeleton);
  assert.deepEqual(contents(skeleton), before);
});

test("deleting a stroke selects the one before at its last node, else the one after at its first, else none", () => {
  const skeleton = stroked();
  skeleton.selected = 6;
  commit(skeleton, deleteStroke(skeleton, 6));
  assert.deepEqual([[...skeleton.nodes.keys()], skeleton.selected], [[1, 2, 3, 4, 7, 8], 4]);
  commit(skeleton, deleteStroke(skeleton, 1));
  skeleton.selected = 2;
  commit(skeleton, deleteStroke(skeleton, 2));
  assert.deepEqual(skeleton.selected, 7);
  skeleton.selected = 8;
  commit(skeleton, deleteStroke(skeleton, 8));
  assert.deepEqual(skeleton.selected, null);
});

test("deleting a stroke selects the stroke begun before it, though an edge joins the next or it ends past it", () => {
  const skeleton = stroked();
  const c = { from: 6, connected: true, stroke: "c", comment: null, view: null };
  commit(
    skeleton,
    addNodes(
      skeleton,
      1,
      [
        [0, 0, 8],
        [0, 0, 9],
      ],
      c,
    ),
  );
  skeleton.selected = 5;
  commit(skeleton, deleteStroke(skeleton, 5));
  const around = stroked();
  restroke(around, 7, "a");
  around.selected = 5;
  commit(around, deleteStroke(around, 5));
  assert.deepEqual([skeleton.selected, around.selected], [4, 7]);
});

test("a new tree of stroke c holds c alone, the selection and the rest kept, as one edit undo takes back", () => {
  const { skeleton, tree } = scribbles();
  selectNode(skeleton, 6);
  const before = contents(skeleton);
  const patch = newTree(skeleton, strokeOf(skeleton, 6));
  commit(skeleton, patch);
  const members = (id) => [...skeleton.members.get(id)].sort((a, b) => a - b);
  assert.deepEqual(
    [patch.trees[0][0], skeleton.trees.get(2).name, members(2), members(tree), skeleton.selected],
    [2, "Class 2", [5, 6], [1, 2, 3, 4], 6],
  );
  assert.deepEqual(treeTarget(skeleton, 2, at({})), at({ tree: 2 }));
  undo(skeleton);
  assert.deepEqual(contents(skeleton), before);
  commit(skeleton, newTree(skeleton, []));
  assert.deepEqual([skeleton.trees.size, skeleton.selected], [2, null]);
});

test("deleting a tree selects the last node of the tree given, none of an empty one, and none for null", () => {
  const skeleton = stroked();
  restroke(skeleton, 7, "a");
  skeleton.selected = 8;
  commit(skeleton, deleteTree(skeleton, 2, 1));
  const before = skeleton.selected;
  undo(skeleton);
  const back = skeleton.selected;
  commit(skeleton, deleteTree(skeleton, 1, 2));
  const after = skeleton.selected;
  commit(skeleton, newTree(skeleton, []));
  const empty = lastNode(skeleton, 3);
  commit(skeleton, deleteTree(skeleton, 2, 3));
  const none = skeleton.selected;
  undo(skeleton);
  commit(skeleton, deleteTree(skeleton, 2, null));
  assert.deepEqual([before, back, after, empty, none, skeleton.selected], [6, 8, 8, null, null, null]);
});

test("undoing a tree's deletion puts it back at its place, and a tree made goes last", () => {
  const skeleton = stroked();
  commit(skeleton, newTree(skeleton, []));
  commit(skeleton, deleteTree(skeleton, 2, 1));
  undo(skeleton);
  const trees = [...skeleton.trees.keys()];
  commit(skeleton, deleteTree(skeleton, 3, 2));
  assert.deepEqual([trees, skeleton.selected], [[1, 2, 3], 8]);
});

test("a press takes the nearest node along the pane within reach, of those within Depth of its plane's half voxel", () => {
  const skeleton = emptySkeleton();
  commit(skeleton, newTree(skeleton, []));
  const points_ZYX = [
    [0, 0, 3],
    [2.5, 0, 0.5],
    [1, 0, -3.2],
  ];
  commit(
    skeleton,
    addNodes(skeleton, 1, points_ZYX, { from: null, connected: false, stroke: null, comment: null, view: null }),
  );
  const all = () => true;
  assert.equal(nearestNode(skeleton, [0, 0, 0], [1, 0, 0], 1.5, 3.5, all), 1);
  assert.equal(nearestNode(skeleton, [0, 0, -0.5], [1, 0, 0], 1.5, 3.5, all), 3);
  assert.equal(nearestNode(skeleton, [0, 0, 0], [1, 0, 0], 1.5, 2, all), null);
  assert.equal(nearestNode(skeleton, [0, 0, 0], [1, 0, 0], 2, 3.5, all), 2);
  commit(skeleton, newTree(skeleton, []));
  commit(
    skeleton,
    addNodes(skeleton, 2, [[0, 0, 1]], { from: null, connected: false, stroke: null, comment: null, view: null }),
  );
  assert.equal(nearestNode(skeleton, [0, 0, 0], [1, 0, 0], 1.5, 3.5, all), 4);
  assert.equal(
    nearestNode(skeleton, [0, 0, 0], [1, 0, 0], 1.5, 3.5, (tree) => tree.id === 1),
    1,
  );
});

test("at Depth 0 on a tilted plane a press takes a node on it or a rounding error off it, not one a voxel off it", () => {
  const normal_ZYX = [Math.cos(Math.PI / 6), 0, Math.sin(Math.PI / 6)];
  const along_ZYX = [-Math.sin(Math.PI / 6), 0, Math.cos(Math.PI / 6)];
  const centre_ZYX = [43903.17, 13999.2, 20510.63];
  const on = (steps, off) => centre_ZYX.map((value, axis) => value + steps * along_ZYX[axis] + off * normal_ZYX[axis]);
  const skeleton = emptySkeleton();
  commit(skeleton, newTree(skeleton, []));
  const points_ZYX = [on(10, 0), on(30, 0).map(Math.fround), on(50, 1)];
  commit(
    skeleton,
    addNodes(skeleton, 1, points_ZYX, { from: null, connected: false, stroke: null, comment: null, view: null }),
  );
  // Float32, as the marks hold a node's point, puts it 2 thousandths of a voxel off the plane.
  assert.ok(Math.abs(dot_ZYX(points_ZYX[1], normal_ZYX) - dot_ZYX(centre_ZYX, normal_ZYX)) > 1e-4);
  const all = () => true;
  assert.equal(nearestNode(skeleton, on(10.5, 0), normal_ZYX, 0, 4, all), 1);
  assert.equal(nearestNode(skeleton, on(30.5, 0), normal_ZYX, 0, 4, all), 2);
  assert.equal(nearestNode(skeleton, on(50.5, 0), normal_ZYX, 0, 4, all), null);
  assert.equal(nearestNode(skeleton, on(50.5, 0), normal_ZYX, 0.5, 4, all), 3);
});
