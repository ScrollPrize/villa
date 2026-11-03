import assert from "node:assert/strict";
import { test } from "node:test";
import { DrawnTrees, treeMarks } from "../src/extensions/points/drawn.js";
import { pressMarks } from "../src/extensions/points/marks.js";
import { addNodes, commit, deleteStroke, emptySkeleton, moveNode, newTree } from "../src/extensions/points/skeleton.js";

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

test("a tree's marks number its nodes in id order and join each edge once; with no press the live marks are the selected", () => {
  const skeleton = chain();
  const { marks, ids_N } = treeMarks(skeleton, 1);
  assert.deepEqual([...ids_N], [1, 2, 3, 4, 5]);
  assert.deepEqual([...marks.points_zyx_N3.subarray(9, 12)], [0, 0, 3]);
  assert.deepEqual([...marks.edges_E2], [0, 1, 1, 2, 2, 3, 3, 4]);
  const live = pressMarks(
    skeleton,
    () => [0, 0, 1],
    2,
    () => true,
    new Set(),
  );
  assert.deepEqual([[...live.ids_N], [...live.marks.edges_E2], live.marks.selected], [[5], [], 0]);
});

test("the marks of a stroke join its nodes in id order, not across to another stroke, and hide another tree's", () => {
  const skeleton = emptySkeleton();
  commit(skeleton, newTree(skeleton, []));
  commit(skeleton, newTree(skeleton, []));
  for (const [tree, stroke, count] of [
    [1, "a", 3],
    [1, "b", 2],
    [2, "c", 2],
  ]) {
    const points_ZYX = Array.from({ length: count }, (_, x) => [0, 0, x]);
    commit(
      skeleton,
      addNodes(skeleton, tree, points_ZYX, { from: null, connected: false, stroke, comment: null, view: null }),
    );
  }
  assert.deepEqual([...treeMarks(skeleton, 1).marks.edges_E2], [0, 1, 1, 2, 3, 4]);
  const trees = new DrawnTrees();
  const { lines } = trees.join(skeleton, () => [0, 0, 1], null, 2);
  assert.deepEqual([...lines.ids_N, ...lines.marks.edges_E2], [1, 2, 3, 4, 5, 6, 7, 0, 1, 1, 2, 3, 4, 5, 6]);
  assert.deepEqual([...trees.hidden((tree) => tree.id === 1, "tree 1", null).lines], [1, 1, 1, 1, 1, 0, 0]);
});

test("a stroke's node has its stroke's nodes before and after it as neighbours, a click's none, and the join keeps them", () => {
  const skeleton = emptySkeleton();
  commit(skeleton, newTree(skeleton, []));
  commit(skeleton, newTree(skeleton, []));
  for (const [tree, stroke, count] of [
    [1, "a", 3],
    [1, null, 1],
    [2, "c", 2],
  ]) {
    const points_ZYX = Array.from({ length: count }, (_, x) => [0, 0, x]);
    commit(
      skeleton,
      addNodes(skeleton, tree, points_ZYX, { from: null, connected: false, stroke, comment: null, view: null }),
    );
  }
  assert.deepEqual([...treeMarks(skeleton, 1).marks.neighbours_N2], [-1, 1, 0, 2, 1, -1, -1, -1]);
  const { lines, bands } = new DrawnTrees().join(skeleton, () => [0, 0, 1], null, 2);
  assert.deepEqual([...lines.marks.neighbours_N2], [-1, 1, 0, 2, 1, -1, -1, -1, -1, 5, 4, -1]);
  assert.equal(bands.marks.neighbours_N2, null);
});

test("the trees joined after edits to one tree are as a fresh join of them, and the same join when none changed", () => {
  const skeleton = stroked();
  const trees = new DrawnTrees();
  const color = (tree) => [tree.id, 0, 0];
  const first = trees.join(skeleton, color, 1, 2);
  skeleton.selected = 3;
  assert.equal(trees.join(skeleton, color, 1, 2), first);
  commit(skeleton, moveNode(skeleton, 8, [0, 3, 7]));
  commit(skeleton, deleteStroke(skeleton, 5));
  commit(skeleton, newTree(skeleton, []));
  const joined = trees.join(skeleton, color, 3, 2);
  const fresh = new DrawnTrees().join(skeleton, color, 3, 2);
  assert.notEqual(joined, first);
  for (const layer of ["lines", "bands"]) assert.deepEqual(joined[layer], fresh[layer]);
  assert.deepEqual(
    [...joined.lines.ids_N, ...joined.lines.marks.points_zyx_N3.subarray(15, 18)],
    [1, 2, 3, 4, 7, 8, 0, 3, 7],
  );
});

test("a press moving node 3 of the chain 1–5 hides it where the trees draw it, and draws 3 and its edges live", () => {
  const skeleton = chain();
  skeleton.selected = 3;
  const trees = new DrawnTrees();
  trees.join(skeleton, () => [0, 0, 1], null, 2);
  const ids = new Set([3]);
  assert.deepEqual([...trees.hidden(() => true, "all", ids).lines], [1, 1, 0, 1, 1]);
  const live = pressMarks(
    skeleton,
    () => [0, 0, 1],
    2,
    () => true,
    ids,
  );
  assert.deepEqual([[...live.ids_N], [...live.marks.edges_E2], live.marks.selected], [[3, 2, 4], [0, 1, 0, 2], 0]);
});
