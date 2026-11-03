import assert from "node:assert/strict";
import { test } from "node:test";
import { DrawnTrees, treeMarks } from "../src/extensions/points/drawn.js";
import {
  bandMarks,
  commentMark,
  commentTags,
  markComment,
  runOf,
  stepRun,
  tagsComment,
} from "../src/extensions/points/runs.js";
import { addNodes, commit, emptySkeleton, newTree } from "../src/extensions/points/skeleton.js";

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

function marked() {
  const skeleton = chain();
  commit(skeleton, newTree(skeleton, []));
  const points_ZYX = [
    [0, 5, 0],
    [0, 5, 1],
  ];
  commit(
    skeleton,
    addNodes(skeleton, 2, points_ZYX, { from: null, connected: true, stroke: null, comment: "[KOLLESIS]", view: null }),
  );
  for (const [id, comment] of [
    [1, "[KOLLESIS]"],
    [2, "[KOLLESIS] torn"],
    [4, "[KOLLESIS]"],
    [5, "[BREAK]"],
  ])
    skeleton.nodes.get(id).comment = comment;
  return skeleton;
}

function runsOf(skeleton, shown = () => true) {
  const trees = new DrawnTrees();
  trees.join(skeleton, () => [1, 1, 1], null, 2);
  return trees.runs(shown, String(shown));
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

test("a node's tag reads from its comment in any case and writes back in upper case, first; [BREAK] is text", () => {
  assert.deepEqual(commentTags("[kollesis] torn [BREAK] edge"), { kollesis: true, text: "torn [BREAK] edge" });
  assert.equal(tagsComment({ kollesis: true, text: "torn [BREAK] edge" }), "[KOLLESIS] torn [BREAK] edge");
  assert.equal(tagsComment({ kollesis: false, text: "[BREAK] " }), "[BREAK]");
  assert.equal(tagsComment(commentTags(null)), null);
});

test("a node's mark is kollesis when its comment has the tag, none of [BREAK], and a mark's comment is its tag", () => {
  assert.deepEqual(["[break] torn [Kollesis]", "[BREAK] torn", "torn", null].map(commentMark), [
    "kollesis",
    null,
    null,
    null,
  ]);
  assert.deepEqual(["kollesis", null].map(markComment), ["[KOLLESIS]", null]);
});

test("a tree's runs end at a node of no mark, one of [BREAK] too, and a hidden tree has none", () => {
  const skeleton = marked();
  assert.deepEqual(runsOf(skeleton), [
    { tree: 1, mark: "kollesis", ids: [1, 2] },
    { tree: 1, mark: "kollesis", ids: [4] },
    { tree: 2, mark: "kollesis", ids: [6, 7] },
  ]);
  assert.deepEqual(
    runsOf(skeleton, (tree) => tree.id === 2).map((run) => run.ids),
    [[6, 7]],
  );
});

test("the run after node 3 starts at 4, before node 6 at 4, before 7 at 6, and none lies after 7 or before 1", () => {
  const skeleton = marked();
  const runs = runsOf(skeleton);
  const first = (selected, step) => stepRun(runs, skeleton, selected, step)?.ids[0] ?? null;
  assert.deepEqual(
    [first(3, 1), first(6, -1), first(7, -1), first(7, 1), first(1, -1), first(null, 1), first(null, -1)],
    [4, 4, 6, null, null, 1, 6],
  );
});

test("node 2 is in the first run, 7 in the third, and 3, marked with none, and 5, of [BREAK], in none", () => {
  const skeleton = marked();
  const runs = runsOf(skeleton);
  assert.deepEqual(
    [2, 7, 3, 5].map((id) => runOf(runs, skeleton, id)),
    [0, 2, -1, -1],
  );
});

test("a band joins the marked nodes an edge joins, of one mark, and a marked node joined to none of its own is a dot", () => {
  const skeleton = marked();
  const marks = new Map([1, 2, 4, 6, 7].map((id) => [id, commentMark(skeleton.nodes.get(id).comment)]));
  marks.set(5, "picked");
  const { marks: bands, ids_N } = bandMarks(skeleton, marks, 3, [treeMarks(skeleton, 1), treeMarks(skeleton, 2)]);
  assert.deepEqual([...ids_N], [1, 2, 4, 6, 7, 5]);
  assert.deepEqual([...bands.edges_E2], [0, 1, 3, 4, 2, 2, 5, 5]);
  const color = (at) => String(bands.colors_N3.subarray(3 * at, 3 * at + 3));
  assert.deepEqual([color(0) === color(2), color(0) === color(5), color(3) === color(4)], [true, false, true]);
  assert.deepEqual([[...bands.points_zyx_N3.subarray(6, 9)], bands.selected, bands.depth], [[0, 0, 3], -1, 3]);
});

test("a band follows a stroke's line where no edge joins its nodes, and not across a node of no mark", () => {
  const skeleton = stroked(Object.fromEntries([2, 3, 4, 5, 6, 7].map((id) => [id, "[KOLLESIS]"])));
  const marks = new Map([2, 3, 4, 5, 6, 7].map((id) => [id, "kollesis"]));
  const lines = (drawn) => [treeMarks(drawn, 1), treeMarks(drawn, 2)];
  assert.deepEqual([...bandMarks(skeleton, marks, 1, lines(skeleton)).marks.edges_E2], [0, 1, 1, 2, 3, 4, 5, 5]);
  const cut = stroked({ 5: "[KOLLESIS]" });
  assert.deepEqual([...bandMarks(cut, new Map([[5, "kollesis"]]), 1, lines(cut)).marks.edges_E2], [0, 0]);
});
