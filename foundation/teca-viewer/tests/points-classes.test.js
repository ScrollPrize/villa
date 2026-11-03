import assert from "node:assert/strict";
import { test } from "node:test";
import { moveClass, moveScribbles, newClass } from "../src/extensions/points/classes.js";
import { addNodes, commit, emptySkeleton, newTree, redo, undo } from "../src/extensions/points/skeleton.js";
import { makeTarget, targetLabel, targetTree, treeTarget } from "../src/extensions/points/targets.js";

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

function edgesOf(skeleton) {
  const edges = [];
  for (const [id, linked] of skeleton.links) for (const other of linked) if (id < other) edges.push([id, other]);
  return edges.sort((a, b) => a[0] - b[0] || a[1] - b[1]).map(([a, b]) => `${a}-${b}`);
}

function membersOf(skeleton, tree) {
  return [...(skeleton.members.get(tree) ?? [])].sort((a, b) => a - b).join(" ");
}

function treesOf(skeleton) {
  return [...skeleton.trees.keys()].map(
    (id) => `${targetLabel(skeleton, treeTarget(skeleton, id, at()))}: ${membersOf(skeleton, id)}`,
  );
}

test("scribble b, drawn Connected from scribble a's end, moves to W+1's ink, made for it, with its edges but 3–4", () => {
  const ink = at({ role: "ink", fiber: null });
  const skeleton = emptySkeleton();
  const made = makeTarget(skeleton, ink);
  commit(skeleton, made.patch);
  const draw = (xs, from, stroke) => {
    const points_ZYX = xs.map((x) => [0, 0, x]);
    commit(
      skeleton,
      addNodes(skeleton, made.tree, points_ZYX, { from, connected: true, stroke, comment: null, view: null }),
    );
  };
  draw([0, 1, 2], null, "a");
  draw([3, 4, 5], 3, "b");
  skeleton.nodes.get(5).comment = "[BREAK] torn";
  skeleton.branchpoints = [5];
  skeleton.selected = 5;
  const [before, joined] = [contents(skeleton), edgesOf(skeleton)];
  const up = { ...ink, winding: 1 };
  commit(skeleton, moveScribbles(skeleton, [5], up).patch);
  const moved = contents(skeleton);
  assert.deepEqual(
    [joined, edgesOf(skeleton), membersOf(skeleton, 1), membersOf(skeleton, 2), targetTree(skeleton, up)],
    [["1-2", "2-3", "3-4", "4-5", "5-6"], ["1-2", "2-3", "4-5", "5-6"], "1 2 3", "4 5 6", 2],
  );
  const groups = [...skeleton.groups.values()].map((group) => group.name).join(" ");
  const node = { id: 5, tree: 2, point_ZYX: [0, 0, 4], radius: 1, stroke: "b", comment: "[BREAK] torn", view: null };
  assert.deepEqual(
    [groups, skeleton.nodes.get(5), skeleton.selected, skeleton.branchpoints],
    ["W0 recto W+1 recto", node, 5, [5]],
  );
  assert.deepEqual(moveScribbles(skeleton, [4], up), {
    patch: null,
    said: "This scribble is W+1 › recto › ink already.",
  });
  undo(skeleton);
  assert.deepEqual(contents(skeleton), before);
  redo(skeleton);
  assert.deepEqual(contents(skeleton), moved);
});

test("scribbles of two classes move as one edit, one in the class already staying, and C makes a class of two", () => {
  const skeleton = stroked();
  const said = [];
  for (const ids of [
    [2, 8],
    [3, 5],
  ]) {
    const moved = moveScribbles(skeleton, ids, at());
    said.push(moved.said);
    commit(skeleton, moved.patch);
  }
  said.push(moveScribbles(skeleton, [2, 5], at()).said, newClass(skeleton, [1, 7]).said);
  assert.deepEqual(
    [said, membersOf(skeleton, 3)],
    [
      [
        "2 scribbles, 4 nodes, are W0 › recto › fiber 1.",
        "Scribble of 2 nodes is W0 › recto › fiber 1.",
        "These scribbles are W0 › recto › fiber 1 already.",
        "2 scribbles, 2 nodes, are Class 4.",
      ],
      "2 3 4 5 6 8",
    ],
  );
});

test("a click's node moves alone, cutting its edge to the scribble after it, and a class a move empties stays", () => {
  const skeleton = stroked();
  commit(skeleton, moveScribbles(skeleton, [8], at()).patch);
  commit(skeleton, moveScribbles(skeleton, [1], at()).patch);
  const trees = [1, 2, 3].map((tree) => membersOf(skeleton, tree));
  assert.deepEqual(
    [[...skeleton.trees.keys()], trees, edgesOf(skeleton), skeleton.selected],
    [[1, 2, 3], ["2 3 4 5 6 7", "", "1 8"], ["2-3", "3-4"], 8],
  );
});

test("scribbles move from recto's ink to verso's, made for it, from ink into the side's no ink, and from a fiber to ink", () => {
  const skeleton = emptySkeleton();
  const tree = (fields) => {
    const made = makeTarget(skeleton, at(fields));
    commit(skeleton, made.patch);
    return made.tree;
  };
  const [ink, noInk, fiber] = [tree({ role: "ink", fiber: null }), tree({ role: "no ink", fiber: null }), tree({})];
  for (const [into, xs, stroke] of [
    [ink, [0, 1], "a"],
    [ink, [2, 3], "b"],
    [noInk, [4, 5], "n"],
    [fiber, [6, 7], "f"],
  ]) {
    const points_ZYX = xs.map((x) => [0, 0, x]);
    commit(
      skeleton,
      addNodes(skeleton, into, points_ZYX, { from: null, connected: true, stroke, comment: null, view: null }),
    );
  }
  const before = contents(skeleton);
  commit(skeleton, moveScribbles(skeleton, [1], at({ side: "verso", role: "ink", fiber: null })).patch);
  commit(skeleton, moveScribbles(skeleton, [3], at({ role: "no ink", fiber: null })).patch);
  commit(skeleton, moveScribbles(skeleton, [7], at({ role: "ink", fiber: null })).patch);
  const moved = contents(skeleton);
  assert.deepEqual(
    [treesOf(skeleton), edgesOf(skeleton), [1, 3, 5, 7].map((id) => skeleton.nodes.get(id).stroke)],
    [
      ["W0 › recto › ink: 7 8", "W0 › recto › no ink: 3 4 5 6", "W0 › recto › fiber 1: ", "W0 › verso › ink: 1 2"],
      ["1-2", "3-4", "5-6", "7-8"],
      ["a", "b", "n", "f"],
    ],
  );
  for (const _ of [1, 2, 3]) undo(skeleton);
  assert.deepEqual(contents(skeleton), before);
  for (const _ of [1, 2, 3]) redo(skeleton);
  assert.deepEqual(contents(skeleton), moved);
});

test("a class moved whole where none is takes the place itself, as recto fiber 1 to verso fiber 3, then verso's ink", () => {
  const { skeleton, tree } = scribbles();
  const before = contents(skeleton);
  const groups = () => [...skeleton.groups.values()].map((group) => `${group.id} ${group.name} ${group.parent}`);
  commit(skeleton, moveClass(skeleton, tree, at({ side: "verso", fiber: 3 })).patch);
  const seen = [[...skeleton.trees.keys()], treesOf(skeleton), groups(), skeleton.trees.get(tree).color_RGB];
  commit(skeleton, moveClass(skeleton, tree, at({ side: "verso", role: "ink", fiber: null })).patch);
  seen.push(treesOf(skeleton), groups().length, skeleton.trees.get(tree).color_RGB, skeleton.selected);
  assert.deepEqual(seen, [
    [tree],
    ["W0 › verso › fiber 3: 1 2 3 4 5 6"],
    ["1 W0 null", "2 recto 1", "3 verso 1"],
    null,
    ["W0 › verso › ink: 1 2 3 4 5 6"],
    3,
    [1, 0.1, 0.1],
    6,
  ]);
  assert.deepEqual(moveClass(skeleton, tree, at({ side: "verso", role: "ink", fiber: null })), {
    patch: null,
    said: "This class is W0 › verso › ink already.",
  });
  undo(skeleton);
  undo(skeleton);
  assert.deepEqual(contents(skeleton), before);
});

test("a fiber moved whole where its number is another class's says so, and not where it is free or its own", () => {
  const { skeleton, tree } = scribbles();
  commit(skeleton, makeTarget(skeleton, at({ winding: 2 })).patch);
  assert.deepEqual(
    [at({ winding: 2, fiber: 2 }), at({ winding: 1 }), at({ fiber: 2 })].map(
      (target) => moveClass(skeleton, tree, target).said,
    ),
    [
      "Class of 3 scribbles, 6 nodes, is W+2 › recto › fiber 2 (fiber 1 is taken there).",
      "Class of 3 scribbles, 6 nodes, is W+1 › recto › fiber 1.",
      "Class of 3 scribbles, 6 nodes, is W0 › recto › fiber 2.",
    ],
  );
});

test("a class moved whole into one that has nodes joins it, edges and scribbles kept apart, and goes; undo gives both", () => {
  const skeleton = stroked();
  const before = contents(skeleton);
  commit(skeleton, moveClass(skeleton, 1, { ...at(), tree: 2 }).patch);
  const moved = contents(skeleton);
  const strokes = [5, 6, 8].map((id) => skeleton.nodes.get(id).stroke);
  assert.deepEqual(
    [[...skeleton.trees.keys()], membersOf(skeleton, 2), edgesOf(skeleton), strokes, skeleton.selected],
    [[2], "1 2 3 4 5 6 7 8", ["1-2", "2-3", "3-4"], ["b~2", "b~2", "b"], 8],
  );
  undo(skeleton);
  assert.deepEqual([contents(skeleton), [...skeleton.trees.keys()]], [before, [1, 2]]);
  redo(skeleton);
  assert.deepEqual(contents(skeleton), moved);
});
