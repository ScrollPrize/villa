import assert from "node:assert/strict";
import { test } from "node:test";
import { Aim } from "../src/extensions/points/aim.js";
import { moveScribbles } from "../src/extensions/points/classes.js";
import { Mode } from "../src/extensions/points/mode.js";
import {
  addNodes,
  commit,
  emptySkeleton,
  newTree,
  redo,
  selectNode,
  strokeOf,
  undo,
} from "../src/extensions/points/skeleton.js";
import { makeTarget, otherSide, switchAir, targetTree, windingStep } from "../src/extensions/points/targets.js";
import { pointsView } from "../src/extensions/points/view.js";

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

function placed(targets, point_ZYX = [0, 0, 0]) {
  const skeleton = emptySkeleton();
  for (const target of targets) {
    let tree = targetTree(skeleton, target);
    if (tree === null) {
      const made = makeTarget(skeleton, target);
      commit(skeleton, made.patch);
      tree = made.tree;
    }
    commit(
      skeleton,
      addNodes(skeleton, tree, [point_ZYX], { from: null, connected: false, stroke: null, comment: null, view: null }),
    );
  }
  return skeleton;
}

function modeOf(skeleton) {
  const aim = new Aim(null, () => {}, assert.fail);
  const mode = new Mode(aim);
  const changed = () => {
    aim.follow(skeleton);
    mode.follow(skeleton);
  };
  return { aim, mode, changed };
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

test("a picked scribble's class is the target through a move and its undo, while the selection stays in it", () => {
  const ink = at({ role: "ink", fiber: null });
  const skeleton = emptySkeleton();
  const made = makeTarget(skeleton, ink);
  commit(skeleton, made.patch);
  const points_ZYX = [0, 1, 2].map((x) => [0, 0, x]);
  commit(
    skeleton,
    addNodes(skeleton, made.tree, points_ZYX, { from: null, connected: false, stroke: "a", comment: null, view: null }),
  );
  commit(
    skeleton,
    addNodes(skeleton, made.tree, [[0, 0, 3]], { from: 3, connected: true, stroke: null, comment: null, view: null }),
  );
  const { aim, mode, changed } = modeOf(skeleton);
  const up = { ...ink, winding: 1 };
  selectNode(skeleton, 2);
  mode.pickNode(skeleton, 2);
  const targets = [aim.target];
  commit(skeleton, moveScribbles(skeleton, [2], up).patch);
  changed();
  targets.push(aim.target, aim.previous);
  undo(skeleton);
  changed();
  targets.push(aim.target, mode.picked());
  selectNode(skeleton, 3);
  changed();
  targets.push(mode.picked());
  redo(skeleton);
  changed();
  targets.push(mode.picked(), aim.target);
  undo(skeleton);
  selectNode(skeleton, 4);
  changed();
  targets.push(mode.picked());
  assert.deepEqual(targets, [ink, up, ink, ink, 2, 3, 3, up, null]);
});

test("Ctrl+presses and Enter keep scribbles through the steps, a change of class moves those alone, and none selected drops them", () => {
  const { skeleton, tree } = scribbles();
  const { mode, changed } = modeOf(skeleton);
  const picks = [];
  const seen = () => picks.push([mode.picked(), mode.pickedIds(skeleton)?.toSorted((a, b) => a - b) ?? null]);
  const toggle = (id) => {
    mode.togglePick(skeleton, id);
    selectNode(skeleton, id);
    changed();
    seen();
  };
  selectNode(skeleton, 1);
  mode.setEditing(true, skeleton);
  toggle(4);
  toggle(5);
  toggle(3);
  const moved = mode.move(skeleton, at({ side: "verso" }));
  commit(skeleton, moved.patch);
  changed();
  seen();
  const left = skeleton.nodes.get(3).tree === tree;
  mode.keep(skeleton);
  seen();
  selectNode(skeleton, 2);
  changed();
  seen();
  toggle(6);
  selectNode(skeleton, null);
  changed();
  seen();
  assert.deepEqual(
    [moved.said, left, picks],
    [
      "2 scribbles, 4 nodes, are W0 › verso › fiber 1.",
      true,
      [
        [4, [1, 2, 3, 4]],
        [5, [1, 2, 3, 4, 5, 6]],
        [3, [1, 2, 5, 6]],
        [3, [1, 2, 5, 6]],
        [3, [1, 2, 3, 4, 5, 6]],
        [2, [1, 2, 3, 4, 5, 6]],
        [6, [1, 2, 3, 4]],
        [null, null],
      ],
    ],
  );
});

test("in Edit Shift+A moves a fiber's scribble into its winding's air and back, and ] the air's class picked whole a gap up", () => {
  const { skeleton } = scribbles();
  const { aim, mode, changed } = modeOf(skeleton);
  const moved = (next) => {
    const { patch, said } = mode.move(skeleton, next);
    commit(skeleton, patch);
    changed();
    return said;
  };
  selectNode(skeleton, 3);
  mode.setEditing(true, skeleton);
  const said = [moved(switchAir(aim.target))];
  const seen = [aim.target, strokeOf(skeleton, 3).map((id) => skeleton.nodes.get(id).tree)];
  said.push(moved(switchAir(aim.target)));
  seen.push(aim.target, skeleton.nodes.get(3).tree);
  said.push(moved(switchAir(aim.target)));
  mode.switchPick(skeleton);
  said.push(moved(windingStep(skeleton, aim.target, 1, mode.whole)));
  seen.push(aim.target, skeleton.groups.get(skeleton.trees.get(skeleton.nodes.get(4).tree).group).name);
  assert.deepEqual(said, [
    "Scribble of 2 nodes is W0–W+1 › air.",
    "Scribble of 2 nodes is W0 › recto › fiber 1.",
    "Scribble of 2 nodes is W0–W+1 › air.",
    "Class of 1 scribble, 2 nodes, is W+1–W+2 › air.",
  ]);
  assert.deepEqual(seen, [at({ air: true }), [2, 2], at({}), 1, at({ winding: 1, air: true }), "W+1–W+2"]);
});

test("Edit picks the selected node's scribble and lets go of Join, and Draw, or Escape with Mark, of the pick", () => {
  const ink = at({ role: "ink", fiber: null });
  const skeleton = placed([ink, { ...ink, winding: 100 }]);
  const { aim, mode } = modeOf(skeleton);
  selectNode(skeleton, 2);
  mode.join(skeleton);
  const seen = [mode.joining];
  mode.setEditing(true, skeleton);
  seen.push(mode.joining, mode.picked(), aim.target);
  mode.setEditing(false, skeleton);
  seen.push(mode.picked());
  mode.setEditing(true, skeleton);
  aim.setMark("kollesis");
  mode.release();
  seen.push(mode.editing, mode.picked(), aim.target.mark);
  assert.deepEqual(seen, [0, null, 2, { ...ink, winding: 100 }, null, false, null, null]);
});

test("a class picked whole stays picked in each of its scribbles, moves whole, follows it, and Edit narrows it", () => {
  const { skeleton } = scribbles();
  const { aim, mode, changed } = modeOf(skeleton);
  selectNode(skeleton, 2);
  mode.setEditing(true, skeleton, true);
  selectNode(skeleton, 4);
  changed();
  const seen = [mode.picked(), mode.pickedIds(skeleton).sort((a, b) => a - b)];
  const moved = mode.move(skeleton, otherSide(skeleton, aim.target, true));
  commit(skeleton, moved.patch);
  changed();
  seen.push(moved.said, aim.target, mode.move(skeleton, aim.target).said);
  mode.setEditing(true, skeleton, false);
  seen.push(mode.pickedIds(skeleton), mode.move(skeleton, at({ side: "verso", role: "ink", fiber: null })).said);
  commit(skeleton, mode.move(skeleton, at({ side: "verso", role: "ink", fiber: null })).patch);
  changed();
  selectNode(skeleton, 1);
  mode.setEditing(true, skeleton, true);
  seen.push(mode.move(skeleton, at({ side: "verso", role: "ink", fiber: null })).said);
  selectNode(skeleton, null);
  changed();
  seen.push(mode.picked(), mode.whole);
  assert.deepEqual(seen, [
    4,
    [1, 2, 3, 4, 5, 6],
    "Class of 3 scribbles, 6 nodes, is W0 › verso › fiber 1.",
    at({ side: "verso" }),
    "This class is W0 › verso › fiber 1 already.",
    [3, 4],
    "Scribble of 2 nodes is W0 › verso › ink.",
    "Class of 2 scribbles, 4 nodes, joins W0 › verso › ink.",
    null,
    false,
  ]);
});

test("C makes an empty class in Draw and scribble c's in Edit, and is refused with none picked or its class whole", () => {
  const { skeleton } = scribbles();
  const { mode } = modeOf(skeleton);
  const drawn = mode.makeClass(skeleton);
  selectNode(skeleton, null);
  mode.setEditing(true, skeleton);
  const seen = [drawn.tree, drawn.patch.nodes.length, drawn.said, mode.makeClass(skeleton).said];
  selectNode(skeleton, 6);
  mode.setEditing(true, skeleton, true);
  seen.push(mode.makeClass(skeleton).said);
  mode.setEditing(true, skeleton);
  const made = mode.makeClass(skeleton);
  seen.push(made.tree, made.patch.nodes.length, made.said);
  assert.deepEqual(seen, [
    2,
    0,
    null,
    "Click a scribble to change its class; Escape goes back to Draw.",
    "W0 › recto › fiber 1 is a class of its own already.",
    2,
    2,
    "Scribble of 2 nodes is Class 2.",
  ]);
});

test("Name refuses a picked scribble's class that holds others and a role name its side has, and names the rest", () => {
  const { skeleton, tree } = scribbles();
  commit(skeleton, newTree(skeleton, strokeOf(skeleton, 6)));
  const made = makeTarget(skeleton, at({ fiber: 2 }));
  commit(skeleton, made.patch);
  const { mode } = modeOf(skeleton);
  const named = (id, name) => {
    const { patch, said } = mode.name(skeleton, id, name);
    return patch === null ? said : patch.trees[0][2].name;
  };
  const seen = [named(made.tree, "fiber 1"), named(tree, "fiber 9")];
  selectNode(skeleton, null);
  mode.setEditing(true, skeleton);
  seen.push(named(tree, "fiber 9"));
  selectNode(skeleton, 1);
  mode.setEditing(true, skeleton);
  seen.push(named(tree, "fiber 9"));
  mode.setEditing(true, skeleton, true);
  seen.push(named(tree, "fiber 9"));
  selectNode(skeleton, 5);
  mode.setEditing(true, skeleton);
  seen.push(named(2, "lone"), named(2, "fiber 1"));
  assert.deepEqual(seen, [
    "W0 › recto › fiber 1 is another class already.",
    "fiber 9",
    "Click a scribble to change its class; Escape goes back to Draw.",
    "This class holds 2 scribbles; C gives this scribble a class of its own, or Escape and Name renames them all.",
    "fiber 9",
    "lone",
    "fiber 1",
  ]);
});

test("S centres on a node of the picker's class, else goes to its last scribble as Shift+; does, and Shift+' to its first", () => {
  const { skeleton } = scribbles();
  const made = makeTarget(skeleton, at({ fiber: 2 }));
  commit(skeleton, made.patch);
  const points_ZYX = [12, 13, 14].map((x) => [0, 0, x]);
  commit(
    skeleton,
    addNodes(skeleton, made.tree, points_ZYX, { from: null, connected: false, stroke: "d", comment: null, view: null }),
  );
  const { aim, mode } = modeOf(skeleton);
  const went = (selected) => {
    selectNode(skeleton, selected);
    return [0, 1, -1].map((step) => mode.scribbleStep(skeleton, step));
  };
  const [a, c] = [
    { node: 1, on: 2 },
    { node: 5, on: 6 },
  ];
  const seen = [went(4), went(8), went(null)];
  aim.to(at({ fiber: 2 }));
  seen.push(went(4)[0]);
  assert.deepEqual(seen, [[{ node: 4, on: 4 }, c, a], [c, a, c], [c, a, c], { node: 7, on: 8 }]);
});

test("S and steps stay at an end and in a class of no nodes; in Edit they go through the pick's class as picked, else its own", () => {
  const { skeleton } = scribbles();
  const { aim, mode, changed } = modeOf(skeleton);
  const said = (step) => mode.scribbleStep(skeleton, step).said;
  selectNode(skeleton, 6);
  const seen = [said(1)];
  aim.to(at({ fiber: 2 }));
  seen.push(said(0), said(1));
  commit(skeleton, newTree(skeleton, []));
  aim.to(at({ tree: 2 }));
  seen.push(said(-1));
  aim.to(at({}));
  selectNode(skeleton, 4);
  mode.setEditing(true, skeleton);
  commit(skeleton, moveScribbles(skeleton, [4], at({ role: "no ink", fiber: null })).patch);
  changed();
  seen.push(aim.target.role, ...[0, 1, -1].map((step) => mode.scribbleStep(skeleton, step)));
  commit(skeleton, newTree(skeleton, strokeOf(skeleton, 6)));
  selectNode(skeleton, 6);
  mode.setEditing(true, skeleton);
  undo(skeleton);
  changed();
  seen.push(said(1), mode.scribbleStep(skeleton, -1));
  assert.deepEqual(seen, [
    "W0 › recto › fiber 1 has no scribble after this one.",
    "W0 › recto › fiber 2 has no nodes yet.",
    "W0 › recto › fiber 2 has no nodes yet.",
    "Class 2 has no nodes yet.",
    "no ink",
    { node: 4, on: 4 },
    { node: 5, on: 6 },
    { node: 1, on: 2 },
    "W0 › recto › fiber 1 has no scribble after this one.",
    { node: 1, on: 2 },
  ]);
});

test("the note counts classes and nodes and names the class its first node makes, and the label names the pick", () => {
  const { skeleton } = scribbles();
  const { aim, mode, changed } = modeOf(skeleton);
  const fields = { watched: null, aim, mode, active: true, unsaved: false, kept: "kept", clear: true, depth: 0 };
  const given = { ...fields, skeleton, drawn: skeleton, runs: [], labelColor: () => [0, 0, 1] };
  const shown = (more) => pointsView({ ...given, ...more }, "");
  const draw = shown({ unsaved: true });
  aim.to(at({ fiber: 2 }));
  const seen = [draw.view.note, draw.view.scribble, draw.view.classes, shown({ kept: "being kept" }).view.note];
  aim.to(at({}));
  mode.setEditing(true, skeleton);
  const picked = shown({});
  mode.switchPick(skeleton);
  const whole = shown({});
  selectNode(skeleton, null);
  changed();
  seen.push(...[draw, picked, whole, shown({})].map(({ label }) => label.text));
  mode.setEditing(false, skeleton);
  const watched = placed([at({})]);
  const elsewhere = shown({ skeleton: null, watched, drawn: watched });
  const empty = emptySkeleton();
  seen.push(draw.view.runRow({ tree: 1, mark: "kollesis", ids: [3, 4] }), elsewhere.view.note, elsewhere.label);
  seen.push(shown({ skeleton: empty }).view.note, shown({ skeleton: empty, active: false }).view.note);
  mode.join(skeleton);
  seen.push(shown({}).view.note);
  assert.deepEqual(seen, [
    "1 class, 6 nodes, not saved, kept in this browser.",
    { place: 3, count: 3, nodes: 2 },
    [{ name: "W0", trees: [{ id: 1, label: "W0 › recto › fiber 1 (6 nodes)" }] }],
    "1 class, 6 nodes, being kept in this browser; W0 › recto › fiber 2 is made with its first node.",
    "W0 › recto › fiber 1",
    "Edit: W0 › recto › fiber 1",
    "Edit class: W0 › recto › fiber 1",
    "Edit: click a scribble",
    { label: "W0 › recto › fiber 1 › kollesis", count: "2 nodes" },
    "Another tab has this CT's nodes open; the 1 node kept shows here, to edit once it lets go of them.",
    null,
    "Click a slice pane to place a node, drag to scribble; Space pans.",
    "",
    "Join W0: choose the winding it is, then Join again; Escape cancels.",
  ]);
});

test("while another tab holds the CT, a marked run of the classes it keeps is named from them: W+1 › verso › ink", () => {
  const watched = placed([at({ winding: 1, side: "verso", role: "ink", fiber: null })]);
  const { aim, mode } = modeOf(watched);
  const run = { tree: 1, mark: "kollesis", ids: [1] };
  const fields = { skeleton: null, watched, drawn: watched, aim, mode, active: false, unsaved: false, kept: null };
  const { view } = pointsView({ ...fields, clear: false, depth: 0, runs: [run], labelColor: () => [1, 1, 1] }, "");
  assert.deepEqual(
    [view.runRow(run), view.currentRun],
    [{ label: "W+1 › verso › ink › kollesis", count: "1 node" }, -1],
  );
});
