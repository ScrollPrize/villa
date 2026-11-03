import assert from "node:assert/strict";
import { test } from "node:test";
import { Aim } from "../src/extensions/points/aim.js";
import { Mode } from "../src/extensions/points/mode.js";
import { writeNml } from "../src/extensions/points/nml.js";
import { insertWinding, joinRegions, newRegion, regionLabel, regionOf } from "../src/extensions/points/regions.js";
import { addNodes, apply, commit, emptyPatch, emptySkeleton, redo, undo } from "../src/extensions/points/skeleton.js";
import { makeTarget, targetLabel, targetTree, treeTarget } from "../src/extensions/points/targets.js";

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

function edgesOf(skeleton) {
  const edges = [];
  for (const [id, linked] of skeleton.links) for (const other of linked) if (id < other) edges.push([id, other]);
  return edges.sort((a, b) => a[0] - b[0] || a[1] - b[1]).map(([a, b]) => `${a}-${b}`);
}

function membersOf(skeleton, tree) {
  return [...(skeleton.members.get(tree) ?? [])].sort((a, b) => a - b).join(" ");
}

function regions(windings) {
  const skeleton = emptySkeleton();
  for (const winding of windings)
    for (const fields of [{ fiber: 1 }, { fiber: 2 }, { role: "ink", fiber: null }, { role: "no ink", fiber: null }]) {
      const made = makeTarget(skeleton, at({ winding, ...fields }));
      commit(skeleton, made.patch);
      const z = winding % 100;
      const points_ZYX =
        fields.role === "ink"
          ? [
              [z, 0, 0],
              [z, 0, 1],
            ]
          : [[z, 0, 0]];
      const stroke = fields.role === "ink" ? "s" : null;
      commit(
        skeleton,
        addNodes(skeleton, made.tree, points_ZYX, { from: null, connected: true, stroke, comment: null, view: null }),
      );
    }
  return skeleton;
}

function treesOf(skeleton) {
  return [...skeleton.trees.keys()].map(
    (id) => `${targetLabel(skeleton, treeTarget(skeleton, id, at()))}: ${membersOf(skeleton, id)}`,
  );
}

test("region 0 holds W-49 to W+50, region 1 W+51 to W+150, region -1 W-149 to W-50, and W? none", () => {
  assert.deepEqual([-50, -49, 0, 50, 51, 150, 151, null].map(regionOf), [-1, 0, 0, 0, 1, 1, 2, null]);
  assert.deepEqual([0, 103, -50].map(regionLabel), ["W-49 to W+50", "W+51 to W+150", "W-149 to W-50"]);
});

test("inserting a winding at W0 renames W0 and W+1 up by one, not W-1, W? or a group named W+2 inside a side", () => {
  const skeleton = emptySkeleton();
  const names = [
    [1, "W-1", null],
    [2, "W0", null],
    [3, "recto", 2],
    [4, "W+1", null],
    [5, "W?", null],
    [6, "W+2", 3],
  ];
  apply(skeleton, { ...emptyPatch(), groups: names.map(([id, name, parent]) => [id, null, { id, name, parent }]) });
  const before = contents(skeleton);
  commit(skeleton, insertWinding(skeleton, 0).patch);
  assert.deepEqual(
    [...skeleton.groups.values()].map((group) => group.name),
    ["W-1", "W+1", "recto", "W+2", "W?", "W+2"],
  );
  undo(skeleton);
  assert.deepEqual(contents(skeleton), before);
});

test("Insert at W+48 renames W+48 and W+49 but not W+51, is refused while W+50 has a group, and at W+51 takes W+100", () => {
  const skeleton = emptySkeleton();
  const names = [
    [1, "W+48", null],
    [2, "W+49", null],
    [3, "W+51", null],
    [4, "W+100", null],
  ];
  apply(skeleton, { ...emptyPatch(), groups: names.map(([id, name, parent]) => [id, null, { id, name, parent }]) });
  commit(skeleton, insertWinding(skeleton, 48).patch);
  assert.deepEqual(
    [...skeleton.groups.values()].map((group) => group.name),
    ["W+49", "W+50", "W+51", "W+100"],
  );
  const refused = { patch: null, said: "Insert would push the last winding of W-49 to W+50 out of that region." };
  assert.deepEqual([insertWinding(skeleton, 48), insertWinding(skeleton, 50)], [refused, refused]);
  commit(skeleton, insertWinding(skeleton, 51).patch);
  assert.deepEqual(
    [...skeleton.groups.values()].map((group) => group.name),
    ["W+49", "W+50", "W+52", "W+101"],
  );
});

test("Insert moves air with its lower winding, W+49–W+50 to W+50–W+51, which then refuses it, and New region is past it", () => {
  const skeleton = placed([at({ winding: 49, air: true }), at({ winding: -1, air: true })]);
  commit(skeleton, insertWinding(skeleton, 0).patch);
  assert.deepEqual(
    [...skeleton.groups.values()].map((group) => group.name),
    ["W+50–W+51", "W-1–W0"],
  );
  const above = "Insert would push the air above the last winding of W-49 to W+50 out of that region.";
  assert.deepEqual(insertWinding(skeleton, 0), { patch: null, said: above });
  assert.deepEqual(newRegion(skeleton, at({ air: true })), at({ winding: 100, air: true }));
});

test("Join moves W+50–W+51's air with W+50, into W+150–W+151's, renames W-1–W0's, and refuses it past W+150", () => {
  const skeleton = placed([
    at({ winding: 50, air: true }),
    at({ winding: -1, air: true }),
    at({ winding: 150, air: true }),
    at({ winding: 0 }),
  ]);
  const before = writeNml(skeleton, "scan", LEVEL);
  assert.deepEqual(joinRegions(skeleton, 0, 101), { patch: null, said: "W+50 would be W+151, outside W+51 to W+150." });
  const { patch, said } = joinRegions(skeleton, 0, 100);
  commit(skeleton, patch);
  assert.equal(said, "Joined W-1 to W+50 as W+99 to W+150.");
  assert.deepEqual(treesOf(skeleton), ["W+99–W+100 › air: 2", "W+150–W+151 › air: 1 3", "W+100 › recto › fiber 1: 4"]);
  const groups = [...skeleton.groups.values()].map(({ id, name, parent }) => `${id} ${name} in ${parent}`);
  assert.deepEqual(groups, ["2 W+99–W+100 in null", "3 W+150–W+151 in null", "4 W+100 in null", "5 recto in 4"]);
  undo(skeleton);
  assert.equal(writeNml(skeleton, "scan", LEVEL), before);
});

test("New region is W0 with none holding a tree, past W-120 W0, past W+2 and W+120 W+200, side and role kept", () => {
  const ink = at({ side: "verso", role: "ink", fiber: null });
  const regions = (...windings) => newRegion(placed(windings.map((winding) => at({ winding }))), ink);
  assert.deepEqual(
    [newRegion(emptySkeleton(), ink), regions(-120), regions(2, 120, null)],
    [at({ side: "verso", role: "ink", fiber: null, winding: 0 }), { ...ink, winding: 0 }, { ...ink, winding: 200 }],
  );
  assert.deepEqual(newRegion(placed([at({ fiber: 3 })]), at({ fiber: 3 })), at({ winding: 100 }));
});

test("W+101 joined onto W+1 moves W+100 to W+102 down 100, inks and no inks merged, fibers 1 and 2 as 3 and 4", () => {
  const skeleton = regions([0, 1, 2, 100, 101, 102]);
  const before = writeNml(skeleton, "scan", LEVEL);
  const { patch, said } = joinRegions(skeleton, 101, 1);
  commit(skeleton, patch);
  const joined = writeNml(skeleton, "scan", LEVEL);
  assert.equal(said, "Joined W+100 to W+102 as W0 to W+2.");
  assert.deepEqual(treesOf(skeleton), [
    "W0 › recto › fiber 1: 1",
    "W0 › recto › fiber 2: 2",
    "W0 › recto › ink: 3 4 18 19",
    "W0 › recto › no ink: 5 20",
    "W+1 › recto › fiber 1: 6",
    "W+1 › recto › fiber 2: 7",
    "W+1 › recto › ink: 8 9 23 24",
    "W+1 › recto › no ink: 10 25",
    "W+2 › recto › fiber 1: 11",
    "W+2 › recto › fiber 2: 12",
    "W+2 › recto › ink: 13 14 28 29",
    "W+2 › recto › no ink: 15 30",
    "W0 › recto › fiber 3: 16",
    "W0 › recto › fiber 4: 17",
    "W+1 › recto › fiber 3: 21",
    "W+1 › recto › fiber 4: 22",
    "W+2 › recto › fiber 3: 26",
    "W+2 › recto › fiber 4: 27",
  ]);
  const groups = [...skeleton.groups.values()].map(({ id, name, parent }) => `${id} ${name} in ${parent}`);
  assert.deepEqual(groups, [
    "1 W0 in null",
    "2 recto in 1",
    "3 W+1 in null",
    "4 recto in 3",
    "5 W+2 in null",
    "6 recto in 5",
  ]);
  const ink = [8, 9, 23, 24].map((id) => skeleton.nodes.get(id));
  assert.deepEqual(
    [ink.map(({ stroke }) => stroke), ink.map(({ point_ZYX }) => point_ZYX.join(",")), edgesOf(skeleton)],
    [
      ["s", "s", "s~2", "s~2"],
      ["1,0,0", "1,0,1", "1,0,0", "1,0,1"],
      ["3-4", "8-9", "13-14", "18-19", "23-24", "28-29"],
    ],
  );
  undo(skeleton);
  assert.equal(writeNml(skeleton, "scan", LEVEL), before);
  redo(skeleton);
  assert.equal(writeNml(skeleton, "scan", LEVEL), joined);
});

test("W+101 joined onto W+1 renames W+102, moves W+101's verso, keeps its recto for a tree of no role, and refuses", () => {
  const skeleton = regions([0, 1, 100, 101]);
  commit(skeleton, makeTarget(skeleton, at({ winding: 101, side: "verso" })).patch);
  commit(skeleton, makeTarget(skeleton, at({ winding: 102, role: "ink", fiber: null })).patch);
  commit(skeleton, { ...emptyPatch(), trees: [[19, null, { id: 19, name: "notes", color_RGB: null, group: 8 }]] });
  const refusals = [
    [101, 120],
    [101, null],
    [100, 50],
    [250, 1],
  ].map(([from, to]) => joinRegions(skeleton, from, to));
  assert.deepEqual(refusals, [
    { patch: null, said: "W+101 and W+120 are both in W+51 to W+150; join to another region." },
    { patch: null, said: "W? is in no region; choose the winding to join to." },
    { patch: null, said: "W+101 would be W+51, outside W-49 to W+50." },
    { patch: null, said: "W+151 to W+250 holds no class to join." },
  ]);
  commit(skeleton, joinRegions(skeleton, 101, 1).patch);
  assert.deepEqual(treesOf(skeleton), [
    "W0 › recto › fiber 1: 1",
    "W0 › recto › fiber 2: 2",
    "W0 › recto › ink: 3 4 13 14",
    "W0 › recto › no ink: 5 15",
    "W+1 › recto › fiber 1: 6",
    "W+1 › recto › fiber 2: 7",
    "W+1 › recto › ink: 8 9 18 19",
    "W+1 › recto › no ink: 10 20",
    "W0 › recto › fiber 3: 11",
    "W0 › recto › fiber 4: 12",
    "W+1 › recto › fiber 3: 16",
    "W+1 › recto › fiber 4: 17",
    "W+1 › verso › fiber 1: ",
    "W+2 › recto › ink: ",
    "notes: ",
  ]);
  const groups = [...skeleton.groups.values()].map(({ id, name, parent }) => `${id} ${name} in ${parent}`);
  assert.deepEqual(groups, [
    "1 W0 in null",
    "2 recto in 1",
    "3 W+1 in null",
    "4 recto in 3",
    "7 W+101 in null",
    "8 recto in 7",
    "9 verso in 3",
    "10 W+2 in null",
    "11 recto in 10",
  ]);
});

test("Join keeps its first press's winding, lets go there again, refuses W? and one region, and moves at another", () => {
  const skeleton = regions([0, 1, 100, 101]);
  const { aim, mode } = modeOf(skeleton);
  const pressed = (winding) => {
    aim.to(at({ winding }));
    const { patch, said } = mode.join(skeleton);
    return [patch !== null, said, mode.joining];
  };
  assert.deepEqual([null, 101, 101, 101, 100, 1].map(pressed), [
    [false, "W? is in no region; choose a winding to join.", null],
    [false, "", 101],
    [false, "", null],
    [false, "", 101],
    [false, "W+101 and W+100 are both in W+51 to W+150; join to another region.", 101],
    [true, "Joined W+100 to W+101 as W0 to W+1.", null],
  ]);
});
