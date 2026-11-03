import assert from "node:assert/strict";
import { test } from "node:test";
import { Aim } from "../src/extensions/points/aim.js";
import { addNodes, commit, emptySkeleton, newTree } from "../src/extensions/points/skeleton.js";
import {
  classList,
  classNeighbour,
  classStep,
  gapOf,
  inWinding,
  makeTarget,
  newFiber,
  otherSide,
  pickerTarget,
  readTarget,
  switchAir,
  targetChoices,
  targetLabel,
  targetTree,
  treeLabel,
  treeTarget,
  windingStep,
  withRole,
} from "../src/extensions/points/targets.js";

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

test("a target's tree is made with its winding and side groups, and a second role on that side reuses them", () => {
  const skeleton = emptySkeleton();
  const fiber = at({ winding: 1, side: "verso", sheet: -1, fiber: 3 });
  const made = makeTarget(skeleton, fiber);
  assert.deepEqual(made.patch.groups, [
    [1, null, { id: 1, name: "W+1", parent: null }],
    [2, null, { id: 2, name: "verso S-1", parent: 1 }],
  ]);
  assert.deepEqual(made.patch.trees, [[1, null, { id: 1, name: "fiber 3", color_RGB: null, group: 2 }]]);
  commit(skeleton, made.patch);
  assert.equal(targetTree(skeleton, fiber), 1);
  assert.equal(targetLabel(skeleton, fiber), "W+1 › verso S-1 › fiber 3");
  const ink = makeTarget(skeleton, { ...fiber, role: "ink", fiber: null });
  assert.deepEqual(ink.patch.groups, []);
  assert.deepEqual(ink.patch.trees, [[2, null, { id: 2, name: "ink", color_RGB: [1, 0.1, 0.1], group: 2 }]]);
  commit(skeleton, ink.patch);
  assert.deepEqual(treeTarget(skeleton, 2, at({})), { ...fiber, role: "ink", fiber: null });
});

test("a target's label names its unknowns, and its mark, a role's or a tree's, is its fourth level", () => {
  const skeleton = emptySkeleton();
  commit(skeleton, newTree(skeleton, []));
  assert.equal(targetLabel(skeleton, at({ winding: null, side: null, sheet: null })), "W? › side ? S? › fiber 1");
  assert.equal(
    targetLabel(skeleton, at({ winding: 1, fiber: 7, mark: "kollesis" })),
    "W+1 › recto › fiber 7 › kollesis",
  );
  assert.equal(targetLabel(skeleton, { ...at({}), tree: 1, mark: "kollesis" }), "Class 1 › kollesis");
});

test("an address's target of no mark has none, of no air is no air, and one of mark break or air 1 is refused", () => {
  const { mark, air, ...bare } = at({ winding: 2 });
  assert.deepEqual([readTarget(bare), readTarget(at({ air: true }))], [at({ winding: 2 }), at({ air: true })]);
  assert.throws(() => readTarget(at({ mark: "break" })), { message: 'The address\'s Points class has mark "break".' });
  assert.throws(() => readTarget(at({ air: 1 })), { message: "The address's Points class has air 1." });
});

test("an address's target is refused naming its first bad field, and an Aim given one keeps none of the view", () => {
  assert.throws(() => readTarget(at({ fiber: 0 })), { message: "The address's Points class has fiber 0." });
  assert.throws(() => readTarget(at({ side: "left" })), { message: 'The address\'s Points class has side "left".' });
  const failures = [];
  const spacings = { fiber: 2, ink: 3, "no ink": 4, tree: 5 };
  const saved = { target: at({ winding: 2 }), previous: at({ fiber: 0 }), spacings, only: true };
  const aim = new Aim(
    saved,
    () => {},
    (error) => failures.push(error.message),
  );
  assert.deepEqual(failures, ["The address's Points class has fiber 0."]);
  assert.deepEqual([aim.target, aim.only, aim.spacing()], [at({}), false, 5]);
});

test("from recto ink, Q gives the fiber with the latest node, N fiber 3, T verso's fiber 1, ] W+1's ink", () => {
  const ink = at({ role: "ink", fiber: null });
  const skeleton = placed([at({ fiber: 1 }), at({ fiber: 2 }), at({ fiber: 1 }), ink]);
  assert.deepEqual(withRole(skeleton, ink, "fiber"), at({ fiber: 1 }));
  assert.deepEqual(newFiber(skeleton, ink), at({ fiber: 3 }));
  assert.deepEqual(otherSide(skeleton, at({ fiber: 2 })), at({ side: "verso", fiber: 1 }));
  assert.deepEqual(otherSide(skeleton, at({ side: null, fiber: 2 })), at({ fiber: 1 }));
  assert.deepEqual(windingStep(skeleton, ink, 1), { ...ink, winding: 1 });
  assert.equal(windingStep(skeleton, at({ winding: null }), -1), null);
});

test("a class moved whole keeps its fiber's number through T, ] and the picker's place, and takes a role as picked", () => {
  const skeleton = placed([at({ fiber: 3 }), at({ side: "verso", fiber: 1 }), at({ side: "verso", fiber: 2 })]);
  const third = at({ fiber: 3 });
  assert.deepEqual(
    [otherSide(skeleton, third), otherSide(skeleton, third, true), windingStep(skeleton, third, 1, true)],
    [at({ side: "verso", fiber: 2 }), at({ side: "verso", fiber: 3 }), at({ winding: 1, fiber: 3 })],
  );
  assert.deepEqual(
    [
      pickerTarget(skeleton, third, { side: "verso" }, true),
      pickerTarget(skeleton, third, { role: "ink" }, true),
      pickerTarget(skeleton, { ...third, tree: 1 }, { side: "verso" }, true),
    ],
    [at({ side: "verso", fiber: 3 }), at({ role: "ink", fiber: null }), at({ side: "verso", fiber: 2 })],
  );
});

test("a class moved whole takes a new fiber where its number is another's: fiber 2 two windings up is fiber 6 there", () => {
  const skeleton = placed([at({ fiber: 2 }), at({ winding: 2, fiber: 2 }), at({ winding: 2, fiber: 5 })]);
  const second = at({ fiber: 2 });
  assert.deepEqual(
    [windingStep(skeleton, second, 2, true), windingStep(skeleton, second, 1, true), otherSide(skeleton, second, true)],
    [at({ winding: 2, fiber: 6 }), at({ winding: 1, fiber: 2 }), at({ side: "verso", fiber: 2 })],
  );
});

test("the picker offers the sheets about the target's and S0, then S?, its side's fibers, its own, a new one, and its tree", () => {
  const skeleton = placed([at({ fiber: 1 }), at({ fiber: 3 }), at({ side: "verso", fiber: 2 })]);
  assert.deepEqual(targetChoices(skeleton, at({ winding: 1, sheet: 2 })), {
    winding: "W+1",
    sheets: [
      [-1, "S-1"],
      [0, "S0"],
      [1, "S+1"],
      [2, "S+2"],
      [3, "S+3"],
      [null, "S?"],
    ],
    fibers: [[1, "new fiber 1"]],
    tree: null,
  });
  const unknown = targetChoices(skeleton, at({ winding: null, sheet: null }));
  assert.deepEqual(
    [unknown.winding, unknown.sheets],
    [
      "W?",
      [
        [-1, "S-1"],
        [0, "S0"],
        [1, "S+1"],
        [null, "S?"],
      ],
    ],
  );
  assert.deepEqual(targetChoices(skeleton, at({ fiber: 2 })).fibers, [
    [1, "fiber 1"],
    [2, "fiber 2"],
    [3, "fiber 3"],
    [4, "new fiber 4"],
  ]);
  assert.deepEqual(targetChoices(skeleton, at({ role: "ink", fiber: null })).fibers, [
    [1, "fiber 1"],
    [3, "fiber 3"],
    [4, "new fiber 4"],
  ]);
  commit(skeleton, newTree(skeleton, []));
  assert.equal(targetChoices(skeleton, { ...at({}), tree: 4 }).tree, "Class 4");
});

test("a picked place keeps the role and the mark, a fiber the side's last drawn, and the place it is changes nothing", () => {
  const skeleton = placed([at({ fiber: 1 }), at({ fiber: 3 }), at({ side: "verso", fiber: 2 })]);
  const mark = "kollesis";
  const fiber = at({ mark });
  assert.deepEqual(pickerTarget(skeleton, fiber, { side: "verso" }), at({ side: "verso", fiber: 2, mark }));
  assert.equal(pickerTarget(skeleton, fiber, { side: "recto" }), fiber);
  assert.deepEqual(pickerTarget(skeleton, { ...fiber, tree: 9 }, { side: "recto" }), at({ fiber: 3, mark }));
  assert.deepEqual(pickerTarget(skeleton, fiber, { sheet: null }), at({ sheet: null, mark }));
  assert.deepEqual(pickerTarget(skeleton, fiber, { winding: null }), at({ winding: null, mark }));
  assert.deepEqual(pickerTarget(skeleton, fiber, { role: "ink" }), at({ role: "ink", fiber: null, mark }));
  assert.deepEqual(pickerTarget(skeleton, at({ role: "no ink", fiber: null }), { fiber: 7 }), at({ fiber: 7 }));
});

test("Fiber picked from ink or a tree is the side's last fiber drawn, 1 of 1 and 3, from fiber 3 fiber 3, else fiber 1", () => {
  const skeleton = placed([at({ fiber: 1 }), at({ fiber: 3 }), at({ fiber: 1 }), at({ side: "verso", fiber: 2 })]);
  const ink = at({ role: "ink", fiber: null, mark: "kollesis" });
  assert.deepEqual(pickerTarget(skeleton, ink, { role: "fiber" }), at({ fiber: 1, mark: "kollesis" }));
  assert.deepEqual(pickerTarget(skeleton, { ...at({ fiber: 3 }), tree: 9 }, { role: "fiber" }), at({ fiber: 1 }));
  const third = at({ fiber: 3 });
  assert.equal(pickerTarget(skeleton, third, { role: "fiber" }), third);
  assert.deepEqual(pickerTarget(skeleton, { ...ink, sheet: 1 }, { role: "fiber" }), at({ sheet: 1, mark: "kollesis" }));
});

test("Shift+A from W+1's verso fiber 2 is the air of W+1–W+2, which keeps the fiber that A, X, Q, N, T and Shift+A go from", () => {
  const fiber = at({ winding: 1, side: "verso", fiber: 2 });
  const skeleton = placed([fiber]);
  const air = switchAir(fiber);
  assert.deepEqual(
    [air, switchAir({ ...fiber, tree: 9 })],
    [
      { ...fiber, air: true },
      { ...fiber, air: true },
    ],
  );
  assert.equal(targetLabel(skeleton, air), "W+1–W+2 › air");
  assert.deepEqual(switchAir(air), fiber);
  assert.deepEqual(
    [withRole(skeleton, air, "ink"), withRole(skeleton, air, "no ink"), withRole(skeleton, air, "fiber")],
    [
      at({ winding: 1, side: "verso", role: "ink", fiber: null }),
      at({ winding: 1, side: "verso", role: "no ink", fiber: null }),
      fiber,
    ],
  );
  assert.deepEqual(
    [newFiber(skeleton, air), otherSide(skeleton, air)],
    [at({ winding: 1, side: "verso", fiber: 3 }), at({ winding: 1, fiber: 1 })],
  );
});

test("[ and ] from W0–W+1's air are W-1–W0's and W+1–W+2's, and W?–W?'s none; Air and a winding keep it air, a side not", () => {
  const skeleton = emptySkeleton();
  const air = at({ air: true });
  const [below, above] = [windingStep(skeleton, air, -1), windingStep(skeleton, air, 1)];
  assert.deepEqual([below, above], [at({ winding: -1, air: true }), at({ winding: 1, air: true })]);
  assert.deepEqual(
    [targetLabel(skeleton, below), targetLabel(skeleton, at({ winding: null, air: true, mark: "kollesis" }))],
    ["W-1–W0 › air", "W?–W? › air › kollesis"],
  );
  assert.equal(windingStep(skeleton, at({ winding: null, air: true }), 1), null);
  assert.deepEqual(pickerTarget(skeleton, air, { winding: null }), at({ winding: null, air: true }));
  assert.equal(pickerTarget(skeleton, air, { air: true }), air);
  assert.deepEqual(pickerTarget(skeleton, at({ role: "ink", fiber: null }), { air: true }), {
    ...air,
    role: "ink",
    fiber: null,
  });
  assert.deepEqual(
    [
      pickerTarget(skeleton, air, { side: "recto" }),
      pickerTarget(skeleton, air, { side: "verso" }),
      pickerTarget(skeleton, air, { role: "ink" }),
      pickerTarget(skeleton, air, { fiber: 2 }),
    ],
    [at({}), at({ side: "verso" }), at({ role: "ink", fiber: null }), at({ fiber: 2 })],
  );
});

test("air is made in a gap group of its own named for both windings, light blue, and its tree's target keeps the class", () => {
  const skeleton = placed([at({})]);
  const air = at({ winding: -1, side: "verso", air: true });
  const made = makeTarget(skeleton, air);
  assert.deepEqual(made.patch.groups, [[3, null, { id: 3, name: "W-1–W0", parent: null }]]);
  assert.deepEqual(made.patch.trees, [[2, null, { id: 2, name: "air", color_RGB: [0.05, 0.7, 1], group: 3 }]]);
  commit(skeleton, made.patch);
  assert.deepEqual(
    [
      targetTree(skeleton, air),
      targetTree(skeleton, at({ winding: -1, air: true })),
      targetTree(skeleton, at({ air: true })),
    ],
    [2, 2, null],
  );
  assert.deepEqual(
    [treeTarget(skeleton, 2, at({ fiber: 4, mark: "kollesis" })), treeTarget(skeleton, 1, air)],
    [at({ winding: -1, fiber: 4, mark: "kollesis", air: true }), at({})],
  );
  assert.deepEqual(
    [treeLabel(skeleton, 2), inWinding(skeleton, 2, -1), inWinding(skeleton, 2, 0)],
    ["W-1–W0 › air", true, false],
  );
  const groups = new Map(
    [
      [1, "W?–W?", null],
      [2, "W0–W+2", null],
      [3, "W+1–W+2", 1],
      [4, "W-2–W-1", null],
    ].map(([id, name, parent]) => [id, { id, name, parent }]),
  );
  assert.deepEqual(
    [1, 2, 3, 4, null].map((id) => gapOf(groups, id)),
    [{ winding: null }, null, null, { winding: -2 }, null],
  );
});

test("the Class list is by winding, W? last, then side, sheet, fiber, ink, no ink and the air, then Other as made", () => {
  const skeleton = placed([
    at({ winding: 1, role: "ink", fiber: null }),
    at({}),
    at({ air: true }),
    at({ side: "verso", fiber: 2 }),
    at({ winding: null }),
    at({ role: "no ink", fiber: null }),
    at({ fiber: 3 }),
    at({ sheet: 1 }),
    at({ side: null }),
  ]);
  commit(skeleton, newTree(skeleton, []));
  assert.deepEqual(classList(skeleton), [
    { name: "W0", ids: [2, 7, 6, 8, 4, 9, 3] },
    { name: "W+1", ids: [1] },
    { name: "W?", ids: [5] },
    { name: "Other", ids: [10] },
  ]);
  assert.deepEqual(
    [2, 1, 10].map((id) => classNeighbour(skeleton, id)),
    [7, 3, 5],
  );
});

test("a class step passes the classes with no nodes, goes from no class to an end, and stops past either end", () => {
  const skeleton = placed([at({ winding: 1 }), at({}), at({ side: "verso" })]);
  commit(skeleton, makeTarget(skeleton, at({ role: "ink", fiber: null })).patch);
  assert.deepEqual(
    [
      classStep(skeleton, 2, 1),
      classStep(skeleton, 3, -1),
      classStep(skeleton, 4, 1),
      classStep(skeleton, null, 1),
      classStep(skeleton, null, -1),
      classStep(skeleton, 1, 1),
      classStep(skeleton, 2, -1),
    ],
    [3, 2, 3, 2, 1, null, null],
  );
});

test("Follow is on for no address and one written before it, and kept off once turned off", () => {
  const kept = [];
  const aim = new Aim(null, (value) => kept.push(value), assert.fail);
  assert.equal(aim.following, true);
  aim.setFollowing(false);
  const { following, ...before } = kept.at(-1);
  assert.deepEqual([new Aim(kept.at(-1), () => {}, assert.fail).following, following], [false, false]);
  assert.equal(new Aim(before, () => {}, assert.fail).following, true);
});

test("going to the target before toggles between the last two targets, and each role keeps its own spacing", () => {
  const aim = new Aim(null, () => {}, assert.fail);
  const ink = at({ role: "ink", fiber: null });
  aim.setSpacing(2);
  aim.to(ink);
  aim.setSpacing(8);
  assert.equal(aim.to(aim.previous), true);
  assert.deepEqual([aim.target, aim.spacing()], [at({}), 2]);
  aim.to(aim.previous);
  assert.deepEqual([aim.target, aim.spacing()], [ink, 8]);
  aim.to(switchAir(ink));
  aim.setSpacing(3);
  aim.to(aim.previous);
  assert.deepEqual([aim.target, aim.spacing(), aim.spacings.air], [ink, 8, 3]);
});

test("an address's spacings with none of air give air the default, and keep the rest", () => {
  const spacings = { fiber: 2, ink: 3, "no ink": 4, tree: 6 };
  const aim = new Aim({ target: at({ air: true }), previous: at({}), spacings }, () => {}, assert.fail);
  assert.deepEqual([aim.spacing(), aim.spacings], [5, { fiber: 2, ink: 3, "no ink": 4, air: 5, tree: 6 }]);
});
