import assert from "node:assert/strict";
import { test } from "node:test";
import { fileName, readNml, readSaved, savedZip, writeNml } from "../src/extensions/points/nml.js";
import { addNodes, commit, deleteTree, emptySkeleton, newTree, undo } from "../src/extensions/points/skeleton.js";
import { makeTarget, targetTree } from "../src/extensions/points/targets.js";
import { unzipped, zipped } from "../src/zip.js";

const LEVEL = { shape_ZYX: [10, 20, 30], downsample_ZYX: [1, 1, 1], offset_ZYX: [0, 0, 0] };

const NML = `<?xml version="1.0" encoding="UTF-8"?>
<things>
  <meta name="writer" content="NmlWriter.scala"/>
  <parameters><experiment name="scan"/><scale x="1.0" y="1.0" z="1.0"/><activeNode id="8"/></parameters>
  <thing id="3" name="ink &amp; fibre" color.r="1" color.g="0.5" color.b="0" color.a="1.0">
    <nodes>
      <node id="7" radius="1.5" x="1" y="2" z="3" rotX="0.0"/>
      <node id="8" radius="1" x="4.25" y="2" z="3" teca.strokeId="a1"/>
    </nodes>
    <edges><edge source="8" target="7"/></edges>
  </thing>
  <thing id="4" name="noink"><nodes><node id="9" x="0" y="0" z="0"/></nodes><edges/></thing>
  <branchpoints><branchpoint id="8"/></branchpoints>
  <comments><comment node="9" content="edge of the sheet"/></comments>
</things>`;

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

const SCALES = [
  { ...LEVEL, scale_ZYX: [2.4, 2.4, 2.4] },
  { shape_ZYX: [5, 10, 15], downsample_ZYX: [2, 2, 2], offset_ZYX: [0, 0, 0], scale_ZYX: [4.8, 4.8, 4.8] },
  { shape_ZYX: [3, 5, 8], downsample_ZYX: [3, 4, 4], offset_ZYX: [1, 0, 0], scale_ZYX: [7.2, 9.6, 9.6] },
];

const MICROMETRES = ["micrometer", "micrometer", "micrometer"];

async function saved(skeleton, name, scales, units_ZYX) {
  const zip = await savedZip(skeleton, name, scales, units_ZYX);
  const files = await unzipped(new Uint8Array(await zip.arrayBuffer()));
  return files.map(({ name, bytes }) => ({ name, text: new TextDecoder().decode(bytes) }));
}

test("an NML names a role's tree in the fixed form, nests its groups, and reads back the same", () => {
  const skeleton = placed([
    at({ winding: 1, fiber: 7 }),
    at({ winding: -2, side: "verso", sheet: 1, role: "ink", fiber: null }),
    at({ winding: null, side: null, sheet: null, role: "no ink", fiber: null }),
    at({ side: "verso", fiber: 2 }),
  ]);
  commit(skeleton, newTree(skeleton, []));
  assert.deepEqual(
    [...skeleton.trees.values()].map((tree) => fileName(skeleton.groups, tree)),
    ["S0_W1_HZ_RECTO_FIBER_7", "S1_W-2_VERSO_SURFACE_INK", "SN_WN_SURFACE_NOINK", "S0_W0_VT_VERSO_FIBER_2", "Class 5"],
  );
  const text = writeNml(skeleton, "scan", LEVEL);
  assert.match(text, /<thing id="1" name="S0_W1_HZ_RECTO_FIBER_7" groupId="2">/);
  assert.match(text, /<group id="1" name="W\+1">\n\s*<group id="2" name="recto"\/>\n\s*<\/group>/);
  assert.deepEqual(contents(readNml(text, LEVEL)), contents(skeleton));
});

test("each gap's air is a tree in a group of the gap's name, as W0_W1_AIR, WN_WN_AIR above W?, and reads back as air", () => {
  const skeleton = placed([
    at({ air: true }),
    at({ winding: -2, side: "verso", air: true }),
    at({ winding: null, air: true }),
    at({}),
  ]);
  assert.deepEqual(
    [...skeleton.trees.values()].map((tree) => fileName(skeleton.groups, tree)),
    ["W0_W1_AIR", "W-2_W-1_AIR", "WN_WN_AIR", "S0_W0_HZ_RECTO_FIBER_1"],
  );
  const text = writeNml(skeleton, "scan", LEVEL);
  assert.match(
    text,
    /<thing id="1" name="W0_W1_AIR" color.r="0.05" color.g="0.7" color.b="1" color.a="1" groupId="1">/,
  );
  assert.match(
    text,
    /<group id="1" name="W0–W\+1"\/>\n\s*<group id="2" name="W-2–W-1"\/>\n\s*<group id="3" name="W\?–W\?"\/>/,
  );
  const opened = readNml(text, LEVEL);
  assert.deepEqual(contents(opened), contents(skeleton));
  assert.deepEqual(
    [...opened.trees.values()].map((tree) => tree.name),
    ["air", "air", "air", "fiber 1"],
  );
  assert.equal(readNml(text.replace('name="W0_W1_AIR"', 'name="W1_W2_AIR"'), LEVEL).trees.get(1).name, "W1_W2_AIR");
});

test("an NML's own groups and names are kept as they are, and a tree in a group it lacks is refused", () => {
  const groups = '<groups><group id="1" name="S0_W1"><group id="2" name="RECTO"/></group></groups></things>';
  const grouped = NML.replace('name="ink &amp; fibre"', 'name="S0_W1_HZ_RECTO_FIBER_7" groupId="2"').replace(
    "</things>",
    groups,
  );
  const skeleton = readNml(grouped, LEVEL);
  assert.deepEqual(skeleton.groups.get(2), { id: 2, name: "RECTO", parent: 1 });
  assert.equal(skeleton.trees.get(3).name, "S0_W1_HZ_RECTO_FIBER_7");
  assert.deepEqual(contents(readNml(writeNml(skeleton, "scan", LEVEL), LEVEL)), contents(skeleton));
  assert.throws(() => readNml(grouped.replace('groupId="2"', 'groupId="9"'), LEVEL), {
    message: "NML tree 3 is in group 9, which the file does not have.",
  });
});

test("an NML reads into trees, nodes at world ZYX, edges, a branchpoint and a comment, and writes back the same", () => {
  const skeleton = readNml(NML, LEVEL);
  assert.deepEqual(skeleton.trees.get(3), { id: 3, name: "ink & fibre", color_RGB: [1, 0.5, 0], group: null });
  assert.deepEqual(skeleton.trees.get(4), { id: 4, name: "noink", color_RGB: null, group: null });
  assert.deepEqual(skeleton.nodes.get(8), {
    id: 8,
    tree: 3,
    point_ZYX: [3, 2, 4.25],
    radius: 1,
    stroke: "a1",
    comment: null,
    view: null,
  });
  assert.equal(skeleton.nodes.get(7).radius, 1.5);
  assert.equal(skeleton.nodes.get(9).comment, "edge of the sheet");
  assert.deepEqual(skeleton.links.get(7), new Set([8]));
  assert.deepEqual(skeleton.branchpoints, [8]);
  assert.equal(skeleton.selected, 8);
  assert.deepEqual(contents(readNml(writeNml(skeleton, "scan", LEVEL), LEVEL)), contents(skeleton));
});

test("a node of a level of voxel 2 at offset 1 is at world offset + 2 × voxel + 0.5", () => {
  const level = { shape_ZYX: [10, 20, 30], downsample_ZYX: [2, 2, 2], offset_ZYX: [1, 1, 1] };
  assert.deepEqual(readNml(NML, level).nodes.get(8).point_ZYX, [7.5, 5.5, 10]);
});

test("an NML is refused for a node outside the CT, an edge between trees, and text outside a tag", () => {
  const small = { ...LEVEL, shape_ZYX: [3, 20, 30] };
  assert.throws(() => readNml(NML, small), { message: "NML node 7 at x, y, z 1, 2, 3 is outside this CT." });
  const across = NML.replace("<edges/>", '<edges><edge source="9" target="7"/></edges>');
  assert.throws(() => readNml(across, LEVEL), { message: "NML edge 9–7 does not join two nodes of one tree." });
  assert.throws(() => readNml(NML.replace("<nodes>", "<nodes>7"), LEVEL), { message: 'NML has "7" outside a tag.' });
});

test("a node placed on an xz pane keeps its pane and its normal to 3 decimals, z first; one from a file has neither", () => {
  const skeleton = readNml(NML, LEVEL);
  const view = { pane: "xz", normal_ZYX: [0.6000004, 0, -0.7999996] };
  commit(
    skeleton,
    addNodes(skeleton, 4, [[1, 1, 1]], { from: null, connected: false, stroke: null, comment: null, view }),
  );
  const text = writeNml(skeleton, "scan", LEVEL);
  assert.match(text, /<node id="10" radius="1" x="1" y="1" z="1" teca.pane="xz" teca.normal_ZYX="0.6,0,-0.8"\/>/);
  assert.doesNotMatch(text, /<node id="9"[^>]*teca/);
  const read = readNml(text, LEVEL);
  assert.deepEqual(
    [read.nodes.get(10).view, read.nodes.get(9).view],
    [{ pane: "xz", normal_ZYX: [0.6, 0, -0.8] }, null],
  );
  assert.equal(writeNml(read, "scan", LEVEL), text);
});

test("an NML node with teca.pane alone, a pane there is not, or a normal of two numbers is refused", () => {
  const node = (attributes) =>
    NML.replace('<node id="9" x="0" y="0" z="0"/>', `<node id="9" x="0" y="0" z="0" ${attributes}/>`);
  assert.throws(() => readNml(node('teca.pane="xy"'), LEVEL), { message: "NML node 9 has teca.pane alone." });
  assert.throws(() => readNml(node('teca.pane="3d" teca.normal_ZYX="1,0,0"'), LEVEL), {
    message: "NML node 9 has teca.pane 3d, not xy, zy, xz or surface.",
  });
  assert.throws(() => readNml(node('teca.pane="xy" teca.normal_ZYX="1,0"'), LEVEL), {
    message: "NML node 9 has teca.normal_ZYX 1,0, not three numbers.",
  });
});

test("an NML is refused when it leaves a tag open, closes one out of order or has a second root", () => {
  assert.throws(() => readNml(NML.replace("</things>", ""), LEVEL), { message: "NML leaves <things> open." });
  assert.throws(() => readNml(NML.replace("</nodes>", ""), LEVEL), {
    message: "NML has </thing> where <nodes> is open.",
  });
  assert.throws(() => readNml(`${NML}<things/>`, LEVEL), { message: "NML has a second root, <things>." });
  assert.throws(() => readNml("<nml/>", LEVEL), { message: "The file has no NML <things>." });
});

test("an NML's trees 3, 1 and 2 open, list and save in its order, and undoing tree 1's deletion keeps it", () => {
  const skeleton = readNml(
    `<things>
      <thing id="3"><nodes><node id="1" x="0" y="0" z="0"/></nodes></thing>
      <thing id="1"><nodes><node id="2" x="1" y="0" z="0"/></nodes></thing>
      <thing id="2"><nodes><node id="3" x="2" y="0" z="0"/></nodes></thing>
    </things>`,
    LEVEL,
  );
  const opened = [...skeleton.trees.keys()];
  const saved = [...writeNml(skeleton, "scan", LEVEL).matchAll(/<thing id="(\d+)"/g)].map((match) => Number(match[1]));
  commit(skeleton, deleteTree(skeleton, 1, 3));
  const selected = skeleton.selected;
  undo(skeleton);
  assert.deepEqual([opened, saved, selected, [...skeleton.trees.keys()]], [[3, 1, 2], [3, 1, 2], 1, [3, 1, 2]]);
});

test("a node at finest voxel 3, 2, 1 is in each coarser level's file where the scaled export writes it", async () => {
  const skeleton = readNml(NML, LEVEL);
  // A separate scaled writer of the finest file, run on these levels, wrote these lines.
  const levels = [
    [
      '<scale x="4.8" y="4.8" z="4.8" unit="micrometer"/>',
      '<node id="7" radius="1.5" x="0.25" y="0.75" z="1.25"/>',
      '<node id="8" radius="1" x="1.875" y="0.75" z="1.25" teca.strokeId="a1"/>',
      '<node id="9" radius="1" x="-0.25" y="-0.25" z="-0.25"/>',
    ],
    [
      '<scale x="9.6" y="9.6" z="7.199999999999999" unit="micrometer"/>',
      '<node id="7" radius="1.5" x="-0.125" y="0.125" z="0.3333333432674408"/>',
      '<node id="8" radius="1" x="0.6875" y="0.125" z="0.3333333432674408" teca.strokeId="a1"/>',
      '<node id="9" radius="1" x="-0.375" y="-0.375" z="-0.6666666865348816"/>',
    ],
  ];
  const files = (await saved(skeleton, "scan", SCALES, MICROMETRES)).map(({ text }) => text);
  const changed = (text) =>
    text
      .split("\n")
      .filter((line) => /<(?:node|scale) /.test(line))
      .map((line) => line.trim());
  const kept = (text) => text.split("\n").filter((line) => !/<(?:node|scale) /.test(line));
  assert.deepEqual(files.slice(1).map(changed), levels);
  for (const text of files.slice(1)) assert.deepEqual(kept(text), kept(files[0]));
  const unitless = (await saved(skeleton, "scan", SCALES, [null, null, null])).map(({ text }) => text);
  assert.deepEqual(
    unitless.slice(1).map(changed),
    levels.map((lines) => lines.slice(1)),
  );
});

test("Save's zip holds the finest level's NML and one for each coarser level, named by their level", async () => {
  const skeleton = readNml(NML, LEVEL);
  const files = await saved(skeleton, "scan", SCALES, MICROMETRES);
  assert.deepEqual(
    files.map(({ name }) => name),
    ["scan.nml", "scan-edited-level-1.nml", "scan-edited-level-2.nml"],
  );
  assert.equal(files[0].text, writeNml(skeleton, "scan", LEVEL));
  const named = (await saved(skeleton, "a:b", SCALES.slice(0, 2), MICROMETRES)).map(({ name }) => name);
  assert.deepEqual(named, ["a:b.nml", "a_b-edited-level-1.nml"]);
});

test("Open… of Save's zip gives the trees Save had, as of their NML, and a zip with no finest NML is refused", async () => {
  const skeleton = readNml(NML, LEVEL);
  const zip = await savedZip(skeleton, "scan", SCALES, MICROMETRES);
  const opened = await readSaved(new Uint8Array(await zip.arrayBuffer()), "scan-points.zip", SCALES, MICROMETRES);
  assert.deepEqual(contents(opened), contents(skeleton));
  const nml = new TextEncoder().encode(writeNml(skeleton, "scan", LEVEL));
  assert.deepEqual(contents(await readSaved(nml, "scan.nml", SCALES, MICROMETRES)), contents(skeleton));
  const files = (names) => names.map((name) => ({ name, bytes: new TextEncoder().encode(NML) }));
  const refused = async (names) =>
    readSaved(
      new Uint8Array(await (await zipped(files(names), async () => {}, true)).arrayBuffer()),
      "scan-points.zip",
      SCALES,
      MICROMETRES,
    );
  const beside = "where Points reads one NML not named -level-<n>, with nothing beside it but -level-<n>.nml files.";
  await assert.rejects(refused(["notes.txt", "scan-edited-level-1.nml"]), {
    message: `The zip holds notes.txt, scan-edited-level-1.nml, ${beside}`,
  });
  await assert.rejects(refused(["scan.nml", "notes.txt"]), { message: `The zip holds scan.nml, notes.txt, ${beside}` });
  await assert.rejects(refused(["a.nml", "b.nml", "c", "d", "e", "f"]), {
    message: `The zip holds a.nml, b.nml, c, d, 2 more, ${beside}`,
  });
});

test("Open… refuses a zip whose level 2 has node 8 at z 9, naming its line 11, and one of a level the CT lacks", async () => {
  const skeleton = readNml(NML, LEVEL);
  const zip = await savedZip(skeleton, "scan", SCALES, MICROMETRES);
  const files = await unzipped(new Uint8Array(await zip.arrayBuffer()));
  const rezipped = async (edit) =>
    new Uint8Array(await (await zipped(edit(files), async () => {}, true)).arrayBuffer());
  const edited = (entries) =>
    entries.map(({ name, bytes }) => {
      const text = new TextDecoder().decode(bytes);
      if (name !== "scan-edited-level-2.nml") return { name, bytes };
      return { name, bytes: new TextEncoder().encode(text.replace('z="0.3333333432674408" teca', 'z="9" teca')) };
    });
  await assert.rejects(readSaved(await rezipped(edited), "scan-points.zip", SCALES, MICROMETRES), {
    message:
      "The zip's scan-edited-level-2.nml is not the level 2 Save makes of scan.nml: its line 11 is " +
      '<node id="8" radius="1" x="0.6875" y="0.125" z="9" teca.strokeId="a1"/>, where Save writes ' +
      '<node id="8" radius="1" x="0.6875" y="0.125" z="0.3333333432674408" teca.strokeId="a1"/>. Points reads ' +
      "scan.nml alone and would drop the change.",
  });
  const twoLevels = await rezipped((entries) => entries.slice(0, 2));
  assert.deepEqual(contents(await readSaved(twoLevels, "scan-points.zip", SCALES, MICROMETRES)), contents(skeleton));
  await assert.rejects(
    readSaved(await rezipped((entries) => entries), "scan-points.zip", SCALES.slice(0, 2), MICROMETRES),
    {
      message: "The zip's scan-edited-level-2.nml is of level 2, and this CT's coarsest is 1.",
    },
  );
});

test("Open… refuses a coarser level's NML alone, whose voxels it would read as the finest level's", async () => {
  const skeleton = emptySkeleton();
  commit(skeleton, newTree(skeleton, []));
  const points_ZYX = [
    [3, 2, 1],
    [5, 6, 7],
  ];
  commit(
    skeleton,
    addNodes(skeleton, 1, points_ZYX, { from: null, connected: false, stroke: null, comment: null, view: null }),
  );
  const zip = await savedZip(skeleton, "scan", SCALES, MICROMETRES);
  const [, level] = await unzipped(new Uint8Array(await zip.arrayBuffer()));
  await assert.rejects(readSaved(level.bytes, level.name, SCALES, MICROMETRES), {
    message:
      "scan-edited-level-1.nml is in a coarser level's voxels, which Points would read as the finest level's; open " +
      "the zip Save wrote, or its NML not named -level-<n>.",
  });
});
