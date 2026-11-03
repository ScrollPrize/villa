import assert from "node:assert/strict";
import { test } from "node:test";
import { savedBookmarks, typedBookmarks } from "../src/extensions/bookmarks.js";

// PHerc1667's CT at 2.399 µm, in finest voxels, and a focus in it.
const SHAPE_ZYX = [37076, 15229, 15229];
const FOCUS_ZYX = [28804, 6121, 6294];
const CT_URL =
  "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr/";

test("a line x, y, z is a bookmark there, ZYX, and ` - ` puts a label after it, which may hold ` - ` itself", () => {
  const typed = (text) => typedBookmarks(text, SHAPE_ZYX, FOCUS_ZYX);
  assert.deepEqual(typed("9116.998, 8047.635, 20525"), [{ point_ZYX: [20525, 8047.635, 9116.998], label: "" }]);
  assert.deepEqual(typed("4540.98, 7815.107, 23104 - fiber"), [
    { point_ZYX: [23104, 7815.107, 4540.98], label: "fiber" },
  ]);
  assert.deepEqual(typed("9181.963, 6078.737, 20247 - void/surface break"), [
    { point_ZYX: [20247, 6078.737, 9181.963], label: "void/surface break" },
  ]);
  assert.deepEqual(typed("  5242.049,6871.464 , 27375 - lol - fold label this  "), [
    { point_ZYX: [27375, 6871.464, 5242.049], label: "lol - fold label this" },
  ]);
  assert.deepEqual(typed("[6293.649, 6120.605, 28804]"), [{ point_ZYX: [28804, 6120.605, 6293.649], label: "" }]);
});

test("one line that is no position labels the focus, an empty field leaves it unlabelled", () => {
  const typed = (text) => typedBookmarks(text, SHAPE_ZYX, FOCUS_ZYX);
  assert.deepEqual(typed("kollesis end"), [{ point_ZYX: FOCUS_ZYX, label: "kollesis end" }]);
  assert.deepEqual(typed(" a - b "), [{ point_ZYX: FOCUS_ZYX, label: "a - b" }]);
  assert.deepEqual(typed("1, 2 - fold"), [{ point_ZYX: FOCUS_ZYX, label: "1, 2 - fold" }]);
  assert.deepEqual(typed(""), [{ point_ZYX: FOCUS_ZYX, label: "" }]);
  assert.deepEqual(typed("\n  \n"), [{ point_ZYX: FOCUS_ZYX, label: "" }]);
});

test("several lines are a bookmark each, in order, blank lines left out, and one that is no position refuses them", () => {
  const typed = (text) => typedBookmarks(text, SHAPE_ZYX, FOCUS_ZYX);
  assert.deepEqual(typed("5385.042, 6711.161, 27774.607\r\n\r\n4625.36, 8480.676, 27651 - label\n"), [
    { point_ZYX: [27774.607, 6711.161, 5385.042], label: "" },
    { point_ZYX: [27651, 8480.676, 4625.36], label: "label" },
  ]);
  for (const [text, line] of [
    ["1, 2, 3\nkollesis", 'Line 2, "kollesis"'],
    ["1, 2, 3\n\n1, 2", 'Line 3, "1, 2"'],
    ["1, 2, 3, 4\n1, 2, 3", 'Line 1, "1, 2, 3, 4"'],
    ["1, 2, 3 -fold\n1, 2, 3", 'Line 1, "1, 2, 3 -fold"'],
    ["1, 2, 3\n1, 2, 3 -", 'Line 2, "1, 2, 3 -"'],
    ["x, y, z\n1, 2, 3", 'Line 1, "x, y, z"'],
  ])
    assert.throws(() => typed(text), { message: `${line}, is not x, y, z.` });
});

test("a position outside the CT is refused, alone or among others", () => {
  const typed = (text) => typedBookmarks(text, SHAPE_ZYX, FOCUS_ZYX);
  assert.throws(() => typed("1, 2, 37076"), {
    message: 'Line 1, "1, 2, 37076", lies outside the CT, whose x, y and z are below 15229, 15229, 37076.',
  });
  assert.throws(() => typed("1, 2, 3\n-1, 2, 3 - before"), {
    message: 'Line 2, "-1, 2, 3 - before", lies outside the CT, whose x, y and z are below 15229, 15229, 37076.',
  });
  assert.deepEqual(typed("15228.999, 0, 37075.5"), [{ point_ZYX: [37075.5, 0, 15228.999], label: "" }]);
});

test("what the address gives back is taken whole, a malformed part named", () => {
  const saved = {
    [CT_URL]: [
      { point_ZYX: [20525, 8047.635, 9116.998], label: "" },
      { point_ZYX: [23104, 7815.107, 4540.98], label: "fiber" },
    ],
    "http://127.0.0.1:8767/small.zarr/": [],
  };
  assert.equal(savedBookmarks(saved), saved);
  assert.deepEqual(savedBookmarks({}), {});
  for (const bookmark of [
    { point_ZYX: [1, 2], label: "" },
    { point_ZYX: [1, 2, Number.NaN], label: "" },
    { point_ZYX: [1, -2, 3], label: "" },
    { point_ZYX: [1, 2, 3], label: null },
    { point_ZYX: [1, 2, 3], label: "a\nb" },
    { point_ZYX: [1, 2, 3] },
    { point_ZYX: [1, 2, 3], label: "", at: 0 },
    [1, 2, 3],
    null,
  ])
    assert.throws(() => savedBookmarks({ [CT_URL]: [bookmark] }), {
      message: `Saved bookmark is malformed: ${JSON.stringify(bookmark)}.`,
    });
  for (const wrong of [[], null, "x", { "PHerc1667.zarr": [] }, { [CT_URL]: {} }])
    assert.throws(() => savedBookmarks(wrong), {
      message: `Saved bookmarks are lists keyed by a CT's URL, not ${JSON.stringify(wrong)}.`,
    });
});
