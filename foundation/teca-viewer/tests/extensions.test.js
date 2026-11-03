import assert from "node:assert/strict";
import { test } from "node:test";
import { drawnStyle, handleOf, layerReaders, taken } from "../src/ui/extensions.js";
import { MadeVolume } from "../src/volume/made-volume.js";

test("a seam takes the options ui/extensions.js declares for it where it declares them, and refuses another key", () => {
  const band = { diameter: 0, width: [4, 32], voxels: 8 };
  assert.equal(taken("addMarks", band, []), band);
  const marks = { points_zyx_N3: [], colors_N3: [], edges_E2: [], neighbours_N2: [], selected: -1, depth: 2 };
  const needs = ["points_zyx_N3", "colors_N3", "edges_E2", "selected", "depth"];
  assert.equal(taken("addMarks", marks, needs, "set"), marks);
  assert.equal(taken("focus", { surface: false }, []).surface, false);
  assert.throws(() => taken("addMarks", { diameter: 0, halo: 2 }, []), {
    message: "addMarks takes diameter, width, voxels, not halo.",
  });
  assert.throws(() => taken("addMarks", { neighbours_N2: [] }, []), {
    message: "addMarks takes diameter, width, voxels, not neighbours_N2.",
  });
  assert.throws(() => taken("addLayer", { color_RGB: null, opacity: 1, blend: "add" }, ["color_RGB", "opacity"]), {
    message: "addLayer takes color_RGB, opacity, not blend.",
  });
  assert.throws(() => taken("addLayer", { color_RGB: null, opacity: 1, surface: null }, ["color_RGB", "opacity"]), {
    message: "addLayer takes color_RGB, opacity, not surface.",
  });
  assert.throws(() => taken("addMarks", { ...marks, diameter: 8 }, needs, "set"), {
    message: "addMarks's set() takes points_zyx_N3, colors_N3, edges_E2, selected, depth, neighbours_N2, not diameter.",
  });
});

test("addLayer draws a made volume on the slices alone, and refuses it as the Surface's reader", () => {
  const level = { shape_ZYX: [2, 2, 2], chunks_ZYX: [2, 2, 2], downsample_ZYX: [1, 1, 1], offset_ZYX: [0, 0, 0] };
  const made = new MadeVolume("band", [level], async () => null);
  assert.deepEqual(layerReaders({ slices: made, surface: null }), { slices: made, surface: null });
  assert.throws(() => layerReaders({ slices: made, surface: made }), {
    message: "Made volume made:band/ is drawn on the slices alone: hand addLayer surface: null.",
  });
  assert.throws(() => layerReaders({ slices: made, surface: null, levels: [] }), {
    message: "addLayer takes slices, surface, not levels.",
  });
});

test("a seam's handle has remove() and the methods ui/extensions.js declares for it, and no other", () => {
  const methods = { set() {}, show() {}, remove() {} };
  assert.equal(handleOf("addMarks", methods), methods);
  assert.throws(() => handleOf("addMarks", { ...methods, hide() {} }), {
    message: "A handle of addMarks has remove(), set(), show(), not hide().",
  });
});

test("a colour is drawn from the low to the high end of its range, and labels whatever their value, with no range", () => {
  const cut = { color_RGB: [0, 1, 0], opacity: 1, range: [100, 255] };
  assert.equal(drawnStyle("addLayer", cut), cut);
  assert.equal(drawnStyle("addSurfaceImage", { ...cut, range: [7, 7] }, "setStyle").range[0], 7);
  const labels = { color_RGB: null, opacity: 0.5 };
  assert.equal(drawnStyle("addLayer", labels), labels);
  assert.throws(() => drawnStyle("addLayer", { ...labels, range: [100, 255] }), {
    message: "Labels take no range, being drawn whatever their value: not [100,255].",
  });
  for (const range of [undefined, [255, 100], [0, 256], [-1, 0], [99.5, 255], [100], "100-255"])
    assert.throws(() => drawnStyle("addLayer", { ...cut, range }, "setStyle"), {
      message: `A range is [low, high] display bytes, 0 ≤ low ≤ high ≤ 255, not ${JSON.stringify(range)}.`,
    });
  assert.throws(() => drawnStyle("addSurfaceImage", { ...cut, cut: [0, 99] }), {
    message: "addSurfaceImage takes color_RGB, opacity, range, not cut.",
  });
});
