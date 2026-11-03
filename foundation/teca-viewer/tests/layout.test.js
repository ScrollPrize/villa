import assert from "node:assert/strict";
import { test } from "node:test";
import { defaultSurfaceLayout, dockedViewport, validateSurfaceLayout } from "../src/surface/layout.js";

test("a layout with an unknown mode, key or pane, a repeated pane or an out-of-range split or weight is invalid", () => {
  const layout = defaultSurfaceLayout();
  validateSurfaceLayout(layout);
  for (const patch of [
    { mode: "free" },
    { planes: [] },
    { planes: ["xy", "xy"] },
    { planes: ["unknown"] },
    { split: NaN },
    { split: 0.9 },
    { weights: [1, 0, 1] },
    { maximized: "zy" },
    { unknown: true },
  ])
    assert.throws(() => validateSurfaceLayout({ ...layout, ...patch }), /Invalid Surface pane layout/);
});

test("a docked plane stays active, and a plane outside the dock or a solo layout falls back to the Surface", () => {
  const docked = { ...defaultSurfaceLayout(), mode: "right", planes: ["xy", "xz"] };
  assert.equal(dockedViewport(docked, "xz"), "xz");
  assert.equal(dockedViewport(docked, "zy"), "surface");
  assert.equal(dockedViewport({ ...docked, mode: "solo" }, "xz"), "surface");
});
