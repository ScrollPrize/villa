import assert from "node:assert/strict";
import { test } from "node:test";
import { planeSpanPixels } from "../src/views/three-pane.js";

// The camera sees the focus from 3.2 units with a 45° vertical field, so the pane's height spans
// 2 × 3.2 × tan 22.5° = 2.650967 units, and a plane's 1.15 units are 1.15 / 2.650967 = 0.433803 of it.
test("a plane spans 0.433803 of the 3D pane's height in device pixels, rounded up, whatever its width", () => {
  // The 3D pane maximised in the checks' 1440 × 1000 window is 922 CSS pixels tall: 922 × 0.433803 = 399.97.
  assert.equal(planeSpanPixels(922, 1), 400);
  assert.equal(planeSpanPixels(922, 2), 800);
  // Below the Surface beside XY it is 273 tall: 118.43.
  assert.equal(planeSpanPixels(273, 1), 119);
  // 273 × 1.5 = 409.5 is a canvas of 409 device pixels, as the pane's framebuffer has it: 177.43.
  assert.equal(planeSpanPixels(273, 1.5), 178);
});
