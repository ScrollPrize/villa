import assert from "node:assert/strict";
import { test } from "node:test";
import { displayWindow } from "../src/volume/display-window.js";

test("the values 0 to 199 give a window from 0.995 to 198.005, and a sample of only NaN throws", () => {
  const window = displayWindow(Float32Array.from({ length: 200 }, (_, index) => index));
  assert.equal(window.kind, "linear_window");
  assert.ok(Math.abs(window.minimum - 0.995) < 1e-9);
  assert.ok(Math.abs(window.maximum - 198.005) < 1e-9);
  assert.throws(() => displayWindow([Number.NaN]), /has no finite values/);
});
