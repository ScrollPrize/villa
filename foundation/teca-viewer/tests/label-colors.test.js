import assert from "node:assert/strict";
import { test } from "node:test";
import { labelColor } from "../src/gpu/label-colors.js";

test("label 1's hue is the golden ratio's 0.618, a blue at 0.7 of the way from white: 0.3, 0.504, 1", () => {
  assert.deepEqual(
    labelColor(1).map((value) => Math.round(value * 1000) / 1000),
    [0.3, 0.504, 1],
  );
});
