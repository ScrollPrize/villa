import assert from "node:assert/strict";
import { test } from "node:test";
import { sliceNavigationPolicy } from "../src/render/navigation-policy.js";

test("a held slice through a rotated pane is held but does not step along an axis", () => {
  assert.deepEqual(sliceNavigationPolicy({ plane: "xy", rotated: true, heldSliceDirection: 1, dirtyPlanes: null }), {
    held: true,
    axisAligned: false,
  });
});

test("a held key does not hold a plane its step left clean", () => {
  assert.deepEqual(sliceNavigationPolicy({ plane: "xy", rotated: false, heldSliceDirection: 1, dirtyPlanes: ["zy"] }), {
    held: false,
    axisAligned: false,
  });
});
