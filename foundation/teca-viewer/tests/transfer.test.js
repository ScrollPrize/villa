import assert from "node:assert/strict";
import { test } from "node:test";
import {
  evaluateSurfaceTransfer,
  surfaceTransferColor,
  surfaceTransferPreset,
  validateSurfaceTransfer,
} from "../src/surface/transfer.js";

const CUSTOM = {
  preset: "custom",
  points: [
    { value: 0, color: "#000000", opacity: 0 },
    { value: 0.5, color: "#ff0000", opacity: 0.25 },
    { value: 1, color: "#ffffff", opacity: 1 },
  ],
};

test("a value of 0.25 halfway from black to red at 0.5 is color [0.5, 0, 0] at opacity 0.125", () => {
  assert.deepEqual(evaluateSurfaceTransfer(0.25, CUSTOM), { color: [0.5, 0, 0], opacity: 0.125 });
});

test("the ink preset is five points from clear red to white at opacity 0.35", () => {
  assert.deepEqual(surfaceTransferPreset("ink"), [
    { value: 0, color: "#ff4242", opacity: 0 },
    { value: 0.22, color: "#ff4242", opacity: 0 },
    { value: 0.34, color: "#dbc34d", opacity: 0.12 },
    { value: 0.37, color: "#cbb962", opacity: 0.22 },
    { value: 1, color: "#ffffff", opacity: 0.35 },
  ]);
});

test("points in falling order are malformed, and a named preset must keep its own points", () => {
  assert.throws(
    () => validateSurfaceTransfer({ ...CUSTOM, points: CUSTOM.points.toReversed() }),
    /points are malformed/,
  );
  assert.throws(
    () => validateSurfaceTransfer({ preset: "linear", points: CUSTOM.points }),
    /must retain their canonical points/,
  );
});

test("#ff8000 is color [1, 128/255, 0], and a four-digit color is malformed", () => {
  assert.deepEqual(surfaceTransferColor("#ff8000"), [1, 128 / 255, 0]);
  assert.throws(() => surfaceTransferColor("#ff80"), /color is malformed/);
});
