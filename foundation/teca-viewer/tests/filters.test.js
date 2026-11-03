import assert from "node:assert/strict";
import { test } from "node:test";
import { KINDS } from "../src/extensions/filters/passes.js";
import { savedStages } from "../src/extensions/filters.js";

test("a saved stack is taken whole, each stage's value in its kind's range or null for a kind without one", () => {
  const stack = [
    { kind: "gaussian", enabled: true, value: 2 },
    { kind: "median", enabled: false, value: null },
    { kind: "threshold", enabled: true, value: 0 },
  ];
  assert.equal(savedStages(stack), stack);
  assert.deepEqual(savedStages([]), []);
  for (const stage of [
    { kind: "gaussian", enabled: true, value: 9 },
    { kind: "median", enabled: true, value: 3 },
    { kind: "edges", enabled: "yes", value: null },
    { kind: "bilateral", enabled: true, value: 1 },
    { kind: "sharpen", enabled: true, value: 1, extra: 0 },
    null,
  ])
    assert.throws(() => savedStages([stage]), /Saved filter is malformed/);
  assert.throws(() => savedStages({}), /must be an array/);
});

test("a stage reaches as many taps from a pixel as its passes read, a blur's two passes three σ each", () => {
  assert.equal(KINDS.gaussian.reach(1), 6);
  assert.equal(KINDS.gaussian.reach(0.25), 2);
  assert.equal(KINDS.gaussian.reach(8), 48);
  assert.equal(KINDS.median.reach(null), 1);
  assert.equal(KINDS.edges.reach(null), 1);
  assert.equal(KINDS.threshold.reach(128), 0);
});
