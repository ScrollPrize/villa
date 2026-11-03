import assert from "node:assert/strict";
import { test } from "node:test";
import { planeReads } from "../src/gpu/plane-reads.js";

const FINE = { path: "0" };
const COARSE = { path: "1" };

const atlas = {
  inWorkingSet: () => true,
  get: (key) => (key === `0\0${[0, 0, 9].join("/")}` ? {} : null),
};

function plan(plane, x, shown = true) {
  return {
    plane,
    shown,
    scaleIndex: 0,
    levels: [
      {
        scaleIndex: 0,
        scale: FINE,
        scaleKey: "fine",
        atlas,
        fetchAddresses: [
          { chunk_ZYX: [0, 0, x], priority: 1, fetchOrder: 0 },
          { chunk_ZYX: [0, 0, x + 1], priority: 2, fetchOrder: 0 },
          { chunk_ZYX: [0, 0, 9], priority: 0, fetchOrder: 0 },
          { chunk_ZYX: [0, 1, x], priority: 0, fetchOrder: 1 },
        ],
      },
      {
        scaleIndex: 1,
        scale: COARSE,
        scaleKey: "coarse",
        atlas,
        fetchAddresses: [{ chunk_ZYX: [0, 0, x], priority: 3, fetchOrder: 0 }],
      },
    ],
  };
}

test("the coarse fallback reads first, then what the panes show taking turns, then the prefetch", () => {
  const reads = planeReads([plan("xy", 0), plan("zy", 1)]);
  assert.deepEqual(
    reads.map(({ scaleKey, chunk_ZYX }) => `${scaleKey} ${chunk_ZYX.join("/")}`),
    ["coarse 0/0/0", "coarse 0/0/1", "fine 0/0/0", "fine 0/0/2", "fine 0/0/1", "fine 0/1/1", "fine 0/1/0"],
  );
  assert.equal(
    reads.find(({ scaleKey, chunk_ZYX }) => `${scaleKey} ${chunk_ZYX.join("/")}` === "fine 0/0/1").priority,
    10,
  );
});

test("a plane the layout hides reads after the shown pane's bricks, at their last priority, and a brick both show in the pane's turn", () => {
  const reads = planeReads([plan("zy", 1, false), plan("xy", 0)]);
  assert.deepEqual(
    reads.map(({ scaleKey, chunk_ZYX, priority }) => `${scaleKey} ${chunk_ZYX.join("/")} ${priority}`),
    [
      "coarse 0/0/0 0",
      "fine 0/0/0 10",
      "fine 0/0/1 20",
      "coarse 0/0/1 399",
      "fine 0/0/2 399",
      "fine 0/1/0 8000",
      "fine 0/1/1 8000",
    ],
  );
});

test("a shown pane's prefetch brick that a hidden plane shows reads with the hidden plane's", () => {
  const hidden = {
    plane: "xz",
    shown: false,
    scaleIndex: 0,
    levels: [
      {
        scaleIndex: 0,
        scale: FINE,
        scaleKey: "fine",
        atlas,
        fetchAddresses: [{ chunk_ZYX: [0, 1, 0], priority: 0, fetchOrder: 0 }],
      },
    ],
  };
  const reads = planeReads([plan("xy", 0), hidden]);
  assert.deepEqual(
    reads.map(({ scaleKey, chunk_ZYX, priority }) => `${scaleKey} ${chunk_ZYX.join("/")} ${priority}`),
    ["coarse 0/0/0 0", "fine 0/0/0 10", "fine 0/0/1 20", "fine 0/1/0 399"],
  );
  assert.equal(reads.at(-1).shown, false);
});
