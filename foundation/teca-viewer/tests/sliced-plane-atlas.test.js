import assert from "node:assert/strict";
import { test } from "node:test";
import { atlasAdmission, atlasCapacity, settledLevelStack } from "../src/gpu/sliced-plane-atlas.js";

const BYTES = 288 * 1024 * 1024;
const LIMITS = { maxLayers: 2048, maxTextureSize: 16384 };
const U8_64 = { chunks_ZYX: [64, 64, 64], numericType: { dtype: "|u1" } };
const F32_512 = { chunks_ZYX: [512, 512, 512], numericType: { dtype: "<f4" } };

test("64³ u1 slots of 262144 bytes fit 1146 beside twelve 8192-texel page tables in 288 MiB, 286 in a quarter, and at most maxLayers", () => {
  assert.deepEqual(atlasCapacity(U8_64, 1, LIMITS, BYTES), { capacity: 1146, slotBytes: 262144 });
  assert.deepEqual(atlasCapacity(U8_64, 4, LIMITS, BYTES), { capacity: 286, slotBytes: 262144 });
  assert.deepEqual(atlasCapacity(U8_64, 1, { ...LIMITS, maxLayers: 1000 }, BYTES), {
    capacity: 1000,
    slotBytes: 262144,
  });
});

test("a 512³ float32 chunk does not fit a share, so a pair of 64³ u1 and 512³ float32 levels is not admitted", () => {
  assert.throws(() => atlasCapacity(F32_512, 1, LIMITS, BYTES), /cannot hold one chunk/);
  assert.deepEqual(atlasAdmission([U8_64, F32_512], LIMITS), {
    supported: false,
    reason: "CT pane budget cannot hold one chunk and its page tables.",
  });
  assert.deepEqual(atlasAdmission([U8_64, { ...U8_64 }], LIMITS), { supported: true, reason: null });
});

test("a plane picking a coarser level for its pixels starts its stack there, and one past the stack keeps its last", () => {
  const scales = Array.from({ length: 6 }, (_, index) => ({ ...U8_64, path: String(index) }));
  const own = { xy: 1, zy: 3, xz: 5 };
  const { planePlans, levelSelection } = settledLevelStack({
    scales,
    scaleIndex: 1,
    coarsestScaleIndex: 5,
    planes: ["xy", "zy", "xz"],
    planeScaleIndex: (plane) => own[plane],
    addressesFor: () => [{ chunk_ZYX: [0, 0, 0] }],
    limits: LIMITS,
  });
  assert.deepEqual(
    planePlans.map(({ plane, levels }) => [plane, levels.map((level) => level.scaleIndex)]),
    [
      ["xy", [1, 2, 3, 4]],
      ["zy", [3, 4]],
      ["xz", [4]],
    ],
  );
  assert.deepEqual(levelSelection, { requestedScaleIndex: 1, scaleIndex: 1 });
});
