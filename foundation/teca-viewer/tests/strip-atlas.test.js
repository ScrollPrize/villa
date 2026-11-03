import assert from "node:assert/strict";
import { test } from "node:test";
import { surfaceStripBounds } from "../src/surface/strip-atlas.js";

test("a 2 by 256 strip has 32 blocks of 8 columns, the first skipping its one invalid vertex", () => {
  const data_HWRGBA = new Float32Array(2 * 256 * 4);
  for (let y = 0; y < 2; y++) for (let x = 0; x < 256; x++) data_HWRGBA.set([127.25, 10 + y, x, 1], (y * 256 + x) * 4);
  data_HWRGBA[3] = 0;
  const bounds_B6 = surfaceStripBounds(data_HWRGBA, 256, 2);
  assert.equal(bounds_B6.length, 32 * 6);
  assert.deepEqual([...bounds_B6.slice(0, 6)], [127.25, 10, 0, 127.25, 11, 8]);
});
