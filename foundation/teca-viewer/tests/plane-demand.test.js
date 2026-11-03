import assert from "node:assert/strict";
import { test } from "node:test";
import { planPlaneDemand } from "../src/gpu/plane-demand.js";
import { IDENTITY_ROTATION, quatFromAxisAngle } from "../src/math.js";
import { rotatedPlaneBasis } from "../src/planes.js";

const keys = (addresses) => addresses.map(({ chunk_ZYX }) => chunk_ZYX.join("/"));

test("a pan 57.6 voxels ahead adds its first 32 chunks outside the 198 shown, to fetch after them", () => {
  const scale = {
    chunks_ZYX: [32, 64, 128],
    downsample_ZYX: [2, 1, 1],
    offset_ZYX: [0, 0, 0],
    shape_ZYX: [2048, 4096, 4096],
  };
  const basis = rotatedPlaneBasis("xy", quatFromAxisAngle([1, 0, 0], 25));
  const prediction = { center_ZYX: [2137.6, 2048, 2048], basis };
  const [[{ addresses, prefetch }]] = planPlaneDemand({
    center_ZYX: [2080, 2048, 2048],
    scales: [scale],
    views: [{ basis, width: 1024, aspect: 1, coverage: 1, predictions: [prediction] }],
  });
  assert.equal(addresses.length, 198);
  assert.equal(prefetch.length, 32);
  assert.deepEqual(keys(prefetch).slice(0, 4), ["33/32/16", "33/32/15", "33/31/16", "33/31/15"]);
  assert.ok(prefetch.every(({ fetchOrder }) => fetchOrder === 1));
  const shown = new Set(keys(addresses));
  assert.ok(keys(prefetch).every((key) => !shown.has(key)));
});

test("a prediction over the address limit prefetches nothing and keeps the shown chunks; a coverage widens them", () => {
  const scale = { chunks_ZYX: [1, 1, 1], downsample_ZYX: [1, 1, 1], offset_ZYX: [0, 0, 0], shape_ZYX: [512, 512, 512] };
  const basis = rotatedPlaneBasis("xy", IDENTITY_ROTATION);
  const plan = (z, coverage = 1) =>
    planPlaneDemand({
      center_ZYX: [100, 256, 256],
      scales: [scale],
      views: [{ basis, width: 200, aspect: 1, coverage, predictions: [{ center_ZYX: [z, 256, 256], basis }] }],
    })[0][0];
  // At a whole Z the plane reads one layer of 201² chunks; between two it reads both, over 65536.
  assert.equal(plan(101).prefetch.length, 32);
  const { addresses, prefetch } = plan(100.5);
  assert.equal(addresses.length, 201 * 201);
  assert.deepEqual(prefetch, []);
  assert.equal(plan(100, 1.1).addresses.length, 221 * 221);
});
