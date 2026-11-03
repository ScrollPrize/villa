import assert from "node:assert/strict";
import { test } from "node:test";
import { SurfacePyramid } from "../src/surface/pyramid.js";
import { SurfacePyramidCache } from "../src/surface/pyramid-cache.js";

// A 4 by 4 pyramid holds 3·3 + 2·2 points of 16 bytes and one intersection block of 48 bytes.
const BYTES = 256;
const small = () => new SurfacePyramid([4, 4]);

test("a 4 by 4 pyramid counts 256 bytes, and taking one hands over its own arrays once", () => {
  const first = small();
  const second = small();
  const cache = new SurfacePyramidCache(BYTES * 2);
  cache.retain("a", first);
  cache.retain("b", second);
  assert.equal(cache.bytes, BYTES * 2);
  assert.equal(cache.take("a"), first);
  assert.equal(cache.take("a"), null);
  assert.equal(cache.bytes, BYTES);
  first.levels[0].data_HWRGBA[0] = 42;
  assert.equal(second.levels[0].data_HWRGBA[0], 0);
});

test("a cache of 512 bytes drops its oldest pyramid for a third, keeps none over its limit and none under a null key", () => {
  const first = small();
  const cache = new SurfacePyramidCache(BYTES * 2);
  cache.retain("b", small());
  cache.retain("a", first);
  cache.retain("c", small());
  assert.equal(cache.take("b"), null);
  cache.retain("huge", new SurfacePyramid([64, 64]));
  assert.equal(cache.take("huge"), null);
  cache.retain(null, first);
  assert.equal(cache.take(null), null);
  assert.equal(cache.bytes, BYTES * 2);
  cache.retain("a", first);
  assert.equal(cache.bytes, BYTES * 2);
});
