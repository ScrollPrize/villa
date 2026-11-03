import assert from "node:assert/strict";
import { test } from "node:test";
import { BrickPool } from "../src/gpu/brick-pool.js";
import { chunkAtlasAllocation } from "../src/gpu/chunk-atlas.js";

const SHAPE = { chunks_ZYX: [2, 2, 2], dtype: "|u1", maxLayers: 8, maxTextureSize: 16 };
const { slotsPerLayer, slotBytes } = chunkAtlasAllocation({ ...SHAPE, byteLimit: 2 ** 20 });
const LAYER = { ...SHAPE, byteLimit: slotsPerLayer * slotBytes };
const RENDERER = {};

const chunk = (label) => new Uint8Array(8).fill(label);

function leases() {
  const pool = new BrickPool(RENDERER);
  const namespace = pool.namespace({}, 0);
  const [first, second] = [pool.acquire(namespace, LAYER, () => {}), pool.acquire(namespace, LAYER, () => {})];
  assert.equal(first.group, second.group, "the two leases share no atlas");
  return [first, second, Array.from({ length: first.capacity + 1 }, (_, index) => `0/0/0/${index}`)];
}

test("a write reaches a chunk the atlas keeps for no lease, one a lease evicted or one a lease that left held", () => {
  const [lease, other, keys] = leases();
  const entries = keys.map((key, index) => {
    lease.beginFrame([key]);
    return lease.admit(key, chunk(index + 1));
  });
  assert.equal(lease.held(keys[0]), null);
  entries[0].uploaded.set(RENDERER, { generation: entries[0].generation, complete: true });
  lease.refresh(keys[0], chunk(7), 0, 8);
  lease.beginFrame([keys[0]]);
  assert.equal(lease.get(keys[0]), entries[0]);
  assert.deepEqual([...entries[0].data], [...chunk(7)]);
  assert.equal(lease.needsUpload(RENDERER, entries[0]), true);

  lease.beginFrame([keys.at(-1)]);
  lease.dispose();
  assert.equal(other.held(keys.at(-1)), null);
  other.refresh(keys.at(-1), chunk(9), 0, 8);
  assert.equal(other.get(keys.at(-1)), entries.at(-1));
  assert.deepEqual([...entries.at(-1).data], [...chunk(9)]);
  other.dispose();
});

test("a fill a writer writes becomes the lease's chunk in a slot, not the other lease's", () => {
  const [lease, other, keys] = leases();
  lease.beginFrame([keys[0]]);
  const fill = lease.admit(keys[0], null);
  lease.refresh(keys[0], chunk(4), 0, 8);
  const written = lease.held(keys[0]);
  assert.notEqual(written, fill);
  assert.deepEqual([written.slot === null, [...written.data], other.held(keys[0])], [false, [...chunk(4)], null]);
  lease.dispose();
  other.dispose();
});
