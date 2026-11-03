import assert from "node:assert/strict";
import { test } from "node:test";
import { FILL_PAGE, PageTable, pageTableLayout } from "../src/gpu/page-table.js";

function contents(table) {
  const slots = new Map();
  for (let offset = 0; offset < table.data.length; offset += 4) {
    const [z, y, x, slot] = table.data.subarray(offset, offset + 4);
    if (slot !== 0) slots.set(`${z - 1}/${y - 1}/${x - 1}`, slot - 1);
  }
  return slots;
}
const lookup = (table, chunk_ZYX) => contents(table).get(chunk_ZYX.join("/")) ?? null;

test("a table for 512 entries on a 64-texel texture has 2048 texels, 64 by 32, of 16 bytes each", () => {
  assert.deepEqual(pageTableLayout(512, 64), { size: 2048, width: 64, height: 32 });
  const table = new PageTable(512, 64);
  assert.equal(table.allocatedBytes(), 2048 * 16);
  table.dispose();
});

test("a write replaces every entry of the one before it", () => {
  const table = new PageTable(4, 64);
  table.write([
    { chunk_ZYX: [1, 2, 3], slot: 7 },
    { chunk_ZYX: [4, 5, 6], slot: 2 },
  ]);
  assert.equal(lookup(table, [1, 2, 3]), 7);
  assert.equal(lookup(table, [4, 5, 6]), 2);
  assert.equal(lookup(table, [0, 0, 0]), null);
  table.write([{ chunk_ZYX: [4, 5, 6], slot: 1 }]);
  assert.equal(lookup(table, [1, 2, 3]), null);
  assert.equal(lookup(table, [4, 5, 6]), 1);
  table.dispose();
});

test("32 chunks that hash to texel 0 of 128 fill texels 0 to 31 in write order", () => {
  const table = new PageTable(32, 64);
  assert.equal(table.size, 128);
  const entries = Array.from({ length: 32 }, (_, slot) => ({ chunk_ZYX: [0, 0, slot * 128], slot }));
  table.write(entries);
  for (const { chunk_ZYX, slot } of entries) {
    assert.deepEqual([...table.data.subarray(slot * 4, slot * 4 + 4)], [1, 1, chunk_ZYX[2] + 1, slot + 1]);
  }
  table.write([...entries].reverse());
  assert.deepEqual([...table.data.subarray(0, 4)], [1, 1, 31 * 128 + 1, 32]);
  table.dispose();
});

test("a 33rd entry in a table for 32 throws and keeps the 32, and a shorter write drops the rest", () => {
  const table = new PageTable(32, 64);
  const entries = Array.from({ length: 32 }, (_, slot) => ({ chunk_ZYX: [0, 0, slot * 128], slot }));
  table.write(entries);
  assert.throws(() => table.write([...entries, { chunk_ZYX: [0, 0, 1], slot: 32 }]), /capacity/);
  assert.equal(contents(table).size, 32);
  table.write(entries.slice(16));
  assert.equal(lookup(table, entries[0].chunk_ZYX), null);
  assert.equal(lookup(table, entries[31].chunk_ZYX), 31);
  table.dispose();
});

test("all 32768 chunks of a 128 by 16 by 16 region are written", () => {
  const entries = [];
  for (let z = 1000; z < 1128; z++)
    for (let y = 400; y < 416; y++)
      for (let x = 1700; x < 1716; x++) entries.push({ chunk_ZYX: [z, y, x], slot: entries.length });
  const table = new PageTable(entries.length, 16384);
  table.write(entries);
  const slots = contents(table);
  for (const { chunk_ZYX, slot } of entries) assert.equal(slots.get(chunk_ZYX.join("/")), slot);
  assert.equal(slots.has("1128/400/1700"), false);
  table.dispose();
});

test("a fill entry's texel holds its chunk + 1 and FILL_PAGE, 2³² - 1, where a slot's holds the slot + 1", () => {
  const table = new PageTable(4, 64);
  table.write([
    { chunk_ZYX: [1, 2, 3], slot: null },
    { chunk_ZYX: [4, 5, 6], slot: 0 },
  ]);
  const texels = [];
  for (let offset = 0; offset < table.data.length; offset += 4)
    if (table.data[offset + 3] !== 0) texels.push([...table.data.subarray(offset, offset + 4)]);
  assert.deepEqual(
    texels.sort((a, b) => a[0] - b[0]),
    [
      [2, 3, 4, FILL_PAGE],
      [5, 6, 7, 1],
    ],
  );
  assert.equal(FILL_PAGE, 4_294_967_295);
  table.dispose();
});
