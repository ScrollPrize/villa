import assert from "node:assert/strict";
import { test } from "node:test";
import { crc32, inflateRawSync } from "node:zlib";
import { unzipped, zipped } from "../src/zip.js";

const pause = async () => {};

test("a zip stores 123456789 with CRC-32 CBF43926 and the 43 bytes of the quick brown fox with 414FA339", async () => {
  const encoder = new TextEncoder();
  for (const [text, crc] of [
    ["123456789", 0xcbf43926],
    ["The quick brown fox jumps over the lazy dog", 0x414fa339],
  ]) {
    const zip = new Uint8Array(
      await (await zipped([{ name: "a", bytes: encoder.encode(text) }], pause, false)).arrayBuffer(),
    );
    assert.equal(new DataView(zip.buffer).getUint16(8, true), 0);
    assert.equal(new DataView(zip.buffer).getUint32(14, true), crc);
    assert.deepEqual(
      (await unzipped(zip)).map(({ name, bytes }) => [name, new TextDecoder().decode(bytes)]),
      [["a", text]],
    );
  }
});

test("a deflated entry of 1000 times 123456789 is method 8, inflates to it in zlib and back, and a wrong CRC is refused", async () => {
  const text = "123456789".repeat(1000);
  const zip = new Uint8Array(
    await (await zipped([{ name: "a", bytes: new TextEncoder().encode(text) }], pause, true)).arrayBuffer(),
  );
  const view = new DataView(zip.buffer);
  const size = view.getUint32(18, true);
  assert.deepEqual(
    [view.getUint16(8, true), view.getUint32(14, true), view.getUint32(22, true)],
    [8, crc32(text), 9000],
  );
  assert.ok(size < 100, `${size} bytes deflated`);
  assert.equal(inflateRawSync(zip.subarray(31, 31 + size)).toString(), text);
  assert.deepEqual(
    (await unzipped(zip)).map(({ name, bytes }) => [name, new TextDecoder().decode(bytes)]),
    [["a", text]],
  );
  view.setUint32(view.getUint32(zip.length - 6, true) + 16, crc32(text) ^ 1, true);
  await assert.rejects(unzipped(zip), { message: "The zip's a has the wrong size or CRC." });
});

test("an empty zip and empty stored or deflated entries remain empty", async () => {
  for (const deflate of [false, true]) {
    for (const entries of [[], [{ name: "empty", bytes: new Uint8Array() }]]) {
      const zip = new Uint8Array(await (await zipped(entries, pause, deflate)).arrayBuffer());
      assert.deepEqual(await unzipped(zip), entries);
    }
  }
});

test("a deflated MiB claiming 0, 8 or 65536 bytes stops at its declared size, even with a correct CRC", async () => {
  const bytes = new Uint8Array(1024 * 1024).fill(7);
  const zip = new Uint8Array(await (await zipped([{ name: "large", bytes }], pause, true)).arrayBuffer());
  const view = new DataView(zip.buffer);
  const directory = view.getUint32(zip.length - 6, true);
  assert.equal(view.getUint32(directory + 16, true), crc32(bytes));
  for (const count of [0, 8, 65536]) {
    view.setUint32(22, count, true);
    view.setUint32(directory + 24, count, true);
    await assert.rejects(unzipped(zip), {
      message: `The zip's large expands past its declared ${count} bytes.`,
    });
  }
  view.setUint32(22, bytes.length, true);
  view.setUint32(directory + 24, bytes.length, true);
  assert.deepEqual(await unzipped(zip), [{ name: "large", bytes }]);
  view.setUint32(directory + 20, view.getUint32(directory + 20, true) - 1, true);
  view.setUint32(directory + 24, 8, true);
  await assert.rejects(unzipped(zip), { message: "The zip's large expands past its declared 8 bytes." });
});

test("a deflated entry claiming more bytes than it contains is refused", async () => {
  const zip = new Uint8Array(
    await (await zipped([{ name: "short", bytes: Uint8Array.of(3) }], pause, true)).arrayBuffer(),
  );
  const view = new DataView(zip.buffer);
  view.setUint32(22, 0xffffffff, true);
  view.setUint32(view.getUint32(zip.length - 6, true) + 24, 0xffffffff, true);
  await assert.rejects(unzipped(zip), { message: "The zip's short has the wrong size or CRC." });
});

test("a broken deflate block reports its decompression error when the reader is cancelled", async () => {
  const zip = new Uint8Array(
    await (await zipped([{ name: "broken", bytes: Uint8Array.of(3) }], pause, true)).arrayBuffer(),
  );
  // The low three bits declare a final block of the reserved deflate type 3.
  zip[30 + "broken".length] = 0xff;
  await assert.rejects(unzipped(zip), { name: "TypeError", code: "Z_DATA_ERROR" });
});
