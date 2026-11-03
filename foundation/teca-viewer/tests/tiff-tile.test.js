import assert from "node:assert/strict";
import { test } from "node:test";
import { decodeFloat32Tile } from "../src/surface/tiff-tile.js";

const LZW_32 = { compression: 5, predictor: 3, littleEndian: true, tileWidth: 32, tileHeight: 32 };

test("an uncompressed little-endian 2 by 1 tile of bytes for 1.5 and -2 decodes to [1.5, -2]", async () => {
  const bytes = new Uint8Array(new Float32Array([1.5, -2]).buffer);
  const layout = { compression: 1, predictor: 1, littleEndian: true, tileWidth: 2, tileHeight: 1 };
  assert.deepEqual([...(await decodeFloat32Tile(bytes, layout))], [1.5, -2]);
});

test("an empty LZW stream is truncated and one that ends early has an incorrect decoded size", async () => {
  await assert.rejects(decodeFloat32Tile(new Uint8Array(), LZW_32), /truncated/);
  await assert.rejects(decodeFloat32Tile(new Uint8Array([0x80, 0x40, 0x40]), LZW_32), /incorrect decoded size/);
});

test("an 8192 by 8192 tile exceeds the 16 MiB limit and an aborted signal rejects with AbortError", async () => {
  await assert.rejects(decodeFloat32Tile(new Uint8Array(), { ...LZW_32, tileWidth: 8192, tileHeight: 8192 }), /16 MiB/);
  await assert.rejects(decodeFloat32Tile(new Uint8Array(), LZW_32, AbortSignal.abort()), { name: "AbortError" });
});
