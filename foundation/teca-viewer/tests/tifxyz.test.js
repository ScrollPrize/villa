import assert from "node:assert/strict";
import { test } from "node:test";
import { gridStorage, readTiffTags, tifxyzUrl } from "../src/surface/tifxyz.js";

const BASE = "https://example.test/collection/surface.tifxyz";

test("an index.html fragment or a query reduces to the tifxyz directory URL with no trailing slash", () => {
  assert.equal(tifxyzUrl("https://example.test/index.html#collection/surface.tifxyz/"), BASE);
  assert.equal(tifxyzUrl(`${BASE}/?token=1`), BASE);
  assert.equal(
    tifxyzUrl("https://example.test/index.html#collection/tifxyz_original/"),
    "https://example.test/collection/tifxyz_original",
  );
});

test("an .nml URL is not a tifxyz directory and a relative path is not an HTTP(S) URL", () => {
  assert.throws(() => tifxyzUrl("https://example.test/skeleton.nml"), /not a tifxyz directory URL/);
  assert.throws(() => tifxyzUrl("local/surface.tifxyz"), /not an HTTP\(S\) tifxyz URL/);
});

function tiled(offsetsCount, offsetsAt) {
  const tables = 8 + 2 + 6 * 12 + 4;
  const entries = [
    [256, 3, 1, 80],
    [257, 3, 1, 64],
    [322, 3, 1, 16],
    [323, 3, 1, 16],
    [324, 4, offsetsCount, offsetsAt ?? tables],
    [325, 4, 20, tables + 20 * 4],
  ];
  const bytes = new Uint8Array(tables + 2 * 20 * 4);
  const view = new DataView(bytes.buffer);
  bytes.set([73, 73]);
  view.setUint16(2, 42, true);
  view.setUint32(4, 8, true);
  view.setUint16(8, entries.length, true);
  for (const [index, [tag, type, count, value]] of entries.entries()) {
    const at = 10 + index * 12;
    view.setUint16(at, tag, true);
    view.setUint16(at + 2, type, true);
    view.setUint32(at + 4, count, true);
    if (type === 3) view.setUint16(at + 8, value, true);
    else view.setUint32(at + 8, value, true);
  }
  return bytes;
}

test("a tiled TIFF whose tile table is short or past the file is refused naming its tile table", async () => {
  const layout = async (bytes) => {
    const tiff = await readTiffTags("x.tif", bytes.length, async (offset, length) =>
      bytes.subarray(offset, offset + length),
    );
    return gridStorage("x.tif", tiff, 64, 80, bytes.length);
  };
  await assert.rejects(layout(tiled(19)), { message: "x.tif tile table 324 must contain 20 unsigned integers." });
  await assert.rejects(layout(tiled(20, 10_000_000)), { message: "x.tif tile table falls outside the file." });
});
