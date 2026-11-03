import assert from "node:assert/strict";
import { test } from "node:test";
import { ChunkAtlas, chunkAtlasLayout } from "../src/gpu/chunk-atlas.js";

const TWO_SLOTS = {
  chunks_ZYX: [2, 2, 2],
  dtype: "|u1",
  byteLimit: 16,
  maxLayers: 8,
  maxTextureSize: 16,
  onEvict: () => {},
};

test("eight voxels lay out 4 by 2", () => {
  assert.deepEqual(chunkAtlasLayout(8, 16), { width: 4, height: 2 });
});

test("a full atlas reuses the least recently used slot outside the frame, and admits nothing when all are in it", () => {
  const atlas = new ChunkAtlas(TWO_SLOTS);
  assert.equal(atlas.capacity, 2);
  const first = atlas.admit("0/0/0", new Uint8Array(8).fill(1));
  const second = atlas.admit("0/0/1", new Uint8Array(8).fill(2));
  atlas.beginFrame(["0/0/1"]);
  const third = atlas.admit("0/1/0", new Uint8Array(8).fill(3));
  assert.equal(third.slot, first.slot);
  assert.equal(atlas.get("0/0/0"), null);
  assert.equal(atlas.get("0/0/1"), second);
  assert.equal(atlas.get("0/1/0"), third);
  atlas.beginFrame(["0/0/1", "0/1/0"]);
  assert.equal(atlas.admit("stale", new Uint8Array(8).fill(4)), null);
  atlas.dispose();
});

test("a frame of three chunks in a two-slot atlas keeps the first two, round after round", () => {
  const atlas = new ChunkAtlas(TWO_SLOTS);
  for (let round = 0; round < 3; round++) {
    atlas.beginFrame(["0/0/0", "0/0/1", "0/1/0"]);
    assert.notEqual(atlas.admit("0/0/0", new Uint8Array(8).fill(1)), null);
    assert.notEqual(atlas.admit("0/0/1", new Uint8Array(8).fill(2)), null);
    assert.equal(atlas.admit("0/1/0", new Uint8Array(8).fill(3)), null);
    assert.deepEqual([...atlas.entries.keys()].sort(), ["0/0/0", "0/0/1"]);
  }
  atlas.dispose();
});

test("a chunk that fills its slot exactly is kept as the same array in every atlas", () => {
  const chunk_DHW = new Uint8Array(8).fill(7);
  const xy = new ChunkAtlas(TWO_SLOTS);
  const zy = new ChunkAtlas(TWO_SLOTS);
  assert.equal(xy.admit("0/0/0", chunk_DHW).data, chunk_DHW);
  assert.equal(zy.admit("0/0/0", chunk_DHW).data, chunk_DHW);
  xy.dispose();
  zy.dispose();
});

test("a smaller atlas retains the visible entry of the one it replaces, in its first slot", () => {
  const previous = new ChunkAtlas(TWO_SLOTS);
  previous.admit("0/0/0", new Uint8Array(8).fill(1));
  const visible = previous.admit("0/0/1", new Uint8Array(8).fill(2));
  previous.admit("0/0/0", new Uint8Array(8).fill(1));
  previous.beginFrame(["0/0/1"]);
  const atlas = new ChunkAtlas({ ...TWO_SLOTS, byteLimit: 8 });
  atlas.retain(null, previous);
  assert.deepEqual([...atlas.entries.keys()], ["0/0/1"]);
  assert.equal(atlas.slots[0], visible);
  previous.dispose();
  atlas.dispose();
});

test("a refresh uploads again the run it changed, and adds it to what a renderer still uploading has left", () => {
  const atlas = new ChunkAtlas(TWO_SLOTS);
  const entry = atlas.admit("0/0/0", new Uint8Array(8));
  const done = {};
  const uploading = {};
  const stale = {};
  entry.uploaded.set(done, { generation: entry.generation, complete: true });
  entry.uploaded.set(uploading, { generation: entry.generation, complete: false, texels: 6, end: 8 });
  entry.uploaded.set(stale, { generation: entry.generation - 1, complete: true });
  const data_DHW = new Uint8Array(8).fill(5);
  atlas.refresh(entry, data_DHW, 2, 4);
  assert.equal(entry.data, data_DHW);
  const generation = entry.generation;
  assert.deepEqual(atlas.uploadProgress(done, entry), { generation, complete: false, texels: 2, end: 4 });
  assert.deepEqual(atlas.uploadProgress(uploading, entry), { generation, complete: false, texels: 2, end: 8 });
  assert.deepEqual(atlas.uploadProgress(stale, entry), { generation, complete: false, texels: 0, end: 8 });
  atlas.dispose();
});

test("a fill takes no slot but an entry: a chunk after it takes slot 0, and a third evicts the fill for slot 1", () => {
  const evicted = [];
  const atlas = new ChunkAtlas({ ...TWO_SLOTS, onEvict: (entry) => evicted.push(entry.key) });
  const renderer = {};
  const fill = atlas.admit("0/0/0", null);
  assert.equal(fill.slot, null);
  assert.equal(atlas.needsUpload(renderer, fill), false);
  assert.deepEqual(atlas.uploadPart(renderer, fill, 8), { uploadedBytes: 0, complete: true });
  const chunk = atlas.admit("0/0/1", new Uint8Array(8).fill(1));
  assert.deepEqual([chunk.slot, atlas.slots[1]], [0, null]);
  const third = atlas.admit("0/1/0", new Uint8Array(8).fill(2));
  assert.deepEqual([evicted, third.slot], [["0/0/0"], 1]);
  atlas.dispose();
});

test("a fill written is a new entry in a slot, holding the voxels to upload", () => {
  const evicted = [];
  const atlas = new ChunkAtlas({ ...TWO_SLOTS, onEvict: (entry) => evicted.push(entry) });
  const fill = atlas.admit("0/0/0", null);
  const data_DHW = new Uint8Array(8).fill(3);
  const written = atlas.refresh(fill, data_DHW, 0, 8);
  assert.deepEqual([written.slot, written.data, atlas.get("0/0/0"), evicted], [0, data_DHW, written, [fill]]);
  assert.equal(atlas.needsUpload({}, written), true);
  atlas.dispose();
});

test("a larger atlas retains a fill without a slot and the chunk beside it in its first slot", () => {
  const previous = new ChunkAtlas(TWO_SLOTS);
  const fill = previous.admit("0/0/0", null);
  const chunk = previous.admit("0/0/1", new Uint8Array(8).fill(1));
  const atlas = new ChunkAtlas({ ...TWO_SLOTS, byteLimit: 32 });
  atlas.retain(null, previous);
  assert.deepEqual([atlas.get("0/0/0"), fill.slot, atlas.slots[0], atlas.slots[1]], [fill, null, chunk, null]);
  previous.dispose();
  atlas.dispose();
});
