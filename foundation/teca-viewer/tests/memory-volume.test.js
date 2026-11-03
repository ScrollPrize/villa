import assert from "node:assert/strict";
import { test } from "node:test";
import { MemoryVolume } from "../src/volume/memory-volume.js";

const LEVELS = [{ shape_ZYX: [3, 4, 4], chunks_ZYX: [2, 2, 2], downsample_ZYX: [1, 1, 1], offset_ZYX: [0, 0, 0] }];

test("a 2³ write at 1,1,1 lands in the eight chunks around it, each changed from the first z plane it reached", () => {
  const volume = new MemoryVolume("test", LEVELS, (length) => new Uint8Array(length));
  volume.write(0, [1, 1, 1], [2, 2, 2], Uint8Array.of(1, 2, 3, 4, 5, 6, 7, 8));
  assert.deepEqual([...volume.chunk(0, [0, 0, 0])], [0, 0, 0, 0, 0, 0, 0, 1]);
  assert.deepEqual([...volume.chunk(0, [1, 1, 1])], [8, 0, 0, 0, 0, 0, 0, 0]);
  const changes = volume.take();
  assert.equal(changes.length, 8);
  const first = changes.find((change) => change.chunk_ZYX.join() === "0,0,0");
  assert.deepEqual([first.start, first.end], [4, 8]);
  const last = changes.find((change) => change.chunk_ZYX.join() === "1,1,1");
  assert.deepEqual([last.start, last.end], [0, 4]);
  assert.equal(last.data_DHW, volume.chunk(0, [1, 1, 1]));
  assert.deepEqual(volume.take(), []);
});

test("a chunk never written reads as zeros, a fill, and a write past the level's edge is refused", async () => {
  const volume = new MemoryVolume("test", LEVELS, (length) => new Uint8Array(length));
  const read = await volume.getChunk(1, 1, 1, 0, volume.scales[0], new AbortController().signal);
  assert.deepEqual([[...read], volume.isFill(read)], [[0, 0, 0, 0, 0, 0, 0, 0], true]);
  assert.throws(() => volume.write(0, [2, 0, 0], [2, 1, 1], new Uint8Array(2)), {
    message: "A write of 2×1×1 at 2,0,0 is not inside level 0 (3×4×4).",
  });
});

test("a whole chunk placed is held as the array itself and handed on whole; one past the level's edge is refused", () => {
  const volume = new MemoryVolume("test", LEVELS, (length) => new Uint8Array(length));
  const data_DHW = Uint8Array.of(1, 2, 3, 4, 5, 6, 7, 8);
  volume.place(0, [0, 1, 1], data_DHW);
  assert.equal(volume.chunk(0, [0, 1, 1]), data_DHW);
  assert.deepEqual(
    volume.take().map(({ chunk_ZYX, data_DHW, start, end }) => [chunk_ZYX, data_DHW, start, end]),
    [[[0, 1, 1], data_DHW, 0, 8]],
  );
  assert.throws(() => volume.place(0, [1, 0, 0], new Uint8Array(8)), {
    message: "Chunk 1,0,0 is not a whole chunk inside level 0 (3×4×4).",
  });
});

test("a chunk placed twice before a take is handed on as the second array, whole", () => {
  const volume = new MemoryVolume("test", LEVELS, (length) => new Uint8Array(length));
  const second = new Uint8Array(8).fill(2);
  volume.place(0, [0, 0, 0], new Uint8Array(8).fill(1));
  volume.place(0, [0, 0, 0], second);
  const [change, ...rest] = volume.take();
  assert.equal(change.data_DHW, second);
  assert.deepEqual([change.start, change.end, rest.length], [0, 8, 0]);
});

test("runs set a chunk's voxels in ZYX order and are kept from the first z plane they reach; one past it is refused", () => {
  const volume = new MemoryVolume("test", LEVELS, (length) => new Uint8Array(length));
  volume.writeRuns(0, [0, 0, 1], 2, Uint32Array.of(1, 6), Uint32Array.of(2, 1), Uint8Array.of(4, 7));
  assert.deepEqual([...volume.chunk(0, [0, 0, 1])], [0, 4, 4, 0, 0, 0, 7, 0]);
  assert.deepEqual(
    volume.take().map(({ chunk_ZYX, start, end }) => [chunk_ZYX, start, end]),
    [[[0, 0, 1], 0, 8]],
  );
  assert.throws(() => volume.writeRuns(0, [0, 0, 0], 2, Uint32Array.of(7), Uint32Array.of(2), Uint8Array.of(1)), {
    message: "A run of 2 voxels along axis 2 at 7 is not inside a chunk of 8.",
  });
});

test("runs along y and z set every second and fourth voxel of a 2³ chunk; one along z from voxel 5 is refused", () => {
  const volume = new MemoryVolume("test", LEVELS, (length) => new Uint8Array(length));
  volume.writeRuns(0, [0, 0, 0], 1, Uint32Array.of(1), Uint32Array.of(4), Uint8Array.of(3));
  volume.writeRuns(0, [0, 0, 0], 0, Uint32Array.of(2), Uint32Array.of(2), Uint8Array.of(9));
  assert.deepEqual([...volume.chunk(0, [0, 0, 0])], [0, 3, 9, 3, 0, 3, 9, 3]);
  assert.deepEqual(
    volume.take().map(({ start, end }) => [start, end]),
    [[0, 8]],
  );
  assert.throws(() => volume.writeRuns(0, [0, 0, 0], 0, Uint32Array.of(5), Uint32Array.of(2), Uint8Array.of(1)), {
    message: "A run of 2 voxels along axis 0 at 5 is not inside a chunk of 8.",
  });
});
