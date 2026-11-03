import assert from "node:assert/strict";
import { test } from "node:test";
import { MadeVolume } from "../src/volume/made-volume.js";

const LEVELS = [
  { shape_ZYX: [4, 4, 4], chunks_ZYX: [2, 2, 2], downsample_ZYX: [1, 1, 1], offset_ZYX: [0, 0, 0] },
  { shape_ZYX: [2, 2, 2], chunks_ZYX: [2, 2, 2], downsample_ZYX: [2, 2, 2], offset_ZYX: [0.5, 0.5, 0.5] },
];

test("a chunk is made each time it is read, from its level, place and signal; one made null reads as zeros, a fill", async () => {
  const asked = [];
  const signal = new AbortController().signal;
  const volume = new MadeVolume("test", LEVELS, async (level, chunk_ZYX, given) => {
    asked.push([level, chunk_ZYX, given === signal]);
    return chunk_ZYX[0] === 0 ? Uint8Array.of(1, 2, 3, 4, 5, 6, 7, 8) : null;
  });
  const made = await volume.getChunk(0, 1, 1, 0, volume.scales[0], signal);
  assert.deepEqual([[...made], volume.isFill(made)], [[1, 2, 3, 4, 5, 6, 7, 8], false]);
  await volume.getChunk(0, 1, 1, 0, volume.scales[0], signal);
  const fill = await volume.getChunk(1, 0, 0, 0, volume.scales[0], signal);
  assert.deepEqual([[...fill], volume.isFill(fill)], [[0, 0, 0, 0, 0, 0, 0, 0], true]);
  await volume.getChunk(0, 0, 0, 0, volume.scales[1], signal);
  assert.deepEqual(asked, [
    [0, [0, 1, 1], true],
    [0, [0, 1, 1], true],
    [0, [1, 0, 0], true],
    [1, [0, 0, 0], true],
  ]);
});

test("a chunk made of other than a whole chunk's bytes is refused, and one whose read is aborted is not made", async () => {
  let made = 0;
  const volume = new MadeVolume("test", LEVELS, async () => {
    made += 1;
    return new Uint8Array(4);
  });
  await assert.rejects(volume.getChunk(1, 1, 1, 0, volume.scales[0], new AbortController().signal), {
    message: "Chunk 1,1,1 of level 0 of made:test/ was made of 4 values, not the 8 bytes of a whole chunk.",
  });
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(volume.getChunk(0, 0, 0, 0, volume.scales[0], controller.signal), { name: "AbortError" });
  assert.equal(made, 1);
});
