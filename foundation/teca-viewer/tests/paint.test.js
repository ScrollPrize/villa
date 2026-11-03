import assert from "node:assert/strict";
import { mock, test } from "node:test";
import { Brush } from "../src/extensions/paint/brush.js";
import {
  chunkPath,
  largestLabel,
  pathChunk,
  readLabels,
  storeDocuments,
  through,
} from "../src/extensions/paint/file.js";
import { planeFill } from "../src/extensions/paint/fill.js";
import { Gate, rawThreshold } from "../src/extensions/paint/gate.js";
import { History, Pieces, piecesOf } from "../src/extensions/paint/history.js";
import { Labels } from "../src/extensions/paint/labels.js";
import { Pyramid } from "../src/extensions/paint/pyramid.js";
import { Spares } from "../src/extensions/paint/spares.js";
import { cropped, pooled, poolRatios, TILE, voxelTiles } from "../src/extensions/paint/stamp.js";
import { ENTRY_LIMIT_MS, Keeping, NONE } from "../src/extensions/paint/storage.js";
import { MemoryVolume } from "../src/volume/memory-volume.js";
import { unzipped, zipped } from "../src/zip.js";

const LEVEL = { shape_ZYX: [64, 64, 64], chunks_ZYX: [64, 64, 64], offset_ZYX: [0, 0, 0] };
const SMALL = [{ shape_ZYX: [3, 4, 5], chunks_ZYX: [2, 2, 2], scale_ZYX: [1, 1, 1], translation_ZYX: [0, 0, 0] }];
const UNITS = ["micrometer", "micrometer", "micrometer"];
const pause = async () => {};

function recorder() {
  const writes = [];
  return {
    writes,
    journal: (_url, sequence, entry, head, edit) => writes.push({ sequence, entry, head, edit }),
    forget: () => writes.push({ forgot: true }),
    compact: async (_url, chunks, last, forgotten) => void writes.push({ chunks, last, forgotten }),
  };
}

async function kept(chunks) {
  const read = new Map();
  for (const [path, data_DHW] of chunks)
    read.set(path, { data_DHW: data_DHW.slice(), bytes: await through(data_DHW, new CompressionStream("gzip")) });
  return read;
}

function labelsOn(sizes, chunk) {
  const levels = sizes.map((size, level) => ({
    shape_ZYX: [size, size, size],
    chunks_ZYX: [chunk, chunk, chunk],
    downsample_ZYX: [2 ** level, 2 ** level, 2 ** level],
    offset_ZYX: [0, 0, 0],
  }));
  const viewer = {
    memoryVolume: (name, zeroed) => new MemoryVolume(name, levels, zeroed),
    addLayer: () => ({ changed() {} }),
  };
  const scales = levels.map(({ downsample_ZYX }) => ({ scale_ZYX: downsample_ZYX, translation_ZYX: [0, 0, 0] }));
  const ct = { rootUrl: "memory://ct/", scales, metadata: { physicalUnits_ZYX: UNITS } };
  return new Labels(viewer, ct, recorder(), (length) => new Uint8Array(length));
}

test("levels of voxel 1, 2 and 4 in one chunk shape pool by 2; a level of voxel 3 over one of 2 is refused", () => {
  const scales = [1, 2, 4].map((size) => ({ ...LEVEL, downsample_ZYX: [size, size, size] }));
  assert.deepEqual(poolRatios(scales), [
    [1, 1, 1],
    [2, 2, 2],
    [2, 2, 2],
  ]);
  scales[2] = { ...LEVEL, downsample_ZYX: [3, 3, 3] };
  assert.throws(() => poolRatios(scales), {
    message:
      "CT level 2 (voxel 3×3×3, origin 0,0,0, chunk 64×64×64) does not pool whole blocks of level 1 " +
      "(2×2×2, 0,0,0, 64×64×64).",
  });
});

test("the 1×2×2 box at 1,1,1 of a 2×3×4 array holds 17, 18, 21 and 22", () => {
  const data_DHW = Uint8Array.from({ length: 24 }, (_, index) => index);
  assert.deepEqual([...cropped(data_DHW, [2, 3, 4], [1, 1, 1], [1, 2, 2])], [17, 18, 21, 22]);
});

test("pooling 2×2×3 voxels by 2 keeps each block's largest label, and a half block pools alone", () => {
  const data_DHW = Uint8Array.of(0, 5, 7, 1, 0, 0, 0, 2, 0, 0, 3, 0);
  assert.deepEqual(pooled(data_DHW, [2, 2, 3], [0, 0, 0], [2, 2, 3], [2, 2, 2]), {
    shape_ZYX: [1, 1, 2],
    out_DHW: Uint8Array.of(5, 7),
  });
});

test("chunk 0,10,3 of level 2 is at 2/0/10/3, and neither a document nor a padded number is a chunk", () => {
  assert.equal(chunkPath(2, [0, 10, 3]), "2/0/10/3");
  assert.deepEqual(pathChunk("2/0/10/3"), { level: 2, chunk_ZYX: [0, 10, 3] });
  assert.equal(pathChunk("0/.zarray"), null);
  assert.equal(pathChunk("0/01/0/0"), null);
});

test("saved labels unzip to the chunks and the largest label they were zipped with", async () => {
  const encoder = new TextEncoder();
  const documents = Object.entries(storeDocuments(SMALL, UNITS)).map(([name, value]) => ({
    name,
    bytes: encoder.encode(JSON.stringify(value)),
  }));
  const chunk_DHW = Uint8Array.of(0, 0, 3, 0, 0, 1, 0, 0);
  const bytes = await through(chunk_DHW, new CompressionStream("gzip"));
  const zip = new Uint8Array(
    await (await zipped([...documents, { name: "0/1/1/2", bytes }], pause, false)).arrayBuffer(),
  );
  assert.deepEqual(
    (await unzipped(zip)).map(({ name }) => name),
    [".zgroup", ".zattrs", "0/.zarray", "0/1/1/2"],
  );
  const { chunks, packed, classes } = await readLabels(zip, SMALL, UNITS);
  assert.deepEqual(chunks, new Map([["0/1/1/2", chunk_DHW]]));
  assert.deepEqual(packed, new Map([["0/1/1/2", bytes]]));
  assert.equal(
    packed.get("0/1/1/2").buffer.byteLength,
    bytes.length,
    "file.js readLabels keeps each chunk's bytes in a buffer of their own, not a view of the zip, which IndexedDB " +
      "would keep whole with each chunk",
  );
  assert.equal(classes, 3);
  const other = [{ ...SMALL[0], shape_ZYX: [3, 4, 6] }];
  await assert.rejects(readLabels(zip, other, UNITS), { message: "0/.zarray shape is [3,4,5], not [3,4,6]." });
  const outside = new Uint8Array(
    await (await zipped([...documents, { name: "0/2/0/0", bytes }], pause, false)).arrayBuffer(),
  );
  await assert.rejects(readLabels(outside, SMALL, UNITS), { message: "Chunk 0/2/0/0 is outside this CT." });
});

test("labels with no chunks stay empty, and gzip chunks must decode to exactly the CT's 8 bytes", async () => {
  const documents = Object.entries(storeDocuments(SMALL, UNITS)).map(([name, value]) => ({
    name,
    bytes: new TextEncoder().encode(JSON.stringify(value)),
  }));
  const empty = new Uint8Array(await (await zipped(documents, pause, false)).arrayBuffer());
  assert.deepEqual(await readLabels(empty, SMALL, UNITS), { chunks: new Map(), packed: new Map(), classes: 0 });
  for (const count of [0, 7, 9, 1024 * 1024]) {
    const bytes = await through(new Uint8Array(count), new CompressionStream("gzip"));
    const zip = new Uint8Array(
      await (await zipped([...documents, { name: "0/0/0/0", bytes }], pause, false)).arrayBuffer(),
    );
    await assert.rejects(readLabels(zip, SMALL, UNITS), {
      message: count < 8 ? `Chunk 0/0/0/0 has ${count} bytes, not 8.` : "Chunk 0/0/0/0 expands past its 8 bytes.",
    });
  }
  const bytes = (await through(new Uint8Array(1024 * 1024), new CompressionStream("gzip"))).subarray(0, -8);
  const truncated = new Uint8Array(
    await (await zipped([...documents, { name: "0/0/0/0", bytes }], pause, false)).arrayBuffer(),
  );
  await assert.rejects(readLabels(truncated, SMALL, UNITS), { message: "Chunk 0/0/0/0 expands past its 8 bytes." });
  const broken = await through(new Uint8Array(8), new CompressionStream("gzip"));
  broken[0] = 0xff;
  const corrupt = new Uint8Array(
    await (await zipped([...documents, { name: "0/0/0/0", bytes: broken }], pause, false)).arrayBuffer(),
  );
  await assert.rejects(readLabels(corrupt, SMALL, UNITS), { name: "TypeError", code: "Z_DATA_ERROR" });
});

test("a 64³ label chunk in more than 64 KiB of gzip imports every voxel across compressed input pieces", async () => {
  const levels = [{ ...SMALL[0], shape_ZYX: [64, 64, 64], chunks_ZYX: [64, 64, 64] }];
  // Seeded xorshift bytes keep the gzip large enough to span several input pieces.
  let value = 1;
  const chunk_DHW = Uint8Array.from({ length: 64 ** 3 }, () => {
    value ^= value << 13;
    value ^= value >>> 17;
    value ^= value << 5;
    return value & 255;
  });
  const documents = Object.entries(storeDocuments(levels, UNITS)).map(([name, value]) => ({
    name,
    bytes: new TextEncoder().encode(JSON.stringify(value)),
  }));
  const bytes = await through(chunk_DHW, new CompressionStream("gzip"));
  assert.ok(bytes.length > 64 * 1024, `${bytes.length} gzip bytes span multiple input pieces`);
  const zip = new Uint8Array(
    await (await zipped([...documents, { name: "0/0/0/0", bytes }], pause, false)).arrayBuffer(),
  );
  const { chunks, classes } = await readLabels(zip, levels, UNITS);
  assert.deepEqual(chunks, new Map([["0/0/0/0", chunk_DHW]]));
  assert.equal(classes, 255);
});

test("the largest label of 23 voxels is found in each byte of a word and in the 3 past the last word", () => {
  const data_DHW = new Uint8Array(23);
  assert.equal(largestLabel(data_DHW, 0), 0);
  assert.equal(largestLabel(data_DHW, 9), 9);
  data_DHW[6] = 3;
  data_DHW[11] = 5;
  assert.equal(largestLabel(data_DHW, 0), 5);
  data_DHW[21] = 200;
  assert.equal(largestLabel(data_DHW, 0), 200);
  data_DHW[0] = 255;
  assert.equal(largestLabel(data_DHW, 0), 255);
});

test("a 3³ chunk whose one label is its last voxel is saved, not left out as all 0", async () => {
  const labels = labelsOn([3], 3);
  labels.volume.write(0, [2, 2, 2], [1, 1, 1], Uint8Array.of(7));
  labels.drawn();
  const saved = (await labels.savedEntries()).find(({ name }) => name === chunkPath(0, [0, 0, 0]));
  assert.deepEqual([...(await through(saved.bytes, new DecompressionStream("gzip")))], [...Array(26).fill(0), 7]);
});

test("label 7 at finest voxel 24,24,24 in chunks of 24³ reaches voxel 12,12,12 of the level of voxel 2", async () => {
  const volume = labelsOn([48, 24], 24).volume;
  const pyramid = new Pyramid(volume);
  volume.write(0, [24, 24, 24], [1, 1, 1], Uint8Array.of(7));
  pyramid.followRuns([1, 1, 1], 2, [0], [1], 7, false);
  await pyramid.followed(pause, () => {});
  assert.equal(volume.chunk(1, [0, 0, 0])[(12 * 24 + 12) * 24 + 12], 7);
});

test("a 4 under a 9 leaves the levels over it at 9, and the 9 erased pools them to the 4", async () => {
  const volume = labelsOn([8, 4, 2], 4).volume;
  const pyramid = new Pyramid(volume);
  const coarser = () => [1, 2].map((level) => volume.chunk(level, [0, 0, 0])[0]);
  for (const [offset, label, raised, levels] of [
    [0, 9, 9, [9, 9]],
    [21, 4, 4, [9, 9]],
    [0, 0, null, [4, 4]],
  ]) {
    volume.writeRuns(0, [0, 0, 0], 2, [offset], [1], [label]);
    pyramid.followRuns([0, 0, 0], 2, [offset], [1], raised, false);
    await pyramid.followed(pause, () => {});
    assert.deepEqual(coarser(), levels);
  }
});

test("a CT of 2^21 + 1 voxels a side is refused, whose tiles would not have a key each", () => {
  assert.throws(() => labelsOn([2 ** 21 + 1], 2), {
    message: "CT level 0 is 2097153×2097153×2097153 voxels; paint takes at most 2097152 a side.",
  });
});

test("the gate at 128 of 255 is 128 of a uint8 CT and 2024 of one drawn from 1000 to 3040", () => {
  assert.equal(rawThreshold({ kind: "identity_uint8" }, 128), 128);
  assert.equal(rawThreshold({ kind: "linear_window", minimum: 1000, maximum: 3040 }, 128), 2024);
});

const GATE_CT = {
  rootUrl: "memory://ct/",
  scales: [
    { shape_ZYX: [4, 4, 4], chunks_ZYX: [2, 2, 2], downsample_ZYX: [1, 1, 1] },
    { shape_ZYX: [2, 2, 2], chunks_ZYX: [2, 2, 2], downsample_ZYX: [2, 2, 2] },
  ],
  metadata: { displayEncoding: { kind: "identity_uint8" } },
};

function opener(read, counts) {
  return async (url) => {
    assert.equal(url, GATE_CT.rootUrl);
    counts.opened += 1;
    return { scales: GATE_CT.scales, getChunk: read, dispose: () => (counts.disposed += 1) };
  };
}

test("the gate at 35 judges a finest chunk a pane holds on it, and one it holds only coarser on that", () => {
  const finest = Uint8Array.from({ length: 8 }, (_, index) => index * 10);
  const coarse = Uint8Array.from({ length: 8 }, (_, index) => (index === 7 ? 50 : 0));
  const held = (level, chunk_ZYX) => (level === 0 ? (chunk_ZYX.join() === "0,0,0" ? finest : null) : coarse);
  const gate = new Gate(GATE_CT, held, null, 35);
  assert.equal(gate.passes(1, 0, 0), true);
  assert.equal(gate.passes(0, 1, 1), false);
  assert.equal(gate.passes(3, 3, 3), true);
  assert.equal(gate.passes(3, 3, 2), true);
  assert.equal(gate.passes(3, 1, 3), false);
});

test("where no pane holds the CT the gate reads the coarsest level once, through a reader it opens and lets go", async () => {
  const counts = { opened: 0, disposed: 0, reads: 0 };
  const read = async () => {
    counts.reads += 1;
    return new Uint8Array(8).fill(40);
  };
  const gate = new Gate(GATE_CT, () => null, opener(read, counts), 35);
  assert.equal(gate.passes(0, 0, 0), null);
  await gate.load([0, 0, 0], [4, 4, 4]);
  assert.equal(gate.passes(3, 3, 3), true);
  await gate.load([0, 0, 0], [4, 4, 4]);
  gate.release();
  assert.deepEqual(counts, { opened: 1, disposed: 1, reads: 1 });
});

test("a read that fails is not held, so the next load reads it again", async () => {
  const counts = { opened: 0, disposed: 0, reads: 0 };
  const read = async () => {
    counts.reads += 1;
    if (counts.reads === 1) throw new Error("503");
    return new Uint8Array(8);
  };
  const gate = new Gate(GATE_CT, () => null, opener(read, counts), 35);
  await assert.rejects(gate.load([0, 0, 0], [1, 1, 1]), { message: "503" });
  await gate.load([0, 0, 0], [1, 1, 1]);
  assert.equal(gate.passes(0, 0, 0), false);
  assert.deepEqual(counts, { opened: 1, disposed: 0, reads: 2 });
});

test("a gated fill whose read fails reports it and lets go of the gate's reader", async () => {
  const counts = { opened: 0, disposed: 0 };
  const read = async () => {
    throw new Error("503");
  };
  const gate = new Gate(GATE_CT, () => null, opener(read, counts), 35);
  const errors = [];
  const brush = new Brush([1, 1, 1], (error) => errors.push(error.message), null);
  await brush.fill(labelsOn([4, 2], 2), gate, { point_ZYX: [1, 1, 1], normal_ZYX: [1, 0, 0] }, 1);
  assert.deepEqual(errors, ["503"]);
  assert.deepEqual(counts, { opened: 1, disposed: 1 });
});

function runsOf(edit) {
  return [...piecesOf(edit)].map(([index, { axis, offsets_N, lengths_N, values_N }]) => [
    index,
    axis,
    ...[offsets_N, lengths_N, values_N].map((array) => [...array]),
  ]);
}

test("runs of chunk 5 along x, of 9 along y and of 5 again pack into 3 pieces of 108 bytes, and a version 4 edit into 2", () => {
  const pieces = new Pieces();
  pieces.enter(5, 2, 1);
  pieces.add(6, 1, 0);
  pieces.add(7, 1, 3);
  pieces.enter(9, 1, 8);
  pieces.add(2, 3, 1);
  pieces.enter(5, 2, 1);
  pieces.add(8, 1, 3);
  pieces.enter(5, 2, 1);
  pieces.add(9, 1, 3);
  const edit = pieces.packed();
  assert.deepEqual(
    [edit.chunks_N3, edit.offsets_N, edit.lengths_N, edit.values_N, edit.bytes],
    [
      Float64Array.of(5, 2, 0, 9, 1, 2, 5, 2, 3),
      Uint32Array.of(6, 7, 2, 8),
      Uint32Array.of(1, 1, 3, 2),
      Uint8Array.of(0, 3, 1, 3),
      108,
    ],
  );
  assert.equal(edit.offsets_N.buffer.byteLength, 16);
  assert.deepEqual(runsOf(pieces.views()), runsOf(edit));
  assert.deepEqual(runsOf(edit), [
    [5, 2, [6, 7], [1, 1], [0, 3]],
    [9, 1, [2], [3], [1]],
    [5, 2, [8], [2], [3]],
  ]);
});

test("a history of 10 bytes holds the last two edits of 4, and a new edit forgets what was undone", () => {
  const edit = (id, bytes = 4) => ({ id, bytes });
  const history = new History(10);
  for (const id of [0, 1, 2]) history.record(edit(id));
  assert.deepEqual(history.head(), { done: [1, 2], undone: [] });
  assert.equal(history.undo().id, 2);
  assert.equal(history.redo().id, 2);
  assert.equal(history.undo().id, 2);
  assert.deepEqual(history.head(), { done: [1], undone: [2] });
  history.record(edit(3));
  assert.equal(history.redo(), null);
  assert.deepEqual(history.head(), { done: [1, 3], undone: [] });
  history.record(edit(4, 20));
  assert.deepEqual(history.head(), { done: [4], undone: [] });
  assert.deepEqual(history.dropped(), [0, 2, 1, 3]);
  assert.deepEqual(history.dropped(), []);
});

test("a history brought back holds the edits its head names and forgets the rest, and none when one is missing", () => {
  const edits = new Map([0, 1, 2, 3].map((id) => [id, { id, bytes: 4 }]));
  const history = new History(100);
  history.restore(edits, { done: [1, 2], undone: [3] });
  assert.deepEqual(history.head(), { done: [1, 2], undone: [3] });
  assert.deepEqual(history.dropped(), [0]);
  const lacking = new History(100);
  lacking.restore(edits, { done: [1, 5], undone: [] });
  assert.deepEqual(lacking.head(), { done: [], undone: [] });
  assert.deepEqual(lacking.dropped(), [0, 1, 2, 3]);
});

function voxelNames(voxels_N3) {
  return Array.from({ length: voxels_N3.length / 3 }, (_, index) =>
    voxels_N3.slice(index * 3, index * 3 + 3).join(),
  ).sort();
}

test("a fill in a 5×5 slice walled at x 2 takes the 10 voxels left of the wall, and none past a limit of 9", async () => {
  const open = (_z, _y, x) => x !== 2;
  const voxels_N3 = await planeFill([0, 2, 0], [1, 0, 0], [1, 5, 5], open, null, 100, pause);
  assert.deepEqual(voxelNames(voxels_N3), [
    ...["0,0,0", "0,0,1", "0,1,0", "0,1,1", "0,2,0", "0,2,1", "0,3,0", "0,3,1", "0,4,0", "0,4,1"],
  ]);
  assert.equal(await planeFill([0, 2, 0], [1, 0, 0], [1, 5, 5], open, null, 9, pause), null);
});

test("a fill waits for the rows it cannot yet tell, reading row 3 and then row 4", async () => {
  const read = new Set();
  const loads = [];
  const open = (_z, y, x) => (x === 2 ? false : y < 3 || read.has(y) ? true : null);
  const load = async (voxels_N3) => {
    loads.push([...new Set(voxelNames(voxels_N3).map((name) => name.split(",")[1]))]);
    for (let index = 1; index < voxels_N3.length; index += 3) read.add(voxels_N3[index]);
  };
  const voxels_N3 = await planeFill([0, 0, 0], [1, 0, 0], [1, 5, 5], open, load, 100, pause);
  assert.equal(voxels_N3.length / 3, 10);
  assert.deepEqual(loads, [["3"], ["4"]]);
});

test("a fill on the plane y + x = 8 of an 8³ volume takes its 56 voxels", async () => {
  const voxels_N3 = await planeFill(
    [4, 4, 4],
    [0, Math.SQRT1_2, Math.SQRT1_2],
    [8, 8, 8],
    () => true,
    null,
    1000,
    pause,
  );
  assert.equal(voxels_N3.length / 3, 56);
  assert.ok(voxelNames(voxels_N3).every((name) => name.split(",")[1] * 1 + name.split(",")[2] * 1 === 8));
});

test("a fill on a plane 16° off xz takes the voxel nearest it of each z and x, through a gap of one voxel", async () => {
  // The plane 0.28 z + 0.96 y = 9.92 through voxel 8, 8, 8, never half way between two y, walled at x 4 but for z 8.
  const open = (z, _y, x) => x !== 4 || z === 8;
  const voxels_N3 = await planeFill([8, 8, 8], [0.28, 0.96, 0], [16, 16, 16], open, null, 1000, pause);
  assert.equal(voxels_N3.length / 3, 16 * 16 - 15);
  for (let index = 0; index < voxels_N3.length; index += 3)
    assert.equal(voxels_N3[index + 1], Math.round((9.92 - 0.28 * voxels_N3[index]) / 0.96));
});

test("fill voxels either side of a tile's edge are two tiles cut to them", async () => {
  const tiles = await voxelTiles(Int32Array.of(0, 0, TILE - 1, 0, 0, TILE, 0, 1, TILE), pause);
  assert.deepEqual(
    tiles.map(({ from_ZYX, shape_ZYX, runs_N4 }) => [from_ZYX, shape_ZYX, [...runs_N4]]),
    [
      [
        [0, 0, TILE - 1],
        [1, 1, 1],
        [0, 0, 0, 1],
      ],
      [
        [0, 0, TILE],
        [1, 2, 1],
        [0, 0, 0, 1, 0, 1, 0, 1],
      ],
    ],
  );
});

test("a stroke of 5 over a row of 0, 3, 0, 5 across two chunks holds 3 runs, is taken back to it and done again", async () => {
  const labels = labelsOn([4, 2], 2);
  labels.volume.write(0, [1, 1, 1], [1, 1, 1], Uint8Array.of(3));
  labels.volume.write(0, [1, 1, 3], [1, 1, 1], Uint8Array.of(5));
  const row = () => [0, 1, 2, 3].map((x) => labels.lookup()(1, 1, x));
  labels.beginStroke();
  labels.stamp(
    { from_ZYX: [0, 0, 0], shape_ZYX: [1, 1, 4], axis: 2, runs_N4: Uint8Array.of(0, 0, 0, 4) },
    [1, 1, 0],
    5,
    null,
  );
  labels.endStroke(5);
  await labels.followed(pause);
  assert.deepEqual(row(), [5, 5, 5, 5]);
  const coarse = () => [labels.volume.chunk(1, [0, 0, 0])[0], labels.volume.chunk(1, [0, 0, 0])[1]];
  assert.deepEqual(coarse(), [5, 5]);
  assert.equal(labels.history.done[0].bytes, 75);
  assert.equal(await labels.turn(true, pause), true);
  await labels.followed(pause);
  assert.deepEqual(row(), [0, 3, 0, 5]);
  assert.deepEqual(coarse(), [3, 5]);
  assert.deepEqual(labels.history.head(), { done: [], undone: [0] });
  assert.equal(await labels.turn(false, pause), true);
  assert.deepEqual(row(), [5, 5, 5, 5]);
  assert.equal(await labels.turn(false, pause), false);
});

const VOXEL = { from_ZYX: [0, 0, 0], shape_ZYX: [1, 1, 1], axis: 2, runs_N4: Uint8Array.of(0, 0, 0, 1) };
const ROW = { from_ZYX: [0, 0, 0], shape_ZYX: [1, 1, 4], axis: 2, runs_N4: Uint8Array.of(0, 0, 0, 4) };

test("a stroke's journal entry and its undo's name its edit, and over the chunks as they were give it and the row back", async () => {
  const labels = labelsOn([4, 2], 2);
  labels.volume.write(0, [1, 1, 1], [1, 1, 1], Uint8Array.of(3));
  labels.volume.write(0, [1, 1, 3], [1, 1, 1], Uint8Array.of(5));
  const before = new Map(["0/0/0/0", "0/0/0/1"].map((path, cx) => [path, labels.volume.chunk(0, [0, 0, cx]).slice()]));
  labels.beginStroke();
  labels.stamp(ROW, [1, 1, 0], 5, null);
  labels.endStroke(5);
  await labels.turn(true, pause);
  const [stroke, undo] = labels.kept.keeping.writes;
  assert.deepEqual(
    [stroke, undo].map(({ entry, head }) => [entry, head]),
    [
      [
        { edit: 0, back: false },
        { done: [0], undone: [] },
      ],
      [
        { edit: 0, back: true },
        { done: [], undone: [0] },
      ],
    ],
  );
  assert.equal(undo.edit, null);
  assert.equal(stroke.edit.value, 5);
  assert.deepEqual(runsOf(stroke.edit), [
    [0, 2, [6, 7], [1, 1], [0, 3]],
    [1, 2, [6], [1], [0]],
  ]);
  const edits = new Map([[0, stroke.edit]]);
  const row = (labels) => [0, 1, 2, 3].map((x) => labels.lookup()(1, 1, x));
  for (const [written, shown] of [
    [[stroke], [5, 5, 5, 5]],
    [
      [stroke, undo],
      [0, 3, 0, 5],
    ],
  ]) {
    const restored = labelsOn([4, 2], 2);
    const entries = written.map(({ sequence, entry }) => ({ sequence, entry }));
    assert.deepEqual(await restored.kept.restore(await kept(before), entries, edits, written.at(-1).head, pause), {
      largest: 5,
      replayed: written.length,
      error: null,
    });
    await restored.followed(pause);
    assert.deepEqual(row(restored), shown);
    assert.deepEqual([...restored.volume.chunk(1, [0, 0, 0]).subarray(0, 2)], [Math.max(...shown.slice(0, 2)), 5]);
    assert.equal(restored.kept.sequence, written.length);
    assert.equal(restored.kept.next, 1);
    assert.equal(await restored.turn(written.length === 1, pause), true);
    assert.deepEqual(row(restored), written.length === 1 ? [0, 3, 0, 5] : [5, 5, 5, 5]);
  }
});

test("a stroke kept in IndexedDB is not undone over labels it did not leave: undo's history is forgotten", async () => {
  const labels = labelsOn([4, 2], 2);
  labels.beginStroke();
  labels.stamp(ROW, [1, 1, 0], 5, null);
  labels.endStroke(5);
  const [{ edit, head }] = labels.kept.keeping.writes;
  const restored = labelsOn([4, 2], 2);
  await restored.kept.restore(new Map(), [], new Map([[0, edit]]), head, pause);
  await assert.rejects(restored.turn(true, pause), {
    message: "the labels are not as edit 0 kept in this browser left them, so undo's history is forgotten.",
  });
  assert.deepEqual(restored.history.head(), { done: [], undone: [] });
  assert.deepEqual(restored.kept.keeping.writes, [{ forgot: true }]);
  assert.deepEqual(
    [0, 1, 2, 3].map((x) => restored.lookup()(1, 1, x)),
    [0, 0, 0, 0],
  );
});

test("a replay stops at an entry of an edit IndexedDB lacks, says why, and writes next past what it read", async () => {
  const labels = labelsOn([4, 2], 2);
  labels.beginStroke();
  labels.stamp(VOXEL, [1, 1, 0], 5, null);
  labels.endStroke(5);
  labels.beginStroke();
  labels.stamp(VOXEL, [1, 1, 2], 7, null);
  labels.endStroke(7);
  await labels.turn(true, pause);
  const [first, , undo] = labels.kept.keeping.writes;
  const restored = labelsOn([4, 2], 2);
  const entries = [first, undo].map(({ sequence, entry }) => ({ sequence, entry }));
  const { largest, replayed, error } = await restored.kept.restore(
    new Map(),
    entries,
    new Map([[0, first.edit]]),
    undo.head,
    pause,
  );
  assert.deepEqual([largest, replayed], [5, 1]);
  assert.equal(
    error.message,
    "IndexedDB's journal entry 2 is of edit 1, which it does not hold, so the labels are brought back as they were " +
      "before it.",
  );
  assert.deepEqual(
    [0, 1, 2, 3].map((x) => restored.lookup()(1, 1, x)),
    [5, 0, 0, 0],
  );
  assert.deepEqual([restored.kept.sequence, restored.kept.next], [3, 2]);
  assert.deepEqual(restored.history.head(), { done: [], undone: [] });
});

test("a restore whose layer is hidden after its first chunk, as another volume opened, stops there with null", async () => {
  const restored = labelsOn([4, 2], 2);
  const chunks = await kept(new Map(["0/0/0/0", "0/0/0/1"].map((path) => [path, new Uint8Array(8).fill(3)])));
  const hiding = async () => restored.hide();
  assert.equal(await restored.kept.restore(chunks, [], new Map(), NONE, hiding), null);
  assert.deepEqual([restored.volume.chunk(0, [0, 0, 0])[0], restored.volume.chunk(0, [0, 0, 1])], [3, null]);
});

test("a compaction packs what the edits wrote, never partway through one, and forgets the entries before it", async () => {
  const labels = labelsOn([4, 2], 2);
  labels.beginStroke();
  labels.stamp(ROW, [1, 1, 0], 5, null);
  labels.endStroke(5);
  await labels.followed(pause);
  labels.beginStroke();
  await labels.kept.keep(pause);
  labels.endStroke();
  assert.equal(labels.kept.keeping.writes.length, 1);
  const kept = labels.kept.keep(pause);
  labels.beginStroke();
  labels.stamp(VOXEL, [1, 1, 2], 7, null);
  await new Promise((resolve) => setTimeout(resolve, 20));
  labels.stamp(VOXEL, [1, 1, 3], 7, null);
  labels.endStroke(7);
  await kept;
  const [, second, { chunks, last, forgotten }] = labels.kept.keeping.writes;
  assert.equal(second.sequence, 1);
  assert.equal(last, 0);
  assert.deepEqual(forgotten, []);
  assert.deepEqual([...chunks.keys()], ["0/0/0/0", "0/0/0/1", "1/0/0/0"]);
  const packed = await through(chunks.get("0/0/0/1"), new DecompressionStream("gzip"));
  assert.deepEqual([...packed.subarray(6, 8)], [7, 7]);
});

test("a save takes the bytes a compaction or a restore packed, and gzips again a chunk written after its copy", async () => {
  const labels = labelsOn([4, 2], 2);
  labels.beginStroke();
  labels.stamp(ROW, [1, 1, 0], 5, null);
  labels.endStroke(5);
  await labels.followed(pause);
  const kept = labels.kept.keep(pause);
  labels.beginStroke();
  labels.stamp(VOXEL, [1, 1, 0], 7, null);
  labels.endStroke(7);
  await labels.followed(pause);
  await kept;
  const { chunks } = labels.kept.keeping.writes.at(-1);
  const saved = new Map((await labels.savedEntries()).map(({ name, bytes }) => [name, bytes]));
  assert.deepEqual(
    ["0/0/0/0", "0/0/0/1", "1/0/0/0"].map((path) => saved.get(path) === chunks.get(path)),
    [false, true, false],
  );
  const first = await through(saved.get("0/0/0/0"), new DecompressionStream("gzip"));
  assert.deepEqual([...first.subarray(6, 8)], [7, 5]);
  const restored = labelsOn([4, 2], 2);
  const read = new Map();
  for (const [path, bytes] of chunks)
    read.set(path, { data_DHW: await through(bytes, new DecompressionStream("gzip")), bytes });
  await restored.kept.restore(read, [], new Map(), NONE, pause);
  const again = (await restored.savedEntries()).filter(({ name }) => chunks.has(name));
  assert.deepEqual(
    again.map(({ name }) => name),
    ["0/0/0/0", "0/0/0/1", "1/0/0/0"],
  );
  assert.ok(again.every(({ name, bytes }) => bytes === chunks.get(name)));
});

test("a replay over levels 8³, 4³ and 2³ packed partway through a stroke of 7 over 1, or its undo, gives 7 or 0 at each", async () => {
  const chunk = (label) => Uint8Array.from({ length: 8 }, (_, index) => (index === 0 ? label : 0));
  const packed = {
    chunks_N3: Float64Array.of(0, 2, 0),
    offsets_N: Uint32Array.of(0),
    lengths_N: Uint32Array.of(1),
    values_N: Uint8Array.of(0),
  };
  const edits = new Map([[0, { value: 7, ...packed, bytes: 33 }]]);
  const stroke = new Map([
    ["0/0/0/0", chunk(7)],
    ["1/0/0/0", chunk(7)],
    ["2/0/0/0", chunk(1)],
  ]);
  const undo = new Map([["2/0/0/0", chunk(1)]]);
  for (const [chunks, back, label] of [
    [stroke, false, 7],
    [undo, true, 0],
  ]) {
    const restored = labelsOn([8, 4, 2], 2);
    const entries = [{ sequence: 0, entry: { edit: 0, back } }];
    await restored.kept.restore(await kept(chunks), entries, edits, { done: [], undone: [] }, pause);
    await restored.followed(pause);
    assert.deepEqual(
      [0, 1, 2].map((level) => restored.volume.chunk(level, [0, 0, 0])?.[0] ?? 0),
      [label, label, label],
    );
  }
});

test("an edit undone and then forgotten by a new stroke goes at the next compaction, with the entries naming it", async () => {
  const labels = labelsOn([4, 2], 2);
  labels.beginStroke();
  labels.stamp(VOXEL, [1, 1, 0], 5, null);
  labels.endStroke(5);
  await labels.turn(true, pause);
  labels.beginStroke();
  labels.stamp(VOXEL, [1, 1, 1], 6, null);
  labels.endStroke(6);
  await labels.followed(pause);
  await labels.kept.keep(pause);
  const [, , made, { last, forgotten }] = labels.kept.keeping.writes;
  assert.deepEqual(
    [made.entry, made.head],
    [
      { edit: 1, back: false },
      { done: [1], undone: [] },
    ],
  );
  assert.deepEqual([last, forgotten], [2, [0]]);
});

test("a compaction IndexedDB refuses gives back its chunks and the edits it would forget, and the next writes them", async () => {
  const labels = labelsOn([4, 2], 2);
  labels.beginStroke();
  labels.stamp(VOXEL, [1, 1, 0], 5, null);
  labels.endStroke(5);
  await labels.turn(true, pause);
  labels.beginStroke();
  labels.stamp(VOXEL, [1, 1, 1], 6, null);
  labels.endStroke(6);
  await labels.followed(pause);
  const asked = [];
  labels.kept.keeping.compact = async (_url, chunks, last, forgotten) => {
    asked.push({ chunks: [...chunks.keys()], last, forgotten });
    if (asked.length === 1) throw new Error("QuotaExceededError: the quota has been exceeded.");
  };
  await assert.rejects(labels.kept.keep(pause), { message: "QuotaExceededError: the quota has been exceeded." });
  await labels.kept.keep(pause);
  const written = { chunks: ["0/0/0/0", "1/0/0/0"], last: 2, forgotten: [0] };
  assert.deepEqual(asked, [written, written]);
});

function keepingOf(posted, reported) {
  return new Keeping(
    (message) => posted.push(message),
    () => assert.fail("a Keeping of the tests reads nothing"),
    (error) => reported.push(error.message),
    () => {},
  );
}

test("a read takes each reader's share of 6 chunks, backwards and at its own pace, in their paths' order", async () => {
  const paths = ["0/0/0/0", "0/0/0/1", "0/0/1/0", "0/1/0/0", "1/0/0/0", "1/0/0/1"];
  const kept = { entries: [{ sequence: 3, entry: { edit: 1, back: false } }], edits: [[1, { value: 5 }]], head: NONE };
  const reader = ({ read: [, part, parts] }, receive) => {
    const share = paths.map((path, index) => [index, path]).filter(([index]) => index % parts === part);
    setTimeout(() => {
      if (part === 0) receive({ kept });
      for (const [index, path] of share.reverse())
        receive({ chunk: [index, path, Uint8Array.of(index), Uint8Array.of(31, 139)] });
      receive({});
    }, parts - part);
    return () => {};
  };
  const keeping = new Keeping(
    () => {},
    reader,
    assert.fail,
    () => {},
  );
  const { chunks, entries, edits, head } = await keeping.read("memory://ct/");
  assert.deepEqual(
    [...chunks].map(([path, { data_DHW }]) => [path, data_DHW[0]]),
    paths.map((path, index) => [path, index]),
  );
  assert.deepEqual([entries, [...edits], head], [kept.entries, kept.edits, NONE]);
});

function unheld() {
  assert.fail("the page waited for a CT no other page holds");
}

test("an entry refused, or not written within 10 s, is reported and not kept until a compaction past it", async () => {
  const url = "memory://late/";
  const posted = [];
  const reported = [];
  const keeping = keepingOf(posted, reported);
  await keeping.hold(url, unheld, () => {});
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const settled = () => new Promise(setImmediate);
    const edit = async (sequence) => {
      const asked = Promise.withResolvers();
      keeping.owe(asked.promise);
      keeping.journal(url, sequence, { edit: sequence, back: false }, NONE, null);
      asked.resolve();
      await settled();
      return posted.at(-1).id;
    };
    keeping.answered({ id: await edit(0) });
    await settled();
    assert.equal(keeping.state(url), "kept");
    const unanswered = await edit(1);
    mock.timers.tick(ENTRY_LIMIT_MS - 1);
    await settled();
    assert.equal(keeping.state(url), "being kept");
    mock.timers.tick(1);
    await settled();
    assert.equal(keeping.state(url), "not kept");
    keeping.answered({ id: unanswered });
    keeping.answered({ id: await edit(2), error: "QuotaExceededError: the quota has been exceeded." });
    await settled();
    assert.equal(keeping.state(url), "not kept");
    assert.deepEqual(reported, [
      "IndexedDB has not written journal entry 1 within 10 s, so a reload may lose its edit.",
      "QuotaExceededError: the quota has been exceeded.",
    ]);
    const compacted = keeping.compact(url, new Map(), 2, []);
    keeping.answered({ id: posted.at(-1).id });
    await compacted;
    assert.equal(keeping.state(url), "kept");
  } finally {
    mock.timers.reset();
    keeping.letGo(url);
  }
});

test("once the worker stops, each write it has not answered and each one after fails with its error", async () => {
  const url = "memory://stopped/";
  const posted = [];
  const reported = [];
  const keeping = keepingOf(posted, reported);
  await keeping.hold(url, unheld, () => {});
  const settled = () => new Promise(setImmediate);
  const asked = Promise.withResolvers();
  keeping.owe(asked.promise);
  keeping.journal(url, 0, { edit: 0, back: false }, NONE, null);
  asked.resolve();
  const compacted = keeping.compact(url, new Map(), 0, []);
  keeping.stop(new Error("Paint's storage worker: it failed to load."));
  await assert.rejects(compacted, /it failed to load/);
  await assert.rejects(keeping.replace(url, new Map()), /it failed to load/);
  await settled();
  assert.equal(keeping.state(url), "not kept");
  assert.equal(posted.length, 2);
  assert.deepEqual(reported, ["Paint's storage worker: it failed to load."]);
  keeping.letGo(url);
});

test("a write of a CT this page does not hold is refused with its URL, before its hold and once it lets go", async () => {
  const url = "memory://refused/";
  const refusal = `this page does not hold the labels of ${url}, so it writes none of them.`;
  const posted = [];
  const reported = [];
  const keeping = keepingOf(posted, reported);
  const compacted = keeping.compact(url, new Map(), 0, []).catch((error) => error.message);
  keeping.journal(url, 0, { edit: 0, back: false }, NONE, null);
  await new Promise(setImmediate);
  assert.equal(posted.length, 0);
  assert.equal(await compacted, refusal);
  assert.equal(keeping.state(url), "not kept");
  await keeping.hold(url, unheld, () => {});
  const replaced = keeping.replace(url, new Map());
  keeping.answered({ id: posted.at(-1).id });
  await replaced;
  assert.equal(keeping.state(url), "kept");
  keeping.letGo(url);
  await assert.rejects(keeping.replace(url, new Map()), { message: refusal });
  assert.equal(posted.length, 1);
  assert.deepEqual(reported, [refusal]);
});

test("a second hold of a CT another page holds waits with the first, is handed the take, and holds it once let go", async () => {
  const url = "memory://asked/";
  const holder = keepingOf([], []);
  const posted = [];
  const reported = [];
  const waiter = keepingOf(posted, reported);
  try {
    await holder.hold(url, unheld, () => holder.letGo(url));
    const takes = [];
    const first = Promise.withResolvers();
    const holds = [
      waiter.hold(
        url,
        (take) => {
          takes.push(take);
          first.resolve();
        },
        () => {},
      ),
    ];
    await first.promise;
    holds.push(
      waiter.hold(
        url,
        (take) => takes.push(take),
        () => {},
      ),
    );
    const unsettled = new Promise((resolve) => setImmediate(() => resolve("waiting")));
    assert.equal(await Promise.race([...holds, unsettled]), "waiting");
    assert.equal(takes.length, 2);
    await assert.rejects(waiter.replace(url, new Map()), {
      message: `this page does not hold the labels of ${url}, so it writes none of them.`,
    });
    takes[1]();
    await Promise.all(holds);
    const replaced = waiter.replace(url, new Map());
    waiter.answered({ id: posted.at(-1).id });
    await replaced;
    assert.equal(posted.length, 1);
    assert.deepEqual(reported, []);
  } finally {
    holder.letGo(url);
    waiter.letGo(url);
  }
});

test("spares of 64³ go last first, 32 asked for at a time until 128 are held or asked for; 128³ drops them, 4 at a time", () => {
  const asked = [];
  const spares = new Spares((count, length) => asked.push([count, length]));
  const answered = () => {
    const [count, length] = asked.at(-1);
    const buffers = Array.from({ length: count }, () => new ArrayBuffer(length));
    spares.received(length, buffers);
    return buffers;
  };
  const chunk = 64 ** 3;
  assert.equal(spares.zeroed(chunk).length, chunk);
  let buffers = answered();
  for (let take = 0; take < 3; take++) {
    assert.equal(spares.zeroed(chunk).buffer, buffers.at(-1));
    buffers = answered();
  }
  assert.equal(spares.zeroed(chunk).buffer, buffers.at(-1));
  assert.deepEqual(asked, [
    [32, chunk],
    [32, chunk],
    [32, chunk],
    [32, chunk],
  ]);
  const other = spares.zeroed(128 ** 3);
  assert.equal(other.length, 128 ** 3);
  assert.ok(!buffers.includes(other.buffer));
  assert.deepEqual(asked.at(-1), [4, 128 ** 3]);
});
