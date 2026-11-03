// The zip stores its entries as they are, so zarr's ZipStore reads it.

import { unzipped } from "../../zip.js";

const GZIP_LEVEL = 6;
const GZIP_INPUT_BYTES = 64 * 1024;
function equalField(actual, expected, name) {
  if (JSON.stringify(actual) !== JSON.stringify(expected))
    throw new Error(`${name} is ${JSON.stringify(actual)}, not ${JSON.stringify(expected)}.`);
}

async function labelChunk(bytes, count, name) {
  const data_DHW = new Uint8Array(count);
  let at = 0;
  const reader = new ReadableStream({
    pull: (controller) => {
      const end = Math.min(at + GZIP_INPUT_BYTES, bytes.length);
      controller.enqueue(bytes.subarray(at, end));
      at = end;
      if (at === bytes.length) controller.close();
    },
  })
    .pipeThrough(new DecompressionStream("gzip"))
    .getReader();
  let length = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value.length > count - length) throw new Error(`Chunk ${name} expands past its ${count} bytes.`);
      data_DHW.set(value, length);
      length += value.length;
    }
  } catch (error) {
    await reader.cancel(error);
    throw error;
  } finally {
    reader.releaseLock();
  }
  if (length !== count) throw new Error(`Chunk ${name} has ${length} bytes, not ${count}.`);
  return data_DHW;
}

export function chunkPath(level, chunk_ZYX) {
  return `${level}/${chunk_ZYX.join("/")}`;
}

export function pathChunk(path) {
  const match = /^(0|[1-9]\d*)\/(0|[1-9]\d*)\/(0|[1-9]\d*)\/(0|[1-9]\d*)$/.exec(path);
  return match === null ? null : { level: Number(match[1]), chunk_ZYX: match.slice(2).map(Number) };
}

// A Response holds the bytes in the page; a Blob would register them with the browser in a call the page waits on, up
// to 99 ms once a compaction has made thousands.
export async function through(bytes, stream) {
  return new Uint8Array(await new Response(new Response(bytes).body.pipeThrough(stream)).arrayBuffer());
}

// Four voxels a word and the last few one by one, since a callback per voxel took 900 ms after a stroke of radius 256.
export function empty(data_DHW) {
  const words = new Uint32Array(data_DHW.buffer, data_DHW.byteOffset, Math.floor(data_DHW.length / 4));
  for (let index = 0; index < words.length; index++) if (words[index] !== 0) return false;
  for (let index = words.length * 4; index < data_DHW.length; index++) if (data_DHW[index] !== 0) return false;
  return true;
}

// For the 2,684 finest chunks of a stroke of radius 256 across the Surface zoomed out 26 notches, in node, a loop over
// every voxel took 1.6 s and a reduce 4.0 s, where four voxels a word takes 0.11 s.
export function largestLabel(data_DHW, largest) {
  const words = new Uint32Array(data_DHW.buffer, data_DHW.byteOffset, Math.floor(data_DHW.length / 4));
  let most = largest;
  for (let index = 0; index < words.length; index++) {
    const word = words[index];
    if (word !== 0) most = Math.max(most, word & 0xff, (word >>> 8) & 0xff, (word >>> 16) & 0xff, word >>> 24);
  }
  for (let index = words.length * 4; index < data_DHW.length; index++) most = Math.max(most, data_DHW[index]);
  return most;
}

export function storeDocuments(levels, units_ZYX) {
  const axes = ["z", "y", "x"].map((name, axis) =>
    units_ZYX[axis] === null ? { name, type: "space" } : { name, type: "space", unit: units_ZYX[axis] },
  );
  const datasets = levels.map((level, index) => ({
    path: String(index),
    coordinateTransformations: [
      { type: "scale", scale: level.scale_ZYX },
      { type: "translation", translation: level.translation_ZYX },
    ],
  }));
  const documents = {
    ".zgroup": { zarr_format: 2 },
    ".zattrs": { "image-label": {}, multiscales: [{ version: "0.4", axes, datasets }] },
  };
  for (const [index, level] of levels.entries()) {
    documents[`${index}/.zarray`] = {
      zarr_format: 2,
      shape: level.shape_ZYX,
      chunks: level.chunks_ZYX,
      dtype: "|u1",
      compressor: { id: "gzip", level: GZIP_LEVEL },
      fill_value: 0,
      order: "C",
      filters: null,
      dimension_separator: "/",
    };
  }
  return documents;
}

export async function readLabels(zip, levels, units_ZYX) {
  const entries = new Map((await unzipped(zip)).map(({ name, bytes }) => [name, bytes]));
  const expected = storeDocuments(levels, units_ZYX);
  const documents = new Map();
  for (const name of Object.keys(expected)) {
    if (!entries.has(name)) throw new Error(`The file has no ${name}: it is not labels of this CT.`);
    documents.set(name, JSON.parse(new TextDecoder().decode(entries.get(name))));
  }
  equalField(documents.get(".zgroup"), expected[".zgroup"], ".zgroup");
  equalField(documents.get(".zattrs"), expected[".zattrs"], ".zattrs");
  for (let level = 0; level < levels.length; level++) {
    const name = `${level}/.zarray`;
    const saved = documents.get(name);
    for (const key of ["zarr_format", "shape", "chunks", "dtype", "fill_value", "order", "filters"])
      equalField(saved[key], expected[name][key], `${name} ${key}`);
    equalField(saved.dimension_separator, "/", `${name} dimension_separator`);
    if (saved.compressor?.id !== "gzip")
      throw new Error(`${name} compressor ${JSON.stringify(saved.compressor)} is not gzip.`);
  }
  const chunks = new Map();
  const packed = new Map();
  let classes = 0;
  for (const [name, bytes] of entries) {
    if (documents.has(name) || name.endsWith("/")) continue;
    const chunk = pathChunk(name);
    if (chunk === null) throw new Error(`The file has ${name}, which is no label chunk.`);
    const { level, chunk_ZYX } = chunk;
    if (
      level >= levels.length ||
      chunk_ZYX.some((chunk, axis) => chunk * levels[level].chunks_ZYX[axis] >= levels[level].shape_ZYX[axis])
    )
      throw new Error(`Chunk ${name} is outside this CT.`);
    const count = levels[level].chunks_ZYX.reduce((product, size) => product * size, 1);
    const data_DHW = await labelChunk(bytes, count, name);
    chunks.set(name, data_DHW);
    // A copy of its own: IndexedDB keeps the whole buffer a view is of, and this one is of the zip.
    packed.set(name, bytes.slice());
    if (level === 0) classes = largestLabel(data_DHW, classes);
  }
  return { chunks, packed, classes };
}
