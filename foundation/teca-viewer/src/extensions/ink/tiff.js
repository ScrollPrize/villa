// The first range read: a TIFF written header, IFD and tile tables first, as Paris 4's ink image is in its first
// 11 KiB, needs no other before its tiles.
const HEAD_BYTES = 64 * 1024;
const IFD_ENTRY_LIMIT = 64;
const TILE_PIXEL_LIMIT = 4096 * 4096;
const TILE_LIMIT = 1_000_000;
// Twice the pixel bytes accommodate incompressible deflate blocks; excessive padding is refused.
const DEFLATE_BYTES_PER_PIXEL = 2;
const DEFLATE_EXTRA_BYTES = 1024;
const TAG = {
  imageWidth: 256,
  imageLength: 257,
  bitsPerSample: 258,
  compression: 259,
  photometric: 262,
  samplesPerPixel: 277,
  planarConfiguration: 284,
  predictor: 317,
  tileWidth: 322,
  tileLength: 323,
  tileOffsets: 324,
  tileByteCounts: 325,
  sampleFormat: 339,
};
const SHORT = 3;
const LONG = 4;

async function boundedBytes(stream, limit, name) {
  const reader = stream.getReader();
  const bytes = new Uint8Array(limit);
  let length = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value.length > limit - length) throw new Error(`${name} exceeds ${limit} bytes.`);
      bytes.set(value, length);
      length += value.length;
    }
  } catch (error) {
    await reader.cancel(error);
    throw error;
  } finally {
    reader.releaseLock();
  }
  return length === limit ? bytes : bytes.slice(0, length);
}

async function range(url, offset, length, signal) {
  const bytes = `bytes=${offset}-${offset + length - 1}`;
  const response = await fetch(url, { headers: { Range: bytes }, signal });
  if (response.status !== 206) {
    await response.body?.cancel();
    throw new Error(`${url} answered ${response.status} to a request for ${bytes}.`);
  }
  const declared = response.headers.get("Content-Length");
  if (declared !== null && Number(declared) > length) {
    await response.body.cancel();
    throw new Error(`${url} declared ${declared} bytes for ${bytes}, more than ${length}.`);
  }
  return boundedBytes(response.body, length, `${url}'s ${bytes}`);
}

async function exactRange(url, offset, length, signal) {
  const bytes = await range(url, offset, length, signal);
  if (bytes.length !== length) throw new Error(`${url} gave ${bytes.length} bytes at ${offset}, not ${length}.`);
  return bytes;
}

export async function tiffLayout(head, read) {
  const at = async (offset, length) => {
    if (offset < 8 || offset + length > 2 ** 32)
      throw new Error(`The TIFF's range of ${length} bytes at ${offset} is refused.`);
    return offset + length <= head.length ? head.subarray(offset, offset + length) : read(offset, length);
  };
  const view = (bytes) => new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (head.length < 8) throw new Error(`A TIFF header has ${head.length} bytes, fewer than 8.`);
  const order = String.fromCharCode(head[0], head[1]);
  if (order !== "II" && order !== "MM") throw new Error(`A TIFF starts with II or MM, not ${JSON.stringify(order)}.`);
  const little = order === "II";
  const header = view(head.subarray(0, 8));
  if (header.getUint16(2, little) !== 42) throw new Error(`TIFF version ${header.getUint16(2, little)} is not 42.`);
  const ifd = header.getUint32(4, little);
  const count = view(await at(ifd, 2)).getUint16(0, little);
  if (count < 1 || count > IFD_ENTRY_LIMIT) throw new Error(`A TIFF IFD of ${count} entries is refused.`);
  const entries = view(await at(ifd + 2, count * 12));
  const tags = new Map();
  for (let entry = 0; entry < count; entry++) {
    const type = entries.getUint16(entry * 12 + 2, little);
    const values = entries.getUint32(entry * 12 + 4, little);
    const value =
      type === SHORT && values === 1
        ? entries.getUint16(entry * 12 + 8, little)
        : entries.getUint32(entry * 12 + 8, little);
    tags.set(entries.getUint16(entry * 12, little), { type, values, value });
  }
  const scalar = (name, absent) => {
    const entry = tags.get(TAG[name]);
    if (entry === undefined && absent !== undefined) return absent;
    if (entry === undefined || entry.values !== 1 || (entry.type !== SHORT && entry.type !== LONG))
      throw new Error(`The TIFF's ${name} is ${JSON.stringify(entry ?? null)}, not one short or long.`);
    return entry.value;
  };
  const shape_HW = [scalar("imageLength"), scalar("imageWidth")];
  const tile_HW = [scalar("tileLength"), scalar("tileWidth")];
  const compression = scalar("compression", 1);
  const predictor = scalar("predictor", 1);
  const grey =
    scalar("bitsPerSample") === 8 &&
    scalar("samplesPerPixel", 1) === 1 &&
    scalar("sampleFormat", 1) === 1 &&
    scalar("photometric") === 1 &&
    scalar("planarConfiguration", 1) === 1;
  if (!grey) throw new Error("The TIFF is not one grey 8-bit sample a pixel, black at 0.");
  if (![1, 8, 32946].includes(compression) || ![1, 2].includes(predictor))
    throw new Error(`TIFF compression ${compression} with predictor ${predictor} is not read; 1 or deflate, 1 or 2.`);
  if (tile_HW.some((size) => size < 1 || size % 16 !== 0) || tile_HW[0] * tile_HW[1] > TILE_PIXEL_LIMIT)
    throw new Error(`TIFF tiles of ${tile_HW.join(" × ")} pixels are refused.`);
  const grid_HW = shape_HW.map((size, axis) => Math.ceil(size / tile_HW[axis]));
  const tiles = grid_HW[0] * grid_HW[1];
  if (shape_HW.some((size) => size < 1) || tiles > TILE_LIMIT)
    throw new Error(`A TIFF of ${shape_HW.join(" × ")} pixels in ${tiles} tiles is refused.`);
  const table = async (name) => {
    const entry = tags.get(TAG[name]);
    if (entry?.type !== LONG || entry.values !== tiles)
      throw new Error(`The TIFF's ${name} is ${JSON.stringify(entry ?? null)}, not ${tiles} longs.`);
    if (tiles === 1) return Uint32Array.of(entry.value);
    const bytes = view(await at(entry.value, tiles * 4));
    return Uint32Array.from({ length: tiles }, (_, index) => bytes.getUint32(index * 4, little));
  };
  const [offsets_N, counts_N] = await Promise.all([table("tileOffsets"), table("tileByteCounts")]);
  const pixels = tile_HW[0] * tile_HW[1];
  const encodedLimit = compression === 1 ? pixels : pixels * DEFLATE_BYTES_PER_PIXEL + DEFLATE_EXTRA_BYTES;
  for (let index = 0; index < tiles; index++) {
    const [offset, count] = [offsets_N[index], counts_N[index]];
    if (count < 1 || count > encodedLimit || (compression === 1 && count !== pixels))
      throw new Error(`TIFF tile ${index} has ${count} encoded bytes for ${pixels} pixels; limit ${encodedLimit}.`);
    if (offset < 8 || offset + count > 2 ** 32)
      throw new Error(`TIFF tile ${index}'s range of ${count} bytes at ${offset} is refused.`);
  }
  return { shape_HW, tile_HW, grid_HW, offsets_N, counts_N, deflated: compression !== 1, differenced: predictor === 2 };
}

export async function tilePixels(layout, bytes) {
  const [height, width] = layout.tile_HW;
  const pixels_HW = layout.deflated
    ? await boundedBytes(
        new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate")),
        height * width,
        `A TIFF tile of ${height} × ${width} pixels`,
      )
    : bytes;
  if (pixels_HW.length !== height * width)
    throw new Error(`A TIFF tile of ${height} × ${width} pixels decoded to ${pixels_HW.length} bytes.`);
  if (layout.differenced)
    for (let row = 0; row < height; row++)
      for (let at = row * width + 1; at < (row + 1) * width; at++) pixels_HW[at] = pixels_HW[at] + pixels_HW[at - 1];
  return pixels_HW;
}

export async function openTiff(url, signal) {
  const head = await range(url, 0, HEAD_BYTES, signal);
  return tiffLayout(head, (offset, length) => exactRange(url, offset, length, signal));
}

export async function readTile(url, layout, index, signal) {
  const bytes = await exactRange(url, layout.offsets_N[index], layout.counts_N[index], signal);
  return tilePixels(layout, bytes).catch((error) => {
    throw new Error(`Tile ${index} of ${url}: ${error.message}`, { cause: error });
  });
}
