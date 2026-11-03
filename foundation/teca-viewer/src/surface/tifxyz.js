import {
  SURFACE_GEOMETRY_PRIORITY,
  TIFF_TILE_BYTE_LIMIT,
  TIFXYZ_FILE_BYTE_LIMIT,
  TIFXYZ_META_BYTE_LIMIT,
  TIFXYZ_PIXEL_LIMIT,
} from "../config.js";
import { storableEtag, strongEtag } from "../volume/download-cache.js";
import { fetchWithRetry, MirrorRequestQueue } from "../volume/request-queue.js";
import { readResponseBytes } from "../volume/response-bytes.js";
import { TifxyzTiles } from "./tifxyz-tiles.js";

// A classic or BigTIFF header and a 64-entry IFD fit in this many bytes at either end of the file.
const LAYOUT_BLOCK_BYTES = 4096;
const IFD_ENTRY_LIMIT = 64;
const TIFF_SHORT = 3;
const TIFF_LONG = 4;
const TIFF_LONG8 = 16;
const TAG = Object.freeze({
  imageWidth: 256,
  imageLength: 257,
  bitsPerSample: 258,
  compression: 259,
  stripOffsets: 273,
  samplesPerPixel: 277,
  rowsPerStrip: 278,
  stripByteCounts: 279,
  planarConfiguration: 284,
  predictor: 317,
  tileWidth: 322,
  tileLength: 323,
  tileOffsets: 324,
  tileByteCounts: 325,
  sampleFormat: 339,
});
const LITTLE_ENDIAN_PLATFORM = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

function tifxyzDirectory(url) {
  return url.pathname.endsWith("/index.html") && url.hash !== ""
    ? new URL(decodeURIComponent(url.hash.slice(1)).replace(/^\/+/, ""), `${url.origin}/`)
    : url;
}

export function tifxyzUrl(path) {
  if (!/^https?:\/\//.test(path)) throw new Error(`${path} is not an HTTP(S) tifxyz URL.`);
  const directory = tifxyzDirectory(new URL(path));
  directory.search = "";
  directory.hash = "";
  const url = directory.toString().replace(/\/+$/, "");
  if (/\.(?:nml|obj|zarr)$/i.test(directory.pathname.replace(/\/+$/, "")))
    throw new Error(`${path} is not a tifxyz directory URL.`);
  return url;
}

async function readMeta(url, signal, queue) {
  const { bytes, version } = await queue.read(
    `${url}/meta.json`,
    async (place, readSignal) => {
      const response = await fetchWithRetry(place, readSignal, { cache: "no-store" });
      if (!response.ok) {
        await response.body?.cancel();
        throw new Error(`tifxyz metadata returned HTTP ${response.status}.`);
      }
      const bytes = await readResponseBytes(response, TIFXYZ_META_BYTE_LIMIT, readSignal);
      return {
        bytes,
        version: { etag: strongEtag(response.headers), totalBytes: bytes.byteLength, cacheable: false },
      };
    },
    0,
    { signal },
  );
  const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  let meta;
  try {
    meta = JSON.parse(text);
  } catch (error) {
    throw new Error(`${url}/meta.json is not valid JSON: ${error.message}`);
  }
  if (
    meta === null ||
    typeof meta !== "object" ||
    !Array.isArray(meta.scale) ||
    meta.scale.length !== 2 ||
    meta.scale.some((value) => typeof value !== "number" || !Number.isFinite(value) || value <= 0)
  ) {
    throw new Error(`${url}/meta.json must declare two positive finite scale values.`);
  }
  return { scale: meta.scale, version };
}

// S3 CORS hides Content-Range; the ETag from HEAD and the byte count check a range then.
async function fetchHttpRange(url, offset, length, signal, totalBytes, expectedEtag) {
  const end = offset + length - 1;
  const response = await fetchWithRetry(url, signal, {
    headers: { Range: `bytes=${offset}-${end}` },
    cache: "no-store",
  });
  if (response.status !== 206) {
    await response.body?.cancel();
    throw new Error(`${url} returned HTTP ${response.status} for bytes ${offset}-${end}, expected 206.`);
  }
  const etag = strongEtag(response.headers);
  if (expectedEtag !== null && etag !== expectedEtag) {
    await response.body?.cancel();
    throw new Error(`${url} changed during geometry reading.`);
  }
  const cacheable = storableEtag(response.headers) !== null;
  const contentRangeHeader = response.headers.get("Content-Range");
  if (contentRangeHeader !== null) {
    const contentRange = /^bytes (\d+)-(\d+)\/(\d+)$/.exec(contentRangeHeader);
    if (
      contentRange === null ||
      Number(contentRange[1]) !== offset ||
      Number(contentRange[2]) !== end ||
      Number(contentRange[3]) !== totalBytes
    ) {
      await response.body?.cancel();
      throw new Error(`${url} returned an invalid Content-Range for bytes ${offset}-${end}.`);
    }
  }
  const bytes = await readResponseBytes(response, length, signal);
  if (bytes.byteLength !== length) {
    throw new Error(`${url} returned ${bytes.byteLength} bytes for bytes ${offset}-${end}, expected ${length}.`);
  }
  return { bytes, totalBytes, etag, cacheable };
}

async function readTifxyzRange(url, offset, length, signal, { queue, priority, version, byteCache }) {
  signal?.throwIfAborted();
  if (version.etag !== null) {
    const bytes = await byteCache.get(url, version, offset, length, signal);
    signal?.throwIfAborted();
    if (bytes !== null) return bytes;
  }
  // A range is read no-store and stored under `url`, so every hostname of the file may serve it.
  const result = await queue.read(
    url,
    (place, readSignal) => fetchHttpRange(place, offset, length, readSignal, version.totalBytes, version.etag),
    priority,
    { signal },
  );
  if (result.etag !== null) await byteCache.put(url, result, offset, result.bytes, signal);
  signal?.throwIfAborted();
  return result.bytes;
}

async function fetchHttpHead(url, signal) {
  const response = await fetchWithRetry(url, signal, { method: "HEAD", cache: "no-store" });
  await response.body?.cancel();
  if (response.status !== 200) throw new Error(`${url} returned HTTP ${response.status} for HEAD, expected 200.`);
  const contentLength = response.headers.get("Content-Length");
  const totalBytes = Number(contentLength);
  if (contentLength === null || !Number.isSafeInteger(totalBytes) || totalBytes < 0)
    throw new Error(`${url} answered HEAD with Content-Length ${contentLength}, not a byte count.`);
  const etag = strongEtag(response.headers);
  return { totalBytes, etag, cacheable: storableEtag(response.headers) !== null };
}

export async function readTiffTags(url, totalBytes, read) {
  const header = await read(0, 8);
  const order = String.fromCharCode(header[0], header[1]);
  if (order !== "II" && order !== "MM") throw new Error(`${url} is not a TIFF file.`);
  const littleEndian = order === "II";
  const headerView = new DataView(header.buffer, header.byteOffset, 8);
  const magic = headerView.getUint16(2, littleEndian);
  if (magic !== 42 && magic !== 43) throw new Error(`${url} has an unsupported TIFF version ${magic}.`);
  const big = magic === 43;
  const uint64 = (view, offset) => {
    const value = Number(view.getBigUint64(offset, littleEndian));
    if (!Number.isSafeInteger(value)) throw new Error(`${url} has a TIFF integer outside the safe range.`);
    return value;
  };
  let ifdOffset = headerView.getUint32(4, littleEndian);
  if (big) {
    if (headerView.getUint16(4, littleEndian) !== 8 || headerView.getUint16(6, littleEndian) !== 0)
      throw new Error(`${url} has an invalid BigTIFF header.`);
    const bytes = await read(8, 8);
    ifdOffset = uint64(new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), 0);
  }
  const headerBytes = big ? 16 : 8;
  const countBytes = big ? 8 : 2;
  const offsetBytes = big ? 8 : 4;
  const entryBytes = big ? 20 : 12;
  if (ifdOffset < headerBytes || ifdOffset + countBytes + offsetBytes > totalBytes)
    throw new Error(`${url} has an invalid IFD offset.`);
  const ifdBytes = await read(
    ifdOffset,
    Math.min(countBytes + IFD_ENTRY_LIMIT * entryBytes + offsetBytes, totalBytes - ifdOffset),
  );
  const ifd = new DataView(ifdBytes.buffer, ifdBytes.byteOffset, ifdBytes.byteLength);
  const entryCount = big ? uint64(ifd, 0) : ifd.getUint16(0, littleEndian);
  const nextOffset = countBytes + entryCount * entryBytes;
  if (entryCount < 1 || entryCount > IFD_ENTRY_LIMIT || nextOffset + offsetBytes > ifdBytes.byteLength) {
    throw new Error(`${url} IFD has ${entryCount} entries, outside 1..${IFD_ENTRY_LIMIT}.`);
  }
  const tags = new Map();
  for (let entry = 0; entry < entryCount; entry++) {
    const offset = countBytes + entry * entryBytes;
    const tag = ifd.getUint16(offset, littleEndian);
    if (tags.has(tag)) throw new Error(`${url} repeats TIFF tag ${tag}.`);
    tags.set(tag, {
      type: ifd.getUint16(offset + 2, littleEndian),
      count: big ? uint64(ifd, offset + 4) : ifd.getUint32(offset + 4, littleEndian),
      offset: offset + (big ? 12 : 8),
    });
  }
  const unsigned = [TIFF_SHORT, TIFF_LONG, ...(big ? [TIFF_LONG8] : [])];
  return {
    littleEndian,
    headerBytes,
    has: (tag) => tags.has(tag),
    scalar(tag, fallback) {
      const entry = tags.get(tag);
      if (entry === undefined) {
        if (fallback === undefined) throw new Error(`${url} lacks TIFF tag ${tag}.`);
        return fallback;
      }
      if (entry.count !== 1 || !unsigned.includes(entry.type)) {
        throw new Error(`${url} TIFF tag ${tag} must hold one unsigned integer value.`);
      }
      return entry.type === TIFF_SHORT
        ? ifd.getUint16(entry.offset, littleEndian)
        : entry.type === TIFF_LONG
          ? ifd.getUint32(entry.offset, littleEndian)
          : uint64(ifd, entry.offset);
    },
    async table(tag, count, name) {
      const entry = tags.get(tag);
      if (entry === undefined || entry.count !== count || !unsigned.includes(entry.type))
        throw new Error(`${url} ${name} ${tag} must contain ${count} unsigned integers.`);
      const size = entry.type === TIFF_SHORT ? 2 : entry.type === TIFF_LONG ? 4 : 8;
      let view = ifd,
        start = entry.offset;
      if (count * size > offsetBytes) {
        const offset = big ? uint64(ifd, entry.offset) : ifd.getUint32(entry.offset, littleEndian);
        if (offset < headerBytes || offset + count * size > totalBytes)
          throw new Error(`${url} ${name} falls outside the file.`);
        const bytes = await read(offset, count * size);
        view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        start = 0;
      }
      return Array.from({ length: count }, (_, index) => {
        const offset = start + index * size;
        return size === 2
          ? view.getUint16(offset, littleEndian)
          : size === 4
            ? view.getUint32(offset, littleEndian)
            : uint64(view, offset);
      });
    },
  };
}

export async function gridStorage(url, tiff, height, width, totalBytes) {
  const dataBytes = height * width * Float32Array.BYTES_PER_ELEMENT;
  if ([TAG.tileWidth, TAG.tileLength, TAG.tileOffsets, TAG.tileByteCounts].some((tag) => tiff.has(tag))) {
    if ([TAG.stripOffsets, TAG.stripByteCounts, TAG.rowsPerStrip].some((tag) => tiff.has(tag)))
      throw new Error(`${url} is tiled but also declares strip storage.`);
    const tileWidth = tiff.scalar(TAG.tileWidth),
      tileHeight = tiff.scalar(TAG.tileLength);
    const compression = tiff.scalar(TAG.compression, 1),
      predictor = tiff.scalar(TAG.predictor, 1);
    if (tileWidth < 1 || tileHeight < 1 || tileWidth * tileHeight * 4 > TIFF_TILE_BYTE_LIMIT)
      throw new Error(`${url} tile exceeds the ${TIFF_TILE_BYTE_LIMIT / 2 ** 20} MiB decoded limit.`);
    if (![1, 5].includes(compression) || ![1, 3].includes(predictor))
      throw new Error(`${url} tiled float32 coordinates require uncompressed or LZW data and predictor 1 or 3.`);
    const count = Math.ceil(width / tileWidth) * Math.ceil(height / tileHeight);
    if (count > 65536) throw new Error(`${url} exceeds the 65536-tile metadata limit.`);
    const [tileOffsets, tileByteCounts] = await Promise.all([
      tiff.table(TAG.tileOffsets, count, "tile table"),
      tiff.table(TAG.tileByteCounts, count, "tile table"),
    ]);
    for (let index = 0; index < count; index++) {
      if (
        tileOffsets[index] < tiff.headerBytes ||
        tileByteCounts[index] < 1 ||
        tileByteCounts[index] > TIFF_TILE_BYTE_LIMIT ||
        tileOffsets[index] + tileByteCounts[index] > totalBytes
      )
        throw new Error(`${url} tile ${index} has an invalid bounded byte range.`);
    }
    return { tileWidth, tileHeight, tileOffsets, tileByteCounts, compression, predictor };
  }
  if (
    tiff.scalar(TAG.compression, 1) !== 1 ||
    tiff.scalar(TAG.predictor, 1) !== 1 ||
    tiff.scalar(TAG.rowsPerStrip, 0xffffffff) < height ||
    tiff.scalar(TAG.stripByteCounts) !== dataBytes
  ) {
    throw new Error(`${url} must store its coordinate grid as one uncompressed float32 strip.`);
  }
  const dataOffset = tiff.scalar(TAG.stripOffsets);
  if (dataOffset < tiff.headerBytes || dataOffset + dataBytes > totalBytes) {
    throw new Error(`${url} strip data falls outside the file.`);
  }
  return { dataOffset };
}

async function readTiffLayout(url, signal, queue, byteCache) {
  const { totalBytes, etag, cacheable } = await queue.read(url, fetchHttpHead, 0, { signal });
  if (totalBytes > TIFXYZ_FILE_BYTE_LIMIT)
    throw new Error(`${url} is ${totalBytes} bytes, exceeding the ${TIFXYZ_FILE_BYTE_LIMIT}-byte tifxyz file limit.`);
  const version = { etag, totalBytes, cacheable };
  const range = (offset, length) =>
    readTifxyzRange(url, offset, length, signal, { queue, priority: 0, version, byteCache });
  const ends = totalBytes <= 2 * LAYOUT_BLOCK_BYTES ? [0] : [0, totalBytes - LAYOUT_BLOCK_BYTES];
  const blocks = await Promise.all(
    ends.map(async (offset) => ({
      offset,
      bytes: await range(offset, Math.min(LAYOUT_BLOCK_BYTES, totalBytes - offset)),
    })),
  );
  const read = async (offset, length) => {
    const block = blocks.find(
      (block) => offset >= block.offset && offset + length <= block.offset + block.bytes.byteLength,
    );
    return block === undefined
      ? range(offset, length)
      : block.bytes.subarray(offset - block.offset, offset - block.offset + length);
  };
  const tiff = await readTiffTags(url, totalBytes, read);
  const width = tiff.scalar(TAG.imageWidth);
  const height = tiff.scalar(TAG.imageLength);
  if (width < 2 || height < 2) throw new Error(`${url} grid ${height}x${width} needs at least two rows and columns.`);
  if (height * width > TIFXYZ_PIXEL_LIMIT) {
    throw new Error(`${url} grid ${height}x${width} exceeds the ${TIFXYZ_PIXEL_LIMIT}-pixel tifxyz limit.`);
  }
  if (
    tiff.scalar(TAG.bitsPerSample) !== 32 ||
    tiff.scalar(TAG.sampleFormat, 1) !== 3 ||
    tiff.scalar(TAG.samplesPerPixel, 1) !== 1 ||
    tiff.scalar(TAG.planarConfiguration, 1) !== 1
  ) {
    throw new Error(`${url} must contain one two-dimensional float32 coordinate grid.`);
  }
  const storage = await gridStorage(url, tiff, height, width, totalBytes);
  return { url, width, height, ...storage, littleEndian: tiff.littleEndian, totalBytes, etag, cacheable };
}

export class Tifxyz {
  constructor(url, grid_HW, scale_UV, axes, metaVersion, queue, byteCache) {
    this.url = url;
    this.grid_HW = Object.freeze([...grid_HW]);
    this.scale_UV = Object.freeze([...scale_UV]);
    this.axes = axes;
    this.metaVersion = metaVersion;
    this.queue = queue;
    this.byteCache = byteCache;
    this.tiles = new TifxyzTiles((axis, offset, length, signal) => this.readRange(axis, offset, length, signal));
  }

  static async open(path, signal, byteCache) {
    const url = tifxyzUrl(path);
    const queue = new MirrorRequestQueue(url);
    const controller = new AbortController();
    const readSignal = AbortSignal.any([signal, controller.signal]);
    const failing = (read) =>
      read.catch((error) => {
        controller.abort(error);
        throw error;
      });
    try {
      const [meta, ...axes] = await Promise.all([
        failing(readMeta(url, readSignal, queue)),
        ...["x", "y", "z"].map((axis) => failing(readTiffLayout(`${url}/${axis}.tif`, readSignal, queue, byteCache))),
      ]);
      for (const axis of axes.slice(1)) {
        if (axis.height !== axes[0].height || axis.width !== axes[0].width) {
          throw new Error(
            `${axis.url} shape ${axis.height}x${axis.width} does not match tifxyz grid ${axes[0].height}x${axes[0].width}.`,
          );
        }
      }
      return new Tifxyz(url, [axes[0].height, axes[0].width], meta.scale, axes, meta.version, queue, byteCache);
    } catch (error) {
      queue.dispose();
      throw error;
    }
  }

  release() {
    this.tiles.clear();
    this.queue?.release();
    this.queue = null;
  }

  dispose() {
    this.tiles.dispose();
    this.queue?.dispose();
  }

  read(url, run, priority, options) {
    this.queue ??= new MirrorRequestQueue(this.url);
    return this.queue.read(url, run, priority, options);
  }

  get cacheKey() {
    const files = [this.metaVersion, ...this.axes];
    if (files.some((file) => file.etag === null)) return null;
    return JSON.stringify([this.url, ...files.map(({ etag, totalBytes }) => [etag, totalBytes])]);
  }

  async readRows(axisIndex, rowStart, rowCount, signal) {
    const axis = this.axes[axisIndex];
    const width = this.grid_HW[1];
    if (axis.tileOffsets !== undefined) return this.tiles.readRows(axis, axisIndex, rowStart, rowCount, width, signal);
    const rowBytes = width * Float32Array.BYTES_PER_ELEMENT;
    const bytes = await this.readRange(axis, axis.dataOffset + rowStart * rowBytes, rowCount * rowBytes, signal);
    const count = rowCount * width;
    if (axis.littleEndian === LITTLE_ENDIAN_PLATFORM && bytes.byteOffset % 4 === 0)
      return new Float32Array(bytes.buffer, bytes.byteOffset, count);
    const values = new Float32Array(count);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (let index = 0; index < count; index++) values[index] = view.getFloat32(index * 4, axis.littleEndian);
    return values;
  }

  readRange(axis, offset, length, signal) {
    return readTifxyzRange(axis.url, offset, length, signal, {
      queue: this,
      priority: SURFACE_GEOMETRY_PRIORITY,
      version: axis,
      byteCache: this.byteCache,
    });
  }
}
