// A point is valid where all three coordinates are finite and not the (-1, -1, -1) that tifxyz marks none with.

import {
  SURFACE_READ_BYTES,
  SURFACE_READS_IN_FLIGHT,
  SURFACE_STRIP_BYTE_LIMIT,
  SURFACE_STRIP_ROWS,
} from "../config.js";
import { MemoryTifxyz } from "./memory-tifxyz.js";
import { Tifxyz } from "./tifxyz.js";

export const STRIP_TEXEL_BYTES = 4 * Float32Array.BYTES_PER_ELEMENT;

function interleaveStrip(xyz_HW, start, count, pixels) {
  const strip_HWRGBA = new Float32Array(pixels * 4);
  for (let cell = 0; cell < count; cell++) {
    const x = xyz_HW[0][start + cell],
      y = xyz_HW[1][start + cell],
      z = xyz_HW[2][start + cell];
    if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z) && (x !== -1 || y !== -1 || z !== -1)) {
      const offset = cell * 4;
      strip_HWRGBA[offset] = z;
      strip_HWRGBA[offset + 1] = y;
      strip_HWRGBA[offset + 2] = x;
      strip_HWRGBA[offset + 3] = 1;
    }
  }
  return strip_HWRGBA;
}

function validateMetadata(metadata) {
  if (
    metadata === null ||
    typeof metadata !== "object" ||
    Array.isArray(metadata) ||
    Object.keys(metadata).sort().join("\0") !== "coordinate_order\0encoding\0grid_hw\0scale_uv\0strip_count\0strip_rows"
  ) {
    throw new Error("Surface metadata has unknown or missing fields.");
  }
  if (metadata.coordinate_order !== "ZYX" || metadata.encoding !== "float32-zyx-valid") {
    throw new Error("Surface metadata must use float32 ZYX coordinates with validity.");
  }
  if (
    !Array.isArray(metadata.grid_hw) ||
    metadata.grid_hw.length !== 2 ||
    metadata.grid_hw.some((value) => !Number.isSafeInteger(value) || value < 2)
  ) {
    throw new Error("Surface grid shape must contain two integers of at least two.");
  }
  if (
    !Array.isArray(metadata.scale_uv) ||
    metadata.scale_uv.length !== 2 ||
    metadata.scale_uv.some((value) => !Number.isFinite(value) || value <= 0)
  ) {
    throw new Error("Surface UV scale must contain two positive finite values.");
  }
  if (
    !Number.isSafeInteger(metadata.strip_rows) ||
    metadata.strip_rows < 2 ||
    !Number.isSafeInteger(metadata.strip_count) ||
    metadata.strip_count !== Math.ceil(metadata.grid_hw[0] / metadata.strip_rows)
  ) {
    throw new Error("Surface strip layout is inconsistent with its grid.");
  }
}

export class TifxyzSurfaceSource {
  constructor(path, metadata, tifxyz, cache) {
    validateMetadata(metadata);
    this.path = path;
    this.grid_HW = Object.freeze([...metadata.grid_hw]);
    this.scale_UV = Object.freeze([...metadata.scale_uv]);
    this.stripRows = metadata.strip_rows;
    this.stripCount = metadata.strip_count;
    this.tifxyz = tifxyz;
    this.cache = cache;
    this.pendingStrips = new Map();
  }

  static async open(path, edited, signal, stripCache, byteCache) {
    const tifxyz = edited === null ? await Tifxyz.open(path, signal, byteCache) : new MemoryTifxyz(edited);
    try {
      signal.throwIfAborted();
      const stripBytes = SURFACE_STRIP_ROWS * tifxyz.grid_HW[1] * STRIP_TEXEL_BYTES;
      if (stripBytes > SURFACE_STRIP_BYTE_LIMIT) {
        throw new Error(
          `${path} surface strips need ${stripBytes} bytes, exceeding the ${SURFACE_STRIP_BYTE_LIMIT}-byte strip limit.`,
        );
      }
      const metadata = {
        coordinate_order: "ZYX",
        grid_hw: [...tifxyz.grid_HW],
        scale_uv: [...tifxyz.scale_UV],
        strip_rows: SURFACE_STRIP_ROWS,
        strip_count: Math.ceil(tifxyz.grid_HW[0] / SURFACE_STRIP_ROWS),
        encoding: "float32-zyx-valid",
      };
      return new TifxyzSurfaceSource(path, metadata, tifxyz, stripCache);
    } catch (error) {
      tifxyz.dispose();
      throw error;
    }
  }

  release() {
    this.cache.release(this);
    this.tifxyz.release();
  }

  dispose() {
    this.cache.release(this);
    this.tifxyz.dispose();
  }

  async getStrip(stripIndex, signal = null) {
    if (!Number.isSafeInteger(stripIndex) || stripIndex < 0 || stripIndex >= this.stripCount) {
      throw new Error(`Surface strip must be inside 0..${this.stripCount - 1}.`);
    }
    signal?.throwIfAborted();
    if (this.tifxyz.cacheKey === null) return this.readStrip(stripIndex, signal);
    const cached = this.cache.get(this, stripIndex);
    if (cached !== null) return cached;
    const pending = this.pendingStrips.get(stripIndex);
    if (pending !== undefined && pending.signal === signal) return pending.promise;
    const entry = { signal, promise: null };
    entry.promise = this.readStrip(stripIndex, signal)
      .then((strip) => {
        signal?.throwIfAborted();
        return this.cache.admit(this, stripIndex, strip);
      })
      .finally(() => {
        if (this.pendingStrips.get(stripIndex) === entry) this.pendingStrips.delete(stripIndex);
      });
    this.pendingStrips.set(stripIndex, entry);
    return entry.promise;
  }

  async readStrip(stripIndex, signal) {
    const [height, width] = this.grid_HW;
    const rowStart = stripIndex * this.stripRows;
    const rowCount = Math.min(this.stripRows, height - rowStart);
    const xyz_HW = await Promise.all([0, 1, 2].map((axis) => this.tifxyz.readRows(axis, rowStart, rowCount, signal)));
    return interleaveStrip(xyz_HW, 0, rowCount * width, this.stripRows * width);
  }

  async forEachStrip(consume, signal) {
    const [height, width] = this.grid_HW;
    const stripsPerRead = Math.max(1, Math.ceil(SURFACE_READ_BYTES / (this.stripRows * width * 4)));
    const groups = Array.from({ length: Math.ceil(this.stripCount / stripsPerRead) }, (_, group) => ({
      start: group * stripsPerRead,
      axes: new Array(3),
      completed: 0,
    }));
    const controller = new AbortController();
    const abort = () => controller.abort(signal.reason);
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) abort();
    let next = 0;
    const read = async () => {
      while (next < groups.length * 3) {
        controller.signal.throwIfAborted();
        const job = next++;
        const group = groups[Math.floor(job / 3)];
        const rowStart = group.start * this.stripRows;
        const rows = Math.min(stripsPerRead * this.stripRows, height - rowStart);
        group.axes[job % 3] = await this.tifxyz.readRows(job % 3, rowStart, rows, controller.signal);
        controller.signal.throwIfAborted();
        if (++group.completed !== 3) continue;
        for (let offset = 0; offset < rows; offset += this.stripRows) {
          const index = group.start + offset / this.stripRows;
          const cells = Math.min(this.stripRows, rows - offset) * width;
          const strip = interleaveStrip(group.axes, offset * width, cells, this.stripRows * width);
          if (this.tifxyz.cacheKey !== null) this.cache.admit(this, index, strip);
          consume(
            index,
            strip,
            group.axes.map((values) => values.subarray(offset * width, offset * width + cells)),
          );
        }
        group.axes = null;
      }
    };
    try {
      await Promise.allSettled(
        Array.from({ length: Math.min(SURFACE_READS_IN_FLIGHT, groups.length * 3) }, () =>
          read().catch((error) => {
            controller.abort(error);
            throw error;
          }),
        ),
      );
      controller.signal.throwIfAborted();
    } finally {
      signal?.removeEventListener("abort", abort);
    }
  }
}
