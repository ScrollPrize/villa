import { TIFXYZ_TILE_CACHE_BYTES, TIFXYZ_TILE_READ_BYTES, TIFXYZ_TILES_IN_FLIGHT } from "../config.js";
import { PriorityRequestQueue } from "../volume/request-queue.js";
import { decodeFloat32Tile } from "./tiff-tile.js";

export class TifxyzTiles {
  constructor(readRange) {
    this.readRange = readRange;
    this.tiles = new Map();
    this.pending = new Map();
    this.bytes = 0;
    this.queue = new PriorityRequestQueue(TIFXYZ_TILES_IN_FLIGHT, TIFXYZ_TILES_IN_FLIGHT, TIFXYZ_TILE_READ_BYTES);
  }

  clear() {
    this.tiles.clear();
    this.bytes = 0;
  }

  dispose() {
    this.queue.dispose();
  }

  async readRows(axis, axisIndex, rowStart, rowCount, width, signal) {
    const result = new Float32Array(rowCount * width);
    const columns = Math.ceil(width / axis.tileWidth);
    for (let y = Math.floor(rowStart / axis.tileHeight); y < Math.ceil((rowStart + rowCount) / axis.tileHeight); y++) {
      for (let first = 0; first < columns; first += TIFXYZ_TILES_IN_FLIGHT) {
        const batch = await Promise.allSettled(
          Array.from({ length: Math.min(TIFXYZ_TILES_IN_FLIGHT, columns - first) }, async (_, i) => {
            const x = first + i;
            const tile = await this.readTile(axis, axisIndex, y * columns + x, signal);
            signal?.throwIfAborted();
            const begin = Math.max(rowStart, y * axis.tileHeight);
            const end = Math.min(rowStart + rowCount, (y + 1) * axis.tileHeight);
            const count = Math.min(axis.tileWidth, width - x * axis.tileWidth);
            for (let row = begin; row < end; row++) {
              const source = (row - y * axis.tileHeight) * axis.tileWidth;
              result.set(tile.subarray(source, source + count), (row - rowStart) * width + x * axis.tileWidth);
            }
          }),
        );
        for (const item of batch) if (item.status === "rejected") throw item.reason;
      }
    }
    return result;
  }

  async readTile(axis, axisIndex, index, signal) {
    signal?.throwIfAborted();
    const key = `${axisIndex}/${index}`;
    const cached = this.tiles.get(key);
    if (cached !== undefined) {
      this.tiles.delete(key);
      this.tiles.set(key, cached);
      return cached;
    }
    const pending = this.pending.get(key);
    if (pending !== undefined && pending.signal === signal) return pending.promise;
    const entry = { signal, promise: null };
    const workingBytes =
      axis.tileByteCounts[index] + axis.tileWidth * axis.tileHeight * 8 + axis.tileWidth * 4 + 16 * 1024;
    entry.promise = this.queue.enqueue(
      async () => {
        const bytes = await this.readRange(axis, axis.tileOffsets[index], axis.tileByteCounts[index], signal);
        const values = await decodeFloat32Tile(bytes, axis, signal);
        signal?.throwIfAborted();
        this.bytes -= this.tiles.get(key)?.byteLength ?? 0;
        this.tiles.delete(key);
        while (this.bytes + values.byteLength > TIFXYZ_TILE_CACHE_BYTES) {
          const oldest = this.tiles.keys().next().value;
          this.bytes -= this.tiles.get(oldest).byteLength;
          this.tiles.delete(oldest);
        }
        this.tiles.set(key, values);
        this.bytes += values.byteLength;
        return values;
      },
      0,
      signal,
      null,
      workingBytes,
    ).promise;
    this.pending.set(key, entry);
    try {
      return await entry.promise;
    } finally {
      if (this.pending.get(key) === entry) this.pending.delete(key);
    }
  }
}
