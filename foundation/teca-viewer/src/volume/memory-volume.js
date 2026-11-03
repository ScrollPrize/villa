import { numericTypeForDtype } from "./numeric-type.js";

export function uint8Scales(name, levels) {
  const numericType = numericTypeForDtype("|u1", name);
  return levels.map(({ shape_ZYX, chunks_ZYX, downsample_ZYX, offset_ZYX }, index) =>
    Object.freeze({
      path: String(index),
      axes: ["z", "y", "x"],
      storageChunks: chunks_ZYX,
      shape_ZYX,
      chunks_ZYX,
      downsample_ZYX,
      offset_ZYX,
      numericType,
      fillValue: 0,
    }),
  );
}

export class MemoryVolume {
  constructor(name, levels, zeroed) {
    this.rootUrl = `memory:${name}/`;
    this.scales = uint8Scales(name, levels);
    this.metadata = { displayEncoding: { kind: "identity_uint8" } };
    this.navigationGeneration = 0;
    this.zeroed = zeroed;
    this.grids = levels.map(({ shape_ZYX, chunks_ZYX }) =>
      shape_ZYX.map((size, axis) => Math.ceil(size / chunks_ZYX[axis])),
    );
    this.chunks = levels.map(() => new Map());
    this.zeros = levels.map(({ chunks_ZYX }) => new Uint8Array(chunks_ZYX[0] * chunks_ZYX[1] * chunks_ZYX[2]));
    this.written = levels.map(() => new Map());
  }

  beginNavigation(generation) {
    this.navigationGeneration = generation;
  }

  retainChunkRequests() {}

  cancelPendingChunks() {}

  async getChunk(chunkZ, chunkY, chunkX, _channel, scale, signal) {
    signal.throwIfAborted();
    const level = this.scales.indexOf(scale);
    return this.chunk(level, [chunkZ, chunkY, chunkX]) ?? this.zeros[level];
  }

  isFill(chunk_DHW) {
    return this.zeros.includes(chunk_DHW);
  }

  dispose() {}

  chunk(level, chunk_ZYX) {
    return this.chunks[level].get(this.index(level, ...chunk_ZYX)) ?? null;
  }

  index(level, cz, cy, cx) {
    return (cz * this.grids[level][1] + cy) * this.grids[level][2] + cx;
  }

  write(level, start_ZYX, shape_ZYX, data_DHW) {
    const { shape_ZYX: level_ZYX, chunks_ZYX } = this.scales[level];
    const inside = [0, 1, 2].every(
      (axis) =>
        Number.isInteger(start_ZYX[axis]) &&
        Number.isInteger(shape_ZYX[axis]) &&
        start_ZYX[axis] >= 0 &&
        shape_ZYX[axis] >= 1 &&
        start_ZYX[axis] + shape_ZYX[axis] <= level_ZYX[axis],
    );
    if (!inside)
      throw new Error(
        `A write of ${shape_ZYX.join("×")} at ${start_ZYX} is not inside level ${level} (${level_ZYX.join("×")}).`,
      );
    const count = shape_ZYX[0] * shape_ZYX[1] * shape_ZYX[2];
    if (data_DHW.length !== count) throw new Error(`A write of ${shape_ZYX.join("×")} voxels got ${data_DHW.length}.`);
    const [chunkZ, chunkY, chunkX] = chunks_ZYX;
    const plane = chunkY * chunkX;
    const first_ZYX = start_ZYX.map((start, axis) => Math.floor(start / chunks_ZYX[axis]));
    const last_ZYX = start_ZYX.map((start, axis) => Math.floor((start + shape_ZYX[axis] - 1) / chunks_ZYX[axis]));
    for (let cz = first_ZYX[0]; cz <= last_ZYX[0]; cz++) {
      for (let cy = first_ZYX[1]; cy <= last_ZYX[1]; cy++) {
        for (let cx = first_ZYX[2]; cx <= last_ZYX[2]; cx++) {
          const chunk_DHW = this.chunks[level].get(this.index(level, cz, cy, cx)) ?? this.zeroed(chunkZ * plane);
          // Plain numbers and not mapped arrays: a stroke is thousands of writes of a few hundred voxels each.
          const z0 = Math.max(start_ZYX[0], cz * chunkZ);
          const z1 = Math.min(start_ZYX[0] + shape_ZYX[0], (cz + 1) * chunkZ);
          const y0 = Math.max(start_ZYX[1], cy * chunkY);
          const y1 = Math.min(start_ZYX[1] + shape_ZYX[1], (cy + 1) * chunkY);
          const x0 = Math.max(start_ZYX[2], cx * chunkX);
          const x1 = Math.min(start_ZYX[2] + shape_ZYX[2], (cx + 1) * chunkX);
          for (let z = z0; z < z1; z++) {
            for (let y = y0; y < y1; y++) {
              const from = ((z - start_ZYX[0]) * shape_ZYX[1] + y - start_ZYX[1]) * shape_ZYX[2] + x0 - start_ZYX[2];
              const to = (z - cz * chunkZ) * plane + (y - cy * chunkY) * chunkX + x0 - cx * chunkX;
              chunk_DHW.set(data_DHW.subarray(from, from + x1 - x0), to);
            }
          }
          this.hold(level, [cz, cy, cx], chunk_DHW, (z0 - cz * chunkZ) * plane, (z1 - cz * chunkZ) * plane);
        }
      }
    }
  }

  // With a copy of each of the 3,740 chunks paint brings back from IndexedDB after a stroke of radius 256 across the
  // Surface, 0.9 GiB, V8's collector held the page 59 to 295 ms in each of 5 reloads to finish sweeping; with the
  // arrays themselves, 28 to 88 ms in 4 of 9.
  place(level, chunk_ZYX, data_DHW) {
    const { shape_ZYX, chunks_ZYX } = this.scales[level];
    const inside = chunk_ZYX.every(
      (chunk, axis) => Number.isInteger(chunk) && chunk >= 0 && (chunk + 1) * chunks_ZYX[axis] <= shape_ZYX[axis],
    );
    if (!inside)
      throw new Error(`Chunk ${chunk_ZYX} is not a whole chunk inside level ${level} (${shape_ZYX.join("×")}).`);
    const count = chunks_ZYX[0] * chunks_ZYX[1] * chunks_ZYX[2];
    if (data_DHW.length !== count) throw new Error(`A chunk of ${chunks_ZYX.join("×")} voxels got ${data_DHW.length}.`);
    this.hold(level, chunk_ZYX, data_DHW, 0, count);
  }

  writeRuns(level, chunk_ZYX, axis, offsets_N, lengths_N, values_N) {
    const { shape_ZYX, chunks_ZYX } = this.scales[level];
    const count = chunks_ZYX[0] * chunks_ZYX[1] * chunks_ZYX[2];
    const inside = chunk_ZYX.every(
      (chunk, axis) => Number.isInteger(chunk) && chunk >= 0 && chunk * chunks_ZYX[axis] < shape_ZYX[axis],
    );
    if (!inside) throw new Error(`Chunk ${chunk_ZYX} is not inside level ${level} (${shape_ZYX.join("×")}).`);
    const stride = [chunks_ZYX[1] * chunks_ZYX[2], chunks_ZYX[2], 1][axis];
    if (stride === undefined) throw new Error(`Runs along axis ${axis}, not 0, 1 or 2.`);
    if (lengths_N.length !== offsets_N.length || values_N.length !== offsets_N.length)
      throw new Error(
        `Runs of ${offsets_N.length} offsets got ${lengths_N.length} lengths and ${values_N.length} values.`,
      );
    if (offsets_N.length === 0) return;
    const chunk_DHW = this.chunk(level, chunk_ZYX) ?? this.zeroed(count);
    let low = count;
    let high = 0;
    for (let run = 0; run < offsets_N.length; run++) {
      const offset = offsets_N[run];
      const last = offset + (lengths_N[run] - 1) * stride;
      if (!(offset >= 0 && lengths_N[run] >= 1 && last < count))
        throw new Error(
          `A run of ${lengths_N[run]} voxels along axis ${axis} at ${offset} is not inside a chunk of ${count}.`,
        );
      if (last === offset) chunk_DHW[offset] = values_N[run];
      else if (stride === 1) chunk_DHW.fill(values_N[run], offset, last + 1);
      else for (let voxel = offset; voxel <= last; voxel += stride) chunk_DHW[voxel] = values_N[run];
      low = Math.min(low, offset);
      high = Math.max(high, last + 1);
    }
    const plane = chunks_ZYX[1] * chunks_ZYX[2];
    this.hold(level, chunk_ZYX, chunk_DHW, Math.floor(low / plane) * plane, Math.ceil(high / plane) * plane);
  }

  hold(level, chunk_ZYX, chunk_DHW, start, end) {
    const index = this.index(level, ...chunk_ZYX);
    this.chunks[level].set(index, chunk_DHW);
    const change = this.written[level].get(index);
    if (change === undefined)
      this.written[level].set(index, { scale: this.scales[level], chunk_ZYX, data_DHW: chunk_DHW, start, end });
    else {
      change.data_DHW = chunk_DHW;
      change.start = Math.min(change.start, start);
      change.end = Math.max(change.end, end);
    }
  }

  take() {
    const changes = this.written.flatMap((changed) => [...changed.values()]);
    this.written = this.scales.map(() => new Map());
    return changes;
  }
}
