import { Runs } from "./history.js";
import { pooled, poolRatios } from "./stamp.js";

const CELL = 16;
const POOL = 256;

function mark({ mask, marked }, index, value) {
  if (mask[index] === 0) marked.push(index);
  if (mask[index] < value) mask[index] = value;
}

function boxSides(ratios, chunks_ZYX) {
  const sides = [[CELL, CELL, CELL]];
  for (let level = 1; level < ratios.length; level++)
    sides.push(
      sides[level - 1].map((side, axis) => {
        const ratio = ratios[level][axis];
        if (level > 1) return Math.floor((side + ratio - 2) / ratio) + 1;
        let most = 0;
        for (let low = 0; low < chunks_ZYX[axis]; low += CELL)
          most = Math.max(most, Math.ceil(Math.min(chunks_ZYX[axis], low + CELL) / ratio) - Math.floor(low / ratio));
        return most;
      }),
    );
  return sides;
}

export class Pyramid {
  constructor(volume) {
    this.volume = volume;
    this.ratios = poolRatios(volume.scales);
    const { shape_ZYX, chunks_ZYX } = volume.scales[0];
    this.chunkCells_ZYX = chunks_ZYX.map((size) => Math.ceil(size / CELL));
    this.cells_ZYX = shape_ZYX.map((size, axis) => Math.ceil(size / chunks_ZYX[axis]) * this.chunkCells_ZYX[axis]);
    this.sides_ZYX = boxSides(this.ratios, chunks_ZYX);
    this.sizes = this.sides_ZYX.map(([sideZ, sideY, sideX]) => sideZ * sideY * sideX);
    this.places_N3 = this.sides_ZYX.map(([sideZ, sideY, sideX]) => {
      const place_N3 = new Uint8Array(sideZ * sideY * sideX * 3);
      for (let z = 0; z < sideZ; z++)
        for (let y = 0; y < sideY; y++)
          for (let x = 0; x < sideX; x++) place_N3.set([z, y, x], ((z * sideY + y) * sideX + x) * 3);
      return place_N3;
    });
    const ratio_ZYX = this.ratios[1] ?? [1, 1, 1];
    this.within_ZYX = chunks_ZYX.map((size, axis) =>
      Int32Array.from(
        { length: size },
        (_, voxel) => Math.floor(voxel / ratio_ZYX[axis]) - Math.floor((voxel - (voxel % CELL)) / ratio_ZYX[axis]),
      ),
    );
    this.marks = this.sizes.map((size, level) => (level > 1 ? { mask: new Uint16Array(size), marked: [] } : null));
    this.zeros = new Uint8Array(chunks_ZYX[0] * chunks_ZYX[1] * chunks_ZYX[2]);
    this.pending = new Map();
    this.mixed = new Set();
    this.writing = new Runs(2, 1);
  }

  cellBox(cell) {
    const { shape_ZYX, chunks_ZYX } = this.volume.scales[0];
    const [, cellsY, cellsX] = this.cells_ZYX;
    const index_ZYX = [Math.floor(cell / cellsX / cellsY), Math.floor(cell / cellsX) % cellsY, cell % cellsX];
    const low_ZYX = index_ZYX.map(
      (index, axis) =>
        Math.floor(index / this.chunkCells_ZYX[axis]) * chunks_ZYX[axis] + (index % this.chunkCells_ZYX[axis]) * CELL,
    );
    const high_ZYX = low_ZYX.map((low, axis) =>
      Math.min(shape_ZYX[axis], low + CELL, (Math.floor(low / chunks_ZYX[axis]) + 1) * chunks_ZYX[axis]),
    );
    return { low_ZYX, high_ZYX };
  }

  followRuns(chunk_ZYX, axis, offsets_N, lengths_N, raised, mixed) {
    const chunks_ZYX = this.volume.scales[0].chunks_ZYX;
    const [, chunkY, chunkX] = chunks_ZYX;
    const [, cellsY, cellsX] = this.cells_ZYX;
    const plane = chunkY * chunkX;
    const stride = [plane, chunkX, 1][axis];
    const [firstZ, firstY, firstX] = chunk_ZYX.map((chunk, axis) => chunk * this.chunkCells_ZYX[axis]);
    const coarser = this.volume.scales.length > 1;
    const [withinZ, withinY, withinX] = this.within_ZYX;
    const within = this.within_ZYX[axis];
    const [, sideY, sideX] = this.sides_ZYX[coarser ? 1 : 0];
    const step = [sideY * sideX, sideX, 1][axis];
    const value = raised ?? POOL;
    let last = -1;
    let marks = null;
    for (let run = 0; run < offsets_N.length; run++) {
      for (let offset = offsets_N[run], left = lengths_N[run]; left > 0; ) {
        const z = Math.floor(offset / plane);
        const y = Math.floor((offset - z * plane) / chunkX);
        const x = offset - z * plane - y * chunkX;
        const cell =
          ((firstZ + Math.floor(z / CELL)) * cellsY + firstY + Math.floor(y / CELL)) * cellsX +
          firstX +
          Math.floor(x / CELL);
        if (cell !== last) {
          last = cell;
          marks = this.pending.get(cell);
          if (marks === undefined) {
            marks = { mask: new Uint16Array(coarser ? this.sizes[1] : 0), marked: [] };
            this.pending.set(cell, marks);
          }
          if (mixed) this.mixed.add(cell);
        }
        const at = axis === 0 ? z : axis === 1 ? y : x;
        const along = Math.min(left, CELL - (at % CELL), chunks_ZYX[axis] - at);
        if (coarser && !mixed) {
          const index = (withinZ[z] * sideY + withinY[y]) * sideX + withinX[x];
          for (let each = 0; each <= within[at + along - 1] - within[at]; each++)
            mark(marks, index + each * step, value);
        }
        offset += along * stride;
        left -= along;
      }
    }
  }

  async followed(pause, drawn) {
    for (const [cell, marks] of this.pending) {
      this.pending.delete(cell);
      if (this.mixed.delete(cell)) this.pooledAll(cell);
      else this.followedMarks(cell, marks);
      await pause(drawn);
    }
    drawn();
  }

  followedMarks(cell, marks) {
    const { chunks_ZYX } = this.volume.scales[0];
    const [chunkZ, chunkY, chunkX] = chunks_ZYX;
    let { low_ZYX, high_ZYX } = this.cellBox(cell);
    for (let level = 1; level < this.volume.scales.length && marks.marked.length > 0; level++) {
      const { mask, marked } = marks;
      const ratio_ZYX = this.ratios[level];
      const [ratioZ, ratioY, ratioX] = ratio_ZYX;
      low_ZYX = low_ZYX.map((low, axis) => Math.floor(low / ratio_ZYX[axis]));
      high_ZYX = high_ZYX.map((high, axis) =>
        Math.min(this.volume.scales[level].shape_ZYX[axis], Math.ceil(high / ratio_ZYX[axis])),
      );
      const [lowZ, lowY, lowX] = low_ZYX;
      const [highZ, highY, highX] = high_ZYX;
      const [shapeZ, shapeY, shapeX] = this.volume.scales[level - 1].shape_ZYX;
      const place_N3 = this.places_N3[level];
      const coarse_ZYX = low_ZYX.map((low, axis) => Math.floor(low / chunks_ZYX[axis]));
      const [coarseZ, coarseY, coarseX] = coarse_ZYX.map((chunk, axis) => chunk * chunks_ZYX[axis]);
      const [fineZ, fineY, fineX] = low_ZYX.map(
        (low, axis) => Math.floor((low * ratio_ZYX[axis]) / chunks_ZYX[axis]) * chunks_ZYX[axis],
      );
      const fine_DHW = this.volume.chunk(level - 1, [fineZ / chunkZ, fineY / chunkY, fineX / chunkX]) ?? this.zeros;
      const coarse_DHW = this.volume.chunk(level, coarse_ZYX) ?? this.zeros;
      const next = this.marks[level + 1] ?? null;
      if (next !== null) next.marked.length = 0;
      const [nextZ, nextY, nextX] = next !== null ? this.ratios[level + 1] : [1, 1, 1];
      const [, nextSideY, nextSideX] = next !== null ? this.sides_ZYX[level + 1] : [1, 1, 1];
      const [originZ, originY, originX] = [
        Math.floor(lowZ / nextZ),
        Math.floor(lowY / nextY),
        Math.floor(lowX / nextX),
      ];
      for (const index of marked) {
        const value = mask[index];
        mask[index] = 0;
        const z = lowZ + place_N3[index * 3];
        const y = lowY + place_N3[index * 3 + 1];
        const x = lowX + place_N3[index * 3 + 2];
        if (z >= highZ || y >= highY || x >= highX) continue;
        const at = ((z - coarseZ) * chunkY + y - coarseY) * chunkX + x - coarseX;
        const was = coarse_DHW[at];
        let label = value === POOL ? 0 : Math.max(was, value);
        if (value === POOL) {
          const lastZ = Math.min(shapeZ, (z + 1) * ratioZ);
          const lastY = Math.min(shapeY, (y + 1) * ratioY);
          const lastX = Math.min(shapeX, (x + 1) * ratioX);
          for (let fz = z * ratioZ; fz < lastZ; fz++) {
            for (let fy = y * ratioY; fy < lastY; fy++) {
              const row = ((fz - fineZ) * chunkY + fy - fineY) * chunkX - fineX;
              for (let fx = x * ratioX; fx < lastX; fx++) if (fine_DHW[row + fx] > label) label = fine_DHW[row + fx];
            }
          }
        }
        if (label === was) continue;
        this.writing.add(at, 1, label);
        if (next !== null) {
          const above =
            ((Math.floor(z / nextZ) - originZ) * nextSideY + Math.floor(y / nextY) - originY) * nextSideX +
            Math.floor(x / nextX) -
            originX;
          mark(next, above, label > was ? label : POOL);
        }
      }
      const { offsets_N, lengths_N, values_N, count } = this.writing;
      if (count === 0) break;
      this.volume.writeRuns(
        level,
        coarse_ZYX,
        2,
        offsets_N.subarray(0, count),
        lengths_N.subarray(0, count),
        values_N.subarray(0, count),
      );
      this.writing.clear();
      marks = next;
    }
  }

  pooledAll(cell) {
    const { chunks_ZYX } = this.volume.scales[0];
    let { low_ZYX, high_ZYX } = this.cellBox(cell);
    for (let level = 1; level < this.volume.scales.length; level++) {
      const fine = this.volume.scales[level - 1];
      const coarse = this.volume.scales[level];
      const ratio_ZYX = this.ratios[level];
      low_ZYX = low_ZYX.map((low, axis) => Math.floor(low / ratio_ZYX[axis]));
      high_ZYX = high_ZYX.map((high, axis) => Math.min(coarse.shape_ZYX[axis], Math.ceil(high / ratio_ZYX[axis])));
      const from_ZYX = low_ZYX.map((low, axis) => low * ratio_ZYX[axis]);
      const size_ZYX = high_ZYX.map(
        (high, axis) => Math.min(fine.shape_ZYX[axis], high * ratio_ZYX[axis]) - from_ZYX[axis],
      );
      if (size_ZYX.some((size) => size < 1)) break;
      const fine_DHW = this.volume.chunk(
        level - 1,
        from_ZYX.map((from, axis) => Math.floor(from / chunks_ZYX[axis])),
      );
      const { shape_ZYX, out_DHW } = pooled(
        fine_DHW ?? this.zeros,
        chunks_ZYX,
        from_ZYX.map((from, axis) => from % chunks_ZYX[axis]),
        size_ZYX,
        ratio_ZYX,
      );
      const coarse_DHW = this.volume.chunk(
        level,
        low_ZYX.map((low, axis) => Math.floor(low / chunks_ZYX[axis])),
      );
      const [inZ, inY, inX] = low_ZYX.map((low, axis) => low % chunks_ZYX[axis]);
      let same = coarse_DHW !== null || out_DHW.every((label) => label === 0);
      for (let z = 0; coarse_DHW !== null && same && z < shape_ZYX[0]; z++)
        for (let y = 0; same && y < shape_ZYX[1]; y++) {
          const out = (z * shape_ZYX[1] + y) * shape_ZYX[2];
          const was = ((z + inZ) * chunks_ZYX[1] + y + inY) * chunks_ZYX[2] + inX;
          for (let x = 0; same && x < shape_ZYX[2]; x++) same = out_DHW[out + x] === coarse_DHW[was + x];
        }
      if (!same) this.volume.write(level, low_ZYX, shape_ZYX, out_DHW);
    }
  }

  forget() {
    this.pending.clear();
    this.mixed.clear();
  }
}
