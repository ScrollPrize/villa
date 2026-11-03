import { MAX_SIDE, TILE } from "./stamp.js";

export const SLAB = 0.5;
const PAUSE_ROWS = 4096;
// A voxel counts as the last segment's, or a prism as held whole by the sweep, only this far inside the brush in its
// metric, and a prism's line is cut this far outside it, so a rounding sets a voxel again rather than missing it; and
// a segment this near a line's direction runs along it.
export const MARGIN = 1e-9;
const GRID = MAX_SIDE / TILE;

function reachAlong(radius, normal, spread) {
  const height = Math.min(SLAB, radius * Math.abs(normal));
  const half = Math.sqrt(Math.max(0, radius ** 2 - (radius ** 2 * normal) ** 2 / spread));
  return (height * radius ** 2 * Math.abs(normal)) / spread + Math.sqrt(Math.max(0, 1 - height ** 2 / spread)) * half;
}

// The first and last whole voxel of a row where n·v - level runs as `height` + `normal` × the voxel, within [-SLAB,
// SLAB): half-open, so a plane half a voxel off the grid takes one layer of it and not two.
export function slabFirst(height, normal) {
  if (normal === 0) return height >= -SLAB && height < SLAB ? -Infinity : Infinity;
  return normal > 0 ? Math.ceil((-SLAB - height) / normal) : Math.floor((SLAB - height) / normal) + 1;
}

export function slabLast(height, normal) {
  if (normal === 0) return height >= -SLAB && height < SLAB ? Infinity : -Infinity;
  return normal > 0 ? Math.ceil((SLAB - height) / normal) - 1 : Math.floor((-SLAB - height) / normal);
}

export function sweptSpan(span, w0w0, w0e, ee, endEnd, endE, w0D, eD, DD, reach2) {
  let low = Infinity;
  let high = -Infinity;
  const start = w0e ** 2 - ee * (w0w0 - reach2);
  if (start >= 0) {
    low = (-w0e - Math.sqrt(start)) / ee;
    high = (-w0e + Math.sqrt(start)) / ee;
  }
  const end = endE ** 2 - ee * (endEnd - reach2);
  if (end >= 0) {
    low = Math.min(low, (-endE - Math.sqrt(end)) / ee);
    high = Math.max(high, (-endE + Math.sqrt(end)) / ee);
  }
  const alpha = DD > 0 ? ee - eD ** 2 / DD : 0;
  if (alpha > MARGIN * ee) {
    const beta = w0e - (w0D * eD) / DD;
    const gamma = w0w0 - w0D ** 2 / DD - reach2;
    const discriminant = beta ** 2 - alpha * gamma;
    if (discriminant >= 0) {
      const sum = -(beta + (beta < 0 ? -1 : 1) * Math.sqrt(discriminant));
      const near = sum / alpha;
      const far = sum === 0 ? 0 : gamma / sum;
      const tNear = (w0D + near * eD) / DD;
      const tFar = (w0D + far * eD) / DD;
      if (tNear >= 0 && tNear <= 1) {
        low = Math.min(low, near);
        high = Math.max(high, near);
      }
      if (tFar >= 0 && tFar <= 1) {
        low = Math.min(low, far);
        high = Math.max(high, far);
      }
    }
  }
  span[0] = low;
  span[1] = high;
}

export class Runs {
  constructor() {
    this.tiles = new Map();
    this.key = -1;
    this.runs = null;
    this.low_ZYX = [Infinity, Infinity, Infinity];
    this.high_ZYX = [-Infinity, -Infinity, -Infinity];
  }

  add(voxel_ZYX, row, first, last) {
    for (let axis = 0; axis < 3; axis++) {
      const low = axis === row ? first : voxel_ZYX[axis];
      const high = axis === row ? last : voxel_ZYX[axis];
      this.low_ZYX[axis] = Math.min(this.low_ZYX[axis], low);
      this.high_ZYX[axis] = Math.max(this.high_ZYX[axis], high + 1);
    }
    for (let voxel = first; voxel <= last; ) {
      voxel_ZYX[row] = voxel;
      const z = voxel_ZYX[0];
      const y = voxel_ZYX[1];
      const x = voxel_ZYX[2];
      const next = ((Math.floor(z / TILE) * GRID + Math.floor(y / TILE)) * GRID + Math.floor(x / TILE)) * 3 + row;
      if (this.key !== next) {
        this.key = next;
        this.runs = this.tiles.get(next);
        if (this.runs === undefined) {
          this.runs = { from_ZYX: [z - (z % TILE), y - (y % TILE), x - (x % TILE)], axis: row, runs: [] };
          this.tiles.set(next, this.runs);
        }
      }
      const length = Math.min(last + 1, (Math.floor(voxel / TILE) + 1) * TILE) - voxel;
      this.runs.runs.push(z % TILE, y % TILE, x % TILE, length);
      voxel += length;
    }
  }

  packed() {
    const runs_N4 = new Uint8Array([...this.tiles.values()].reduce((total, { runs }) => total + runs.length, 0));
    const tiles_N5 = new Int32Array(this.tiles.size * 5);
    let end = 0;
    for (const [index, { from_ZYX, axis, runs }] of [...this.tiles.values()].entries()) {
      runs_N4.set(runs, end);
      end += runs.length;
      tiles_N5.set([...from_ZYX, axis, end], index * 5);
    }
    return { runs_N4, tiles_N5, low_ZYX: this.low_ZYX, high_ZYX: this.high_ZYX };
  }
}

export function tilesOf({ runs_N4, tiles_N5, low_ZYX, high_ZYX }) {
  const tiles = [];
  for (let at = 0, start = 0; at < tiles_N5.length; at += 5) {
    const end = tiles_N5[at + 4];
    tiles.push({
      from_ZYX: [tiles_N5[at], tiles_N5[at + 1], tiles_N5[at + 2]],
      shape_ZYX: [TILE, TILE, TILE],
      axis: tiles_N5[at + 3],
      runs_N4: runs_N4.subarray(start, end),
    });
    start = end;
  }
  return { tiles, low_ZYX, high_ZYX };
}

export async function sweptTiles(radii_ZYX, plane, from_ZYX, to_ZYX, before, shape_ZYX, pause) {
  const { normal_ZYX, level } = plane;
  const off = normal_ZYX.reduce((total, normal, axis) => total + normal * from_ZYX[axis], 0) - level;
  const start_ZYX = from_ZYX.map((from, axis) => from - off * normal_ZYX[axis]);
  const spread = radii_ZYX.reduce((total, radius, axis) => total + (radius * normal_ZYX[axis]) ** 2, 0);
  const low_ZYX = radii_ZYX.map((radius, axis) =>
    Math.max(0, Math.floor(Math.min(start_ZYX[axis], to_ZYX[axis]) - reachAlong(radius, normal_ZYX[axis], spread))),
  );
  const high_ZYX = radii_ZYX.map((radius, axis) =>
    Math.min(
      shape_ZYX[axis] - 1,
      Math.ceil(Math.max(start_ZYX[axis], to_ZYX[axis]) + reachAlong(radius, normal_ZYX[axis], spread)),
    ),
  );
  const lean_ZYX = normal_ZYX.map(Math.abs);
  const along = [0, 1, 2].reduce((best, axis) => (lean_ZYX[axis] > lean_ZYX[best] ? axis : best));
  const row = along !== 2 ? 2 : lean_ZYX[1] <= lean_ZYX[0] ? 1 : 0;
  const side = 3 - along - row;
  const [qSide, qRow, qAlong] = [side, row, along].map((axis) => 1 / radii_ZYX[axis] ** 2);
  const [dSide, dRow, dAlong] = [side, row, along].map((axis) => to_ZYX[axis] - start_ZYX[axis]);
  const [nSide, nRow, nAlong] = [side, row, along].map((axis) => normal_ZYX[axis]);
  const dd = dSide ** 2 * qSide + dRow ** 2 * qRow + dAlong ** 2 * qAlong;
  const blur = nRow === 0 ? Infinity : (SLAB / Math.abs(nRow)) * Math.sqrt(qRow);
  const bounded = blur < 1;
  const a = qSide + (nSide ** 2 * qRow) / nRow ** 2;
  const b = (nSide * nAlong * qRow) / nRow ** 2;
  const c = qAlong + (nAlong ** 2 * qRow) / nRow ** 2;
  const passed = bounded && before === plane;
  const [startSide, startRow, startAlong] = [side, row, along].map((axis) => start_ZYX[axis]);
  const [fromSide, fromRow, fromAlong] = [side, row, along].map((axis) => from_ZYX[axis]);
  const [heldSide, heldRow, heldAlong] = [side, row, along].map((axis) => before?.normal_ZYX[axis]);
  const [lowRow, highRow] = [low_ZYX[row], high_ZYX[row]];
  const span = new Float64Array(2);
  const runs = new Runs();
  const voxel_ZYX = [0, 0, 0];
  const add = (at, across, first, last) => {
    voxel_ZYX[side] = at;
    voxel_ZYX[along] = across;
    runs.add(voxel_ZYX, row, first, last);
  };
  const rowsAt = (at) => {
    const lowest = Math.min(-SLAB + level - nSide * at - nRow * lowRow, -SLAB + level - nSide * at - nRow * highRow);
    const highest = Math.max(SLAB + level - nSide * at - nRow * lowRow, SLAB + level - nSide * at - nRow * highRow);
    let first = Math.max(low_ZYX[along], Math.floor(Math.min(lowest / nAlong, highest / nAlong)));
    let last = Math.min(high_ZYX[along], Math.ceil(Math.max(lowest / nAlong, highest / nAlong)));
    let heldFirst = Infinity;
    let heldLast = -Infinity;
    if (bounded) {
      const w = at - startSide;
      const e = w - dSide;
      const endEnd = a * e ** 2 - 2 * b * e * dAlong + c * dAlong ** 2;
      sweptSpan(
        span,
        a * w ** 2,
        b * w,
        c,
        endEnd,
        b * e - c * dAlong,
        a * w * dSide + b * w * dAlong,
        b * dSide + c * dAlong,
        dd,
        (1 + blur) ** 2,
      );
      first = Math.max(first, Math.ceil(startAlong + span[0]));
      last = Math.min(last, Math.floor(startAlong + span[1]));
      if (passed) {
        sweptSpan(span, a * w ** 2, b * w, c, a * w ** 2, b * w, 0, 0, 0, (Math.sqrt(1 - MARGIN) - blur) ** 2);
        heldFirst = Math.ceil(startAlong + span[0]);
        heldLast = Math.floor(startAlong + span[1]);
      }
    }
    const wSide = at - startSide;
    const fromSideSquare = (at - fromSide) ** 2 * qSide;
    for (let across = first; across <= last; across++) {
      if (across === heldFirst && heldFirst <= heldLast) {
        across = heldLast;
        continue;
      }
      const height = nSide * at + nAlong * across - level;
      let begin = Math.max(slabFirst(height, nRow), lowRow);
      let final = Math.min(slabLast(height, nRow), highRow);
      if (begin > final) continue;
      const wAlong = across - startAlong;
      const w0w0 = wSide ** 2 * qSide + wAlong ** 2 * qAlong;
      const endEnd = (wSide - dSide) ** 2 * qSide + (wAlong - dAlong) ** 2 * qAlong + dRow ** 2 * qRow;
      sweptSpan(
        span,
        w0w0,
        0,
        qRow,
        endEnd,
        -dRow * qRow,
        wSide * dSide * qSide + wAlong * dAlong * qAlong,
        dRow * qRow,
        dd,
        1,
      );
      begin = Math.max(begin, Math.ceil(startRow + span[0]));
      final = Math.min(final, Math.floor(startRow + span[1]));
      if (begin > final) continue;
      if (before !== null) {
        const inside = fromSideSquare + (across - fromAlong) ** 2 * qAlong;
        sweptSpan(span, inside, 0, qRow, inside, 0, 0, 0, 0, 1 - MARGIN);
        const held = heldSide * at + heldAlong * across - before.level;
        const skip = Math.max(begin, slabFirst(held, heldRow), Math.ceil(fromRow + span[0]));
        const stop = Math.min(final, slabLast(held, heldRow), Math.floor(fromRow + span[1]));
        if (skip <= stop) {
          if (begin < skip) add(at, across, begin, skip - 1);
          if (stop < final) add(at, across, stop + 1, final);
          continue;
        }
      }
      add(at, across, begin, final);
    }
    return Math.max(0, last - first + 1);
  };
  let rows = 0;
  for (let at = low_ZYX[side]; at <= high_ZYX[side]; at++) {
    rows += rowsAt(at);
    if (rows < PAUSE_ROWS) continue;
    rows = 0;
    await pause();
  }
  return tilesOf(runs.packed());
}
