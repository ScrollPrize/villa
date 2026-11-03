import { SURFACE_OVERVIEW_POINT_LIMIT } from "../config.js";

const EMPTY_BOUNDS = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];

export class SurfacePyramid {
  constructor(grid_HW) {
    this.grid_HW = [...grid_HW];
    this.levels = [];
    // Native bounds include unsampled extrema and holes, so coarse geometry cannot hide a candidate.
    this.intersectionBlockCells = 32;
    while (grid_HW.map((size) => Math.ceil((size - 1) / this.intersectionBlockCells)).reduce((a, b) => a * b) > 65536)
      this.intersectionBlockCells *= 2;
    this.intersectionGrid_HW = grid_HW.map((size) => Math.ceil((size - 1) / this.intersectionBlockCells));
    this.intersectionBounds_B6 = new Float64Array(this.intersectionGrid_HW[0] * this.intersectionGrid_HW[1] * 6);
    for (let offset = 0; offset < this.intersectionBounds_B6.length; offset += 6)
      this.intersectionBounds_B6.set(EMPTY_BOUNDS, offset);
    this.revision = 0;
    let stride = 2;
    let pixels = 0;
    while (true) {
      const shape_HW = grid_HW.map((size) => Math.ceil((size - 1) / stride) + 1);
      this.levels.push({ stride, grid_HW: shape_HW, data_HWRGBA: null });
      pixels += shape_HW[0] * shape_HW[1];
      if (Math.max(...shape_HW) <= 2) break;
      stride *= 2;
    }
    const data_HWRGBA = new Float32Array(pixels * 4);
    let offset = 0;
    for (const level of this.levels) {
      const length = level.grid_HW[0] * level.grid_HW[1] * 4;
      level.data_HWRGBA = data_HWRGBA.subarray(offset, offset + length);
      offset += length;
    }
  }

  addStrip(source, index, strip_HWRGBA) {
    this.includeIntersectionBounds(source, index, strip_HWRGBA);
    const first = this.levels[0];
    const width = source.grid_HW[1];
    const start = index * source.stripRows;
    const stop = Math.min(source.grid_HW[0], start + source.stripRows);
    for (let v = start; v < stop; v++) {
      for (let u = 0; u < width; u++) {
        const offset = ((v - start) * width + u) * 4;
        const column = u === width - 1 ? first.grid_HW[1] - 1 : Math.floor(u / 2),
          row = v === source.grid_HW[0] - 1 ? first.grid_HW[0] - 1 : Math.floor(v / 2);
        if ((u % 2 === 0 || u === width - 1) && (v % 2 === 0 || v === source.grid_HW[0] - 1)) {
          const destination = (row * first.grid_HW[1] + column) * 4;
          first.data_HWRGBA[destination] = strip_HWRGBA[offset];
          first.data_HWRGBA[destination + 1] = strip_HWRGBA[offset + 1];
          first.data_HWRGBA[destination + 2] = strip_HWRGBA[offset + 2];
        }
        if (strip_HWRGBA[offset + 3] < 0.5) {
          for (let y = Math.max(0, Math.ceil(v / 2) - 1); y <= row; y++)
            for (let x = Math.max(0, Math.ceil(u / 2) - 1); x <= column; x++)
              first.data_HWRGBA[(y * first.grid_HW[1] + x) * 4 + 3] = -1;
        }
      }
    }
  }

  includeIntersectionBounds(source, index, strip_HWRGBA) {
    const size = this.intersectionBlockCells;
    const [rows, columns] = this.intersectionGrid_HW;
    const width = source.grid_HW[1];
    const start = index * source.stripRows;
    const stop = Math.min(source.grid_HW[0], start + source.stripRows);
    const bounds = this.intersectionBounds_B6;
    for (let v = start; v < stop; v++)
      for (let u = 0; u < width; u++) {
        const pixel = ((v - start) * width + u) * 4;
        if (strip_HWRGBA[pixel + 3] < 0.5) continue;
        for (let y = Math.max(0, Math.ceil(v / size) - 1); y <= Math.min(rows - 1, Math.floor(v / size)); y++)
          for (let x = Math.max(0, Math.ceil(u / size) - 1); x <= Math.min(columns - 1, Math.floor(u / size)); x++) {
            const offset = (y * columns + x) * 6;
            for (let axis = 0; axis < 3; axis++) {
              bounds[offset + axis] = Math.min(bounds[offset + axis], strip_HWRGBA[pixel + axis]);
              bounds[offset + axis + 3] = Math.max(bounds[offset + axis + 3], strip_HWRGBA[pixel + axis]);
            }
          }
      }
  }

  finish() {
    const first = this.levels[0];
    for (let offset = 3; offset < first.data_HWRGBA.length; offset += 4)
      first.data_HWRGBA[offset] = first.data_HWRGBA[offset] < 0 ? 0 : 1;
    this.coarsen(0, Number.POSITIVE_INFINITY, 0, Number.POSITIVE_INFINITY);
  }

  coarsen(top, bottom, left, right) {
    for (let level = 1; level < this.levels.length; level++) {
      const previous = this.levels[level - 1];
      const current = this.levels[level];
      top = Math.floor(top / 2);
      bottom = Math.min(current.grid_HW[0] - 1, Math.floor(bottom / 2) + 1);
      left = Math.floor(left / 2);
      right = Math.min(current.grid_HW[1] - 1, Math.floor(right / 2) + 1);
      for (let v = top; v <= bottom; v++)
        for (let u = left; u <= right; u++) {
          const x = Math.min(u * 2, previous.grid_HW[1] - 1);
          const y = Math.min(v * 2, previous.grid_HW[0] - 1);
          const destination = (v * current.grid_HW[1] + u) * 4;
          const origin = (y * previous.grid_HW[1] + x) * 4;
          for (let axis = 0; axis < 4; axis++)
            current.data_HWRGBA[destination + axis] = previous.data_HWRGBA[origin + axis];
          for (let dy = 0; dy <= 1; dy++)
            for (let dx = 0; dx <= 1; dx++) {
              const child =
                (Math.min(y + dy, previous.grid_HW[0] - 1) * previous.grid_HW[1] +
                  Math.min(x + dx, previous.grid_HW[1] - 1)) *
                4;
              current.data_HWRGBA[destination + 3] = Math.min(
                current.data_HWRGBA[destination + 3],
                previous.data_HWRGBA[child + 3],
              );
            }
        }
    }
  }

  async patch(source, start_HW, stop_HW, signal) {
    const [height, width] = source.grid_HW;
    const rows = source.stripRows;
    const first = this.levels[0];
    const size = this.intersectionBlockCells;
    const [top, left] = start_HW.map((start) => Math.max(0, Math.ceil(start / 2) - 1));
    const [bottom, right] = stop_HW.map((stop, axis) =>
      stop >= source.grid_HW[axis] ? first.grid_HW[axis] - 1 : Math.floor((stop - 1) / 2),
    );
    const [blockTop, blockLeft] = start_HW.map((start) => Math.max(0, Math.ceil(start / size) - 1));
    const [blockBottom, blockRight] = stop_HW.map((stop, axis) =>
      Math.min(this.intersectionGrid_HW[axis] - 1, Math.floor((stop - 1) / size)),
    );
    const strips = new Map();
    const lastRow = Math.min(height - 1, Math.max(2 * bottom + 2, (blockBottom + 1) * size));
    for (let index = Math.floor(Math.min(2 * top, blockTop * size) / rows); index <= lastRow / rows; index++)
      strips.set(index, await source.getStrip(index, signal));
    for (let y = top; y <= bottom; y++) {
      const v = y === first.grid_HW[0] - 1 ? height - 1 : 2 * y;
      for (let x = left; x <= right; x++) {
        const u = x === first.grid_HW[1] - 1 ? width - 1 : 2 * x;
        const destination = (y * first.grid_HW[1] + x) * 4;
        const sample = ((v % rows) * width + u) * 4;
        first.data_HWRGBA.set(strips.get(Math.floor(v / rows)).subarray(sample, sample + 3), destination);
        let valid = 1;
        for (let row = Math.min(2 * y, height - 1); row <= Math.min(2 * y + 2, height - 1); row++)
          for (let column = Math.min(2 * x, width - 1); column <= Math.min(2 * x + 2, width - 1); column++)
            if (strips.get(Math.floor(row / rows))[((row % rows) * width + column) * 4 + 3] < 0.5) valid = 0;
        first.data_HWRGBA[destination + 3] = valid;
      }
    }
    for (let y = blockTop; y <= blockBottom; y++)
      for (let x = blockLeft; x <= blockRight; x++)
        this.intersectionBounds_B6.set(EMPTY_BOUNDS, (y * this.intersectionGrid_HW[1] + x) * 6);
    for (const [index, strip] of strips) this.includeIntersectionBounds(source, index, strip);
    this.revision += 1;
    this.coarsen(top, bottom, left, right);
  }

  async overview(source, signal) {
    if (source.grid_HW[0] * source.grid_HW[1] <= SURFACE_OVERVIEW_POINT_LIMIT) {
      const points_zyx_HW3 = new Float32Array(source.grid_HW[0] * source.grid_HW[1] * 3);
      for (let index = 0; index < source.stripCount; index++) {
        const strip_HWRGBA = await source.getStrip(index, signal);
        for (let pixel = 0; pixel < strip_HWRGBA.length / 4; pixel++) {
          const destination = index * source.stripRows * source.grid_HW[1] + pixel;
          if (destination >= points_zyx_HW3.length / 3) break;
          for (let axis = 0; axis < 3; axis++)
            points_zyx_HW3[destination * 3 + axis] =
              strip_HWRGBA[pixel * 4 + 3] < 0.5 ? NaN : strip_HWRGBA[pixel * 4 + axis];
        }
      }
      return { points_zyx_HW3, grid_HW: source.grid_HW, stride: 1 };
    }
    // Budget by area: a per-axis limit discards most geometry on narrow surfaces.
    const level = this.levels.find(
      (candidate) => candidate.grid_HW[0] * candidate.grid_HW[1] <= SURFACE_OVERVIEW_POINT_LIMIT,
    );
    const grid_HW = source.grid_HW.map((size) => Math.ceil(size / level.stride));
    const points_zyx_HW3 = new Float32Array(grid_HW[0] * grid_HW[1] * 3);
    for (let i = 0; i < points_zyx_HW3.length / 3; i++)
      for (let a = 0; a < 3; a++) {
        const pixel = Math.floor(i / grid_HW[1]) * level.grid_HW[1] + (i % grid_HW[1]);
        points_zyx_HW3[i * 3 + a] = level.data_HWRGBA[pixel * 4 + 3] < 0.5 ? NaN : level.data_HWRGBA[pixel * 4 + a];
      }
    return { points_zyx_HW3, grid_HW, stride: level.stride };
  }
}
