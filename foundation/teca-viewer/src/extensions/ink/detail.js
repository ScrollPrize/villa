import { openTiff, readTile } from "./tiff.js";

const TILE_CACHE_BYTES = 16 * 2 ** 20;
// Pixels a side of the region read at once: its detail takes at most 16 MiB, 21 MiB with mipmaps on the GPU.
const REGION_SIDE = 4096;
// Tile reads under way at once; they share the host's connections with the CT's reads.
const READS_AT_ONCE = 4;
// The coarsest downsampling of a detail, which divides a tile's sides (tiff.js has them multiples of 16).
const COARSEST = 16;

export function detailRegion(layout, downsample, view) {
  const detail = Math.min(COARSEST, 2 ** Math.floor(Math.log2(Math.max(1, view.pixel))));
  if (detail >= downsample) return null;
  const { shape_HW, tile_HW, grid_HW } = layout;
  const box = [0, 1].map((axis) => [
    Math.max(0, view.start_HW[axis] / view.scale_UV[1 - axis]),
    Math.min(shape_HW[axis], view.stop_HW[axis] / view.scale_UV[1 - axis]),
  ]);
  if (box.some(([low, high]) => low >= high)) return null;
  const center = box.map(([low, high]) => (low + high) / 2);
  const spans = box.map(([low, high], axis) => {
    const [first, last] = [Math.floor(low / tile_HW[axis]), Math.ceil(high / tile_HW[axis])];
    const most = Math.max(1, Math.floor((REGION_SIDE * detail) / tile_HW[axis]));
    if (last - first <= most) return [first, last];
    const from = Math.min(last - most, Math.max(first, Math.round(center[axis] / tile_HW[axis] - most / 2)));
    return [from, from + most];
  });
  const start_HW = spans.map(([first], axis) => first * tile_HW[axis]);
  const tiles = [];
  for (let row = spans[0][0]; row < spans[0][1]; row++)
    for (let column = spans[1][0]; column < spans[1][1]; column++) tiles.push(row * grid_HW[1] + column);
  const distance = (tile) =>
    Math.hypot(
      ...[Math.floor(tile / grid_HW[1]), tile % grid_HW[1]].map(
        (index, axis) => (index + 0.5) * tile_HW[axis] - center[axis],
      ),
    );
  tiles.sort((a, b) => distance(a) - distance(b));
  const shape = spans.map(([, last], axis) => Math.min(last * tile_HW[axis], shape_HW[axis]) - start_HW[axis]);
  return { downsample: detail, start_HW, shape_HW: shape.map((size) => Math.ceil(size / detail)), tiles };
}

export function downsampled(pixels_HW, tile_HW, downsample) {
  if (downsample === 1) return pixels_HW;
  const [rows, columns] = tile_HW.map((size) => size / downsample);
  const sums = new Uint32Array(columns);
  const kept_HW = new Uint8Array(rows * columns);
  for (let row = 0; row < rows; row++) {
    sums.fill(0);
    for (let at = row * downsample * tile_HW[1]; at < (row + 1) * downsample * tile_HW[1]; )
      for (let column = 0; column < columns; column++)
        for (let step = 0; step < downsample; step++) sums[column] += pixels_HW[at++];
    for (let column = 0; column < columns; column++)
      kept_HW[row * columns + column] = Math.round(sums[column] / downsample ** 2);
  }
  return kept_HW;
}

function holds(region, plan) {
  return (
    region !== null && region.downsample === plan.downsample && plan.tiles.every((tile) => region.tiles.includes(tile))
  );
}

export class Detail {
  constructor(url, surface, shape_HW, failed) {
    this.url = url;
    this.surface = surface;
    this.image_HW = shape_HW;
    this.failed = failed;
    this.layout = null;
    this.downsample = null;
    this.opening = null;
    this.handle = null;
    this.view = null;
    this.cache = new Map();
    this.cacheBytes = 0;
    this.reads = new Set();
    this.queue = [];
    this.region = null;
    this.shown = null;
    this.drawn = false;
  }

  follow(handle, view) {
    if (handle !== this.handle) [this.handle, this.shown, this.drawn] = [handle, null, false];
    this.view = view;
    if (this.layout === null) {
      this.opening ??= this.open();
      return;
    }
    if (handle === null || view?.surface !== this.surface) {
      [this.region, this.queue] = [null, []];
      return;
    }
    if (this.downsample === null) {
      const full_HW = view.grid_HW.map((size, axis) => size / view.scale_UV[1 - axis]);
      if (this.layout.shape_HW.some((size, axis) => Math.abs(size - full_HW[axis]) >= 1)) {
        const shapes = [this.layout.shape_HW, full_HW.map(Math.round)].map((shape_HW) => shape_HW.join(" × "));
        this.failed(new Error(`The TIFF of ${shapes[0]} pixels is not the surface's ${shapes[1]} full pixels.`));
        this.layout = null;
        return;
      }
      this.downsample = this.layout.shape_HW[0] / this.image_HW[0];
    }
    const plan = detailRegion(this.layout, this.downsample, view);
    if (plan === null) {
      if (this.drawn) handle.setDetail(null);
      [this.region, this.queue, this.shown, this.drawn] = [null, [], null, false];
      return;
    }
    if (holds(this.shown, plan)) {
      [this.region, this.queue] = [null, []];
      return;
    }
    if (holds(this.region, plan) && this.region.tiles.length === plan.tiles.length) return;
    const keys = plan.tiles.map((tile) => `${tile}/${plan.downsample}`);
    const data_HW = new Uint8Array(plan.shape_HW[0] * plan.shape_HW[1]);
    this.region = { ...plan, data_HW, missing: new Set(keys), drawn: null };
    for (const [index, key] of keys.entries()) {
      const pixels_HW = this.cache.get(key);
      if (pixels_HW === undefined) continue;
      this.cache.delete(key);
      this.cache.set(key, pixels_HW);
      this.place(plan.tiles[index], key, pixels_HW);
    }
    this.queue = plan.tiles.filter((_, index) => this.region.missing.has(keys[index]));
    this.draw();
    this.pump();
  }

  async open() {
    try {
      this.layout = await openTiff(this.url, null);
      this.follow(this.handle, this.view);
    } catch (error) {
      this.failed(error);
    }
  }

  place(tile, key, pixels_HW) {
    const { tile_HW, grid_HW } = this.layout;
    const { data_HW, shape_HW, start_HW, downsample, missing } = this.region;
    const [rows, columns] = tile_HW.map((size) => size / downsample);
    const [top, left] = [Math.floor(tile / grid_HW[1]), tile % grid_HW[1]].map(
      (index, axis) => (index * tile_HW[axis] - start_HW[axis]) / downsample,
    );
    const across = Math.min(columns, shape_HW[1] - left);
    for (let row = 0; row < Math.min(rows, shape_HW[0] - top); row++)
      data_HW.set(pixels_HW.subarray(row * columns, row * columns + across), (top + row) * shape_HW[1] + left);
    missing.delete(key);
  }

  readRectangle() {
    const { tiles, start_HW, shape_HW, downsample, missing } = this.region;
    const { tile_HW, grid_HW } = this.layout;
    const first_HW = start_HW.map((start, axis) => start / tile_HW[axis]);
    const count_HW = shape_HW.map((size, axis) => Math.ceil((size * downsample) / tile_HW[axis]));
    const read = (row, column) =>
      !missing.has(`${(first_HW[0] + row) * grid_HW[1] + first_HW[1] + column}/${downsample}`);
    const all = (from, to, test) => Array.from({ length: to - from }, (_, index) => from + index).every(test);
    const [row, column] = [Math.floor(tiles[0] / grid_HW[1]) - first_HW[0], (tiles[0] % grid_HW[1]) - first_HW[1]];
    if (!read(row, column)) return null;
    let [top, left, bottom, right] = [row, column, row + 1, column + 1];
    for (let grown = true; grown; ) {
      grown = false;
      if (top > 0 && all(left, right, (at) => read(top - 1, at))) [top, grown] = [top - 1, true];
      if (bottom < count_HW[0] && all(left, right, (at) => read(bottom, at))) [bottom, grown] = [bottom + 1, true];
      if (left > 0 && all(top, bottom, (at) => read(at, left - 1))) [left, grown] = [left - 1, true];
      if (right < count_HW[1] && all(top, bottom, (at) => read(at, right))) [right, grown] = [right + 1, true];
    }
    return [top, left, bottom, right];
  }

  draw() {
    const rectangle = this.readRectangle();
    const region = this.region;
    if (rectangle === null || rectangle.join() === region.drawn?.join()) return;
    region.drawn = rectangle;
    const { data_HW, shape_HW, start_HW, downsample } = region;
    const side_HW = this.layout.tile_HW.map((size) => size / downsample);
    const [top, left] = [rectangle[0] * side_HW[0], rectangle[1] * side_HW[1]];
    const [bottom, right] = [
      Math.min(rectangle[2] * side_HW[0], shape_HW[0]),
      Math.min(rectangle[3] * side_HW[1], shape_HW[1]),
    ];
    const detail = { data_HW, shape_HW, start_HW, downsample };
    if (region.missing.size === 0) [this.shown, this.region, this.queue] = [region, null, []];
    else {
      this.shown = null;
      detail.shape_HW = [bottom - top, right - left];
      detail.start_HW = [start_HW[0] + top * downsample, start_HW[1] + left * downsample];
      detail.data_HW = new Uint8Array(detail.shape_HW[0] * detail.shape_HW[1]);
      for (let row = top; row < bottom; row++)
        detail.data_HW.set(
          data_HW.subarray(row * shape_HW[1] + left, row * shape_HW[1] + right),
          (row - top) * (right - left),
        );
    }
    try {
      this.handle.setDetail(detail);
      this.drawn = true;
    } catch (error) {
      this.shown = null;
      this.failed(error);
    }
  }

  pump() {
    while (this.reads.size < READS_AT_ONCE && this.queue.length > 0) {
      const tile = this.queue.shift();
      const { downsample } = this.region;
      const key = `${tile}/${downsample}`;
      if (this.reads.has(key) || this.cache.has(key)) continue;
      this.reads.add(key);
      readTile(this.url, this.layout, tile, null)
        .then((pixels_HW) => {
          const kept_HW = downsampled(pixels_HW, this.layout.tile_HW, downsample);
          this.keep(key, kept_HW);
          if (!this.region?.missing.has(key)) return;
          this.place(tile, key, kept_HW);
          this.draw();
        })
        .catch(this.failed)
        .finally(() => {
          this.reads.delete(key);
          this.pump();
        });
    }
  }

  keep(key, pixels_HW) {
    this.cache.set(key, pixels_HW);
    this.cacheBytes += pixels_HW.length;
    for (const [oldest, kept_HW] of this.cache) {
      if (this.cacheBytes <= TILE_CACHE_BYTES) break;
      this.cache.delete(oldest);
      this.cacheBytes -= kept_HW.length;
    }
  }
}
