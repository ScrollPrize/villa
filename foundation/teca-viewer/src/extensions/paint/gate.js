const GATE_PRIORITY = 400;

export function rawThreshold(displayEncoding, threshold) {
  if (displayEncoding.kind === "identity_uint8") return threshold;
  const { minimum, maximum } = displayEncoding;
  return minimum + (threshold / 255) * (maximum - minimum);
}

export class Gate {
  constructor(ct, held, open, threshold) {
    this.ct = ct;
    this.held = held;
    this.open = open;
    this.opening = null;
    this.reader = null;
    this.raw = rawThreshold(ct.metadata.displayEncoding, threshold);
    this.ratios = ct.scales.map((scale) =>
      scale.downsample_ZYX.map((size, axis) => size / ct.scales[0].downsample_ZYX[axis]),
    );
    this.grids = ct.scales.map(({ shape_ZYX, chunks_ZYX }) =>
      shape_ZYX.map((size, axis) => Math.ceil(size / chunks_ZYX[axis])),
    );
    this.levels = new Map();
    this.chunks = ct.scales.map(() => new Map());
    this.last = { region: -1, level: -1, index: -1, data_DHW: null };
  }

  index(level, chunk_ZYX) {
    const grid_ZYX = this.grids[level];
    return (chunk_ZYX[0] * grid_ZYX[1] + chunk_ZYX[1]) * grid_ZYX[2] + chunk_ZYX[2];
  }

  chunk(level, chunk_ZYX, index) {
    const known = this.chunks[level].get(index);
    if (known !== undefined) return known instanceof Promise ? null : known;
    const data_DHW = this.held(level, chunk_ZYX);
    if (data_DHW !== null) this.chunks[level].set(index, data_DHW);
    return data_DHW;
  }

  cover(level, low_ZYX, high_ZYX) {
    const { chunks_ZYX } = this.ct.scales[level];
    const [first_ZYX, last_ZYX] = [low_ZYX, high_ZYX.map((high) => high - 1)].map((voxel_ZYX) =>
      voxel_ZYX.map((voxel, axis) => Math.floor(Math.floor(voxel / this.ratios[level][axis]) / chunks_ZYX[axis])),
    );
    const cover = [];
    for (let cz = first_ZYX[0]; cz <= last_ZYX[0]; cz++)
      for (let cy = first_ZYX[1]; cy <= last_ZYX[1]; cy++)
        for (let cx = first_ZYX[2]; cx <= last_ZYX[2]; cx++) cover.push([cz, cy, cx]);
    return cover;
  }

  level(region_ZYX) {
    const region = this.index(0, region_ZYX);
    const known = this.levels.get(region);
    if (known !== undefined) return known;
    const { shape_ZYX, chunks_ZYX } = this.ct.scales[0];
    const low_ZYX = region_ZYX.map((chunk, axis) => chunk * chunks_ZYX[axis]);
    const high_ZYX = low_ZYX.map((low, axis) => Math.min(shape_ZYX[axis], low + chunks_ZYX[axis]));
    for (let level = 0; level < this.ct.scales.length; level++) {
      const whole = this.cover(level, low_ZYX, high_ZYX).every(
        (chunk_ZYX) => this.chunk(level, chunk_ZYX, this.index(level, chunk_ZYX)) !== null,
      );
      if (!whole) continue;
      this.levels.set(region, level);
      return level;
    }
    return -1;
  }

  async load(low_ZYX, high_ZYX) {
    const coarsest = this.ct.scales.length - 1;
    const reads = [];
    for (const region_ZYX of this.cover(0, low_ZYX, high_ZYX)) {
      if (this.level(region_ZYX) >= 0) continue;
      const { shape_ZYX, chunks_ZYX } = this.ct.scales[0];
      const from_ZYX = region_ZYX.map((chunk, axis) => chunk * chunks_ZYX[axis]);
      const to_ZYX = from_ZYX.map((from, axis) => Math.min(shape_ZYX[axis], from + chunks_ZYX[axis]));
      for (const chunk_ZYX of this.cover(coarsest, from_ZYX, to_ZYX)) {
        const index = this.index(coarsest, chunk_ZYX);
        const known = this.chunks[coarsest].get(index);
        if (known instanceof Promise) reads.push(known);
        if (known !== undefined) continue;
        const read = this.read(coarsest, chunk_ZYX, index);
        this.chunks[coarsest].set(index, read);
        reads.push(read);
      }
    }
    await Promise.all(reads);
  }

  async read(level, chunk_ZYX, index) {
    try {
      this.opening ??= this.open(this.ct.rootUrl, null);
      this.reader = await this.opening;
      const data_DHW = await this.reader.getChunk(...chunk_ZYX, 0, this.reader.scales[level], null, GATE_PRIORITY);
      this.chunks[level].set(index, data_DHW);
    } catch (error) {
      this.chunks[level].delete(index);
      throw error;
    }
  }

  sample(z, y, x) {
    const [regionZ, regionY, regionX] = this.ct.scales[0].chunks_ZYX;
    const [, gridY, gridX] = this.grids[0];
    const region = (Math.floor(z / regionZ) * gridY + Math.floor(y / regionY)) * gridX;
    const last = this.last;
    if (last.region !== region + Math.floor(x / regionX)) {
      last.region = region + Math.floor(x / regionX);
      last.level = this.level([Math.floor(z / regionZ), Math.floor(y / regionY), Math.floor(x / regionX)]);
      last.index = -1;
    }
    if (last.level < 0) {
      last.region = -1;
      return null;
    }
    const ratio_ZYX = this.ratios[last.level];
    const { chunks_ZYX } = this.ct.scales[last.level];
    const lz = Math.floor(z / ratio_ZYX[0]);
    const ly = Math.floor(y / ratio_ZYX[1]);
    const lx = Math.floor(x / ratio_ZYX[2]);
    const cz = Math.floor(lz / chunks_ZYX[0]);
    const cy = Math.floor(ly / chunks_ZYX[1]);
    const cx = Math.floor(lx / chunks_ZYX[2]);
    const [, levelY, levelX] = this.grids[last.level];
    const index = (cz * levelY + cy) * levelX + cx;
    if (last.index !== index) {
      last.index = index;
      last.data_DHW = this.chunk(last.level, [cz, cy, cx], index);
    }
    return last.data_DHW[
      ((lz - cz * chunks_ZYX[0]) * chunks_ZYX[1] + ly - cy * chunks_ZYX[1]) * chunks_ZYX[2] + lx - cx * chunks_ZYX[2]
    ];
  }

  passes(z, y, x) {
    const value = this.sample(z, y, x);
    return value === null ? null : value >= this.raw;
  }

  release() {
    for (const chunks of this.chunks) chunks.clear();
    this.levels.clear();
    this.last = { region: -1, level: -1, index: -1, data_DHW: null };
    this.reader?.dispose();
  }
}
