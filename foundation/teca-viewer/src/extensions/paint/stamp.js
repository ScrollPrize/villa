export const TILE = 16;
const PAUSE_VOXELS = 4096;
// Tiles along each axis a level holds at most: a tile's key holds its index on each axis in 17 bits, 51 in all, which a
// double holds exactly.
const TILE_INDICES = 2 ** 17;
export const MAX_SIDE = TILE * TILE_INDICES;

export function cropped(data_DHW, shape_ZYX, from_ZYX, size_ZYX) {
  const out_DHW = new data_DHW.constructor(size_ZYX[0] * size_ZYX[1] * size_ZYX[2]);
  for (let z = 0; z < size_ZYX[0]; z++) {
    for (let y = 0; y < size_ZYX[1]; y++) {
      const from = ((z + from_ZYX[0]) * shape_ZYX[1] + y + from_ZYX[1]) * shape_ZYX[2] + from_ZYX[2];
      out_DHW.set(data_DHW.subarray(from, from + size_ZYX[2]), (z * size_ZYX[1] + y) * size_ZYX[2]);
    }
  }
  return out_DHW;
}

// A stamp walks a tile's runs and not its box, of which a disc on a tilted plane sets one voxel in 8 to 21.
export async function voxelTiles(voxels_N3, pause) {
  const boxes = new Map();
  const bounds = [];
  const counts = [];
  const box = new Int32Array(voxels_N3.length / 3);
  let lastKey = -1;
  let last = -1;
  for (let voxel = 0; voxel < box.length; voxel++) {
    if (voxel % PAUSE_VOXELS === 0) await pause();
    const z = voxels_N3[voxel * 3];
    const y = voxels_N3[voxel * 3 + 1];
    const x = voxels_N3[voxel * 3 + 2];
    const key = (Math.floor(z / TILE) * TILE_INDICES + Math.floor(y / TILE)) * TILE_INDICES + Math.floor(x / TILE);
    if (key !== lastKey) {
      lastKey = key;
      last = boxes.get(key) ?? -1;
      if (last < 0) {
        last = boxes.size;
        boxes.set(key, last);
        bounds.push(z, y, x, z + 1, y + 1, x + 1);
        counts.push(0);
      }
    }
    box[voxel] = last;
    counts[last] += 1;
    const at = last * 6;
    if (z < bounds[at]) bounds[at] = z;
    if (y < bounds[at + 1]) bounds[at + 1] = y;
    if (x < bounds[at + 2]) bounds[at + 2] = x;
    if (z >= bounds[at + 3]) bounds[at + 3] = z + 1;
    if (y >= bounds[at + 4]) bounds[at + 4] = y + 1;
    if (x >= bounds[at + 5]) bounds[at + 5] = x + 1;
  }
  const voxels = counts.map((count) => new Uint16Array(count));
  const filled = new Int32Array(voxels.length);
  for (let voxel = 0; voxel < box.length; voxel++) {
    if (voxel % PAUSE_VOXELS === 0) await pause();
    const at = voxel * 3;
    const low = box[voxel] * 6;
    voxels[box[voxel]][filled[box[voxel]]++] =
      ((voxels_N3[at] - bounds[low]) * (bounds[low + 4] - bounds[low + 1]) + voxels_N3[at + 1] - bounds[low + 1]) *
        (bounds[low + 5] - bounds[low + 2]) +
      voxels_N3[at + 2] -
      bounds[low + 2];
  }
  const tiles = [];
  for (let index = 0; index < voxels.length; index++) {
    await pause();
    const from_ZYX = bounds.slice(index * 6, index * 6 + 3);
    const shape_ZYX = from_ZYX.map((low, axis) => bounds[index * 6 + 3 + axis] - low);
    const [, sizeY, sizeX] = shape_ZYX;
    const indices_N = voxels[index].sort();
    // A box is at most TILE voxels a side, so its coordinates and lengths fit 8 bits.
    const runs_N4 = new Uint8Array(indices_N.length * 4);
    let length = 0;
    for (let voxel = 0; voxel < indices_N.length; voxel++) {
      const at = indices_N[voxel];
      if (voxel > 0 && at === indices_N[voxel - 1] + 1 && at % sizeX !== 0) {
        runs_N4[length - 1] += 1;
        continue;
      }
      runs_N4[length] = Math.floor(at / (sizeY * sizeX));
      runs_N4[length + 1] = Math.floor(at / sizeX) % sizeY;
      runs_N4[length + 2] = at % sizeX;
      runs_N4[length + 3] = 1;
      length += 4;
    }
    tiles.push({ from_ZYX, shape_ZYX, axis: 2, runs_N4: runs_N4.slice(0, length) });
  }
  return tiles;
}

export function poolRatios(scales) {
  return scales.map((scale, level) => {
    if (level === 0) return [1, 1, 1];
    const below = scales[level - 1];
    const ratio_ZYX = scale.downsample_ZYX.map((size, axis) => size / below.downsample_ZYX[axis]);
    const whole = ratio_ZYX.every(
      (ratio, axis) =>
        Number.isInteger(ratio) &&
        scale.chunks_ZYX[axis] === below.chunks_ZYX[axis] &&
        scale.chunks_ZYX[axis] % ratio === 0 &&
        scale.offset_ZYX[axis] === below.offset_ZYX[axis],
    );
    if (!whole)
      throw new Error(
        `CT level ${level} (voxel ${scale.downsample_ZYX.join("×")}, origin ${scale.offset_ZYX.join(",")}, chunk ` +
          `${scale.chunks_ZYX.join("×")}) does not pool whole blocks of level ${level - 1} ` +
          `(${below.downsample_ZYX.join("×")}, ${below.offset_ZYX.join(",")}, ${below.chunks_ZYX.join("×")}).`,
      );
    return ratio_ZYX;
  });
}

export function pooled(data_DHW, shape_ZYX, from_ZYX, size_ZYX, ratio_ZYX) {
  const out_ZYX = size_ZYX.map((size, axis) => Math.ceil(size / ratio_ZYX[axis]));
  const out_DHW = new data_DHW.constructor(out_ZYX[0] * out_ZYX[1] * out_ZYX[2]);
  for (let z = 0; z < size_ZYX[0]; z++) {
    for (let y = 0; y < size_ZYX[1]; y++) {
      const from = ((z + from_ZYX[0]) * shape_ZYX[1] + y + from_ZYX[1]) * shape_ZYX[2] + from_ZYX[2];
      const row = (Math.floor(z / ratio_ZYX[0]) * out_ZYX[1] + Math.floor(y / ratio_ZYX[1])) * out_ZYX[2];
      for (let x = 0; x < size_ZYX[2]; x++) {
        const label = data_DHW[from + x];
        if (label === 0) continue;
        const to = row + Math.floor(x / ratio_ZYX[2]);
        if (label > out_DHW[to]) out_DHW[to] = label;
      }
    }
  }
  return { shape_ZYX: out_ZYX, out_DHW };
}
