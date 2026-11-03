// Voxel g of a level with downsample d and offset o is centred at world o + g·d + (d − 1)/2 and covers world [o + g·d −
// 1/2, o + (g + 1)·d − 1/2).

export function boxLevel(scales, side, texels) {
  const index = scales.findIndex((scale) => scale.downsample_ZYX.every((d) => Math.ceil(side / d) <= texels));
  if (index === -1) throw new Error(`No level of the CT holds ${side} voxels in ${texels} texels.`);
  return index;
}

export function boxRegion(scale, center_ZYX, side) {
  const start_ZYX = [];
  const size_ZYX = [];
  for (let axis = 0; axis < 3; axis++) {
    const d = scale.downsample_ZYX[axis];
    const count = Math.ceil(side / d);
    const center = (center_ZYX[axis] - scale.offset_ZYX[axis] - (d - 1) / 2) / d;
    const first = Math.ceil(center - count / 2);
    const from = Math.max(0, first);
    const to = Math.min(scale.shape_ZYX[axis], first + count);
    if (to <= from) throw new Error(`The box about ${center_ZYX} lies outside the CT.`);
    start_ZYX.push(from);
    size_ZYX.push(to - from);
  }
  return {
    start_ZYX,
    size_ZYX,
    min_ZYX: start_ZYX.map((g, axis) => scale.offset_ZYX[axis] + g * scale.downsample_ZYX[axis] - 0.5),
    extent_ZYX: size_ZYX.map((n, axis) => n * scale.downsample_ZYX[axis]),
  };
}

export function regionChunks(scale, region, held) {
  const shape = scale.chunks_ZYX;
  const [first, last] = [0, 1].map((end) =>
    region.start_ZYX.map((g, axis) => Math.floor((g + end * (region.size_ZYX[axis] - 1)) / shape[axis])),
  );
  const chunks = [];
  for (let z = first[0]; z <= last[0]; z++)
    for (let y = first[1]; y <= last[1]; y++)
      for (let x = first[2]; x <= last[2]; x++) {
        const inHeld =
          held !== null &&
          [z, y, x].every((c, axis) => {
            const from = Math.max(c * shape[axis], region.start_ZYX[axis]);
            const to = Math.min((c + 1) * shape[axis], region.start_ZYX[axis] + region.size_ZYX[axis]);
            return held.start_ZYX[axis] <= from && to <= held.start_ZYX[axis] + held.size_ZYX[axis];
          });
        if (!inHeld) chunks.push([z, y, x]);
      }
  return chunks;
}

export function displayBytes(values, encoding) {
  if (encoding.kind === "identity_uint8") return values;
  if (encoding.kind !== "linear_window") throw new Error(`The 3D CT does not draw display encoding ${encoding.kind}.`);
  // A Uint8ClampedArray rounds and clamps what it is given; NaN, unknown, becomes 0.
  const bytes = new Uint8ClampedArray(values.length);
  const scale = 255 / (encoding.maximum - encoding.minimum);
  for (let index = 0; index < values.length; index++) bytes[index] = (values[index] - encoding.minimum) * scale;
  return new Uint8Array(bytes.buffer);
}

export function copyBlock(box_ZYX, region, bytes, origin, shape) {
  if (bytes.length !== shape[0] * shape[1] * shape[2])
    throw new Error(`The block at ${origin} has ${bytes.length} values, not ${shape.join(" × ")}.`);
  const from = origin.map((o, axis) => Math.max(o, region.start_ZYX[axis]));
  const to = origin.map((o, axis) => Math.min(o + shape[axis], region.start_ZYX[axis] + region.size_ZYX[axis]));
  if (from.some((f, axis) => f >= to[axis])) return;
  const [, sizeY, sizeX] = region.size_ZYX;
  for (let z = from[0]; z < to[0]; z++)
    for (let y = from[1]; y < to[1]; y++) {
      const source = ((z - origin[0]) * shape[1] + (y - origin[1])) * shape[2] - origin[2];
      const target = ((z - region.start_ZYX[0]) * sizeY + (y - region.start_ZYX[1])) * sizeX - region.start_ZYX[2];
      box_ZYX.set(bytes.subarray(source + from[2], source + to[2]), target + from[2]);
    }
}
