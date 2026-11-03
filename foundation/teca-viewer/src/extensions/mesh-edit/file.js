// x.tif, y.tif and z.tif are each one little-endian float32 page in one uncompressed strip. This viewer and Python's
// tifffile read them; no other reader is known to.

import { onSurface } from "./stroke.js";

const PIXELS_OFFSET = 160;
const TAGS = [
  [256, 4, "width"],
  [257, 4, "height"],
  [258, 3, 32],
  [259, 3, 1],
  [262, 3, 1],
  [273, 4, PIXELS_OFFSET],
  [277, 3, 1],
  [278, 4, "height"],
  [279, 4, "bytes"],
  [284, 3, 1],
  [339, 3, 3],
];

export function tiff(height, width) {
  if (new Uint8Array(Uint16Array.of(1).buffer)[0] !== 1)
    throw new Error("Save writes little-endian TIFFs through a Float32Array, and this host is big-endian.");
  const sizes = { width, height, bytes: 4 * height * width };
  const bytes = new Uint8Array(PIXELS_OFFSET + sizes.bytes);
  const view = new DataView(bytes.buffer);
  view.setUint16(0, 0x4949);
  view.setUint16(2, 42, true);
  view.setUint32(4, 8, true);
  view.setUint16(8, TAGS.length, true);
  TAGS.forEach(([tag, type, value], index) => {
    const at = 10 + 12 * index;
    view.setUint16(at, tag, true);
    view.setUint16(at + 2, type, true);
    view.setUint32(at + 4, 1, true);
    if (type === 3) view.setUint16(at + 8, sizes[value] ?? value, true);
    else view.setUint32(at + 8, sizes[value] ?? value, true);
  });
  return { bytes, pixels_HW: new Float32Array(bytes.buffer, PIXELS_OFFSET, height * width) };
}

export function* surfaceFiles(surface, name) {
  const [height, width] = surface.grid_HW;
  const points = surface.points_zyx_HW3;
  const low = [Infinity, Infinity, Infinity];
  const high = [-Infinity, -Infinity, -Infinity];
  for (let cell = 0; cell < height * width; cell++) {
    if (!onSurface(points, cell)) continue;
    for (let axis = 0; axis < 3; axis++) {
      low[axis] = Math.min(low[axis], points[3 * cell + 2 - axis]);
      high[axis] = Math.max(high[axis], points[3 * cell + 2 - axis]);
    }
  }
  const meta = { format: "tifxyz", type: "seg", uuid: name, scale: surface.scale_UV, bbox: [low, high] };
  yield { name: `${name}/meta.json`, bytes: new TextEncoder().encode(JSON.stringify(meta)) };
  for (const [axis, letter] of ["x", "y", "z"].entries()) {
    const { bytes, pixels_HW } = tiff(height, width);
    for (let cell = 0; cell < height * width; cell++) pixels_HW[cell] = points[3 * cell + 2 - axis];
    yield { name: `${name}/${letter}.tif`, bytes };
  }
}
