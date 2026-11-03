// Marched on the CPU as ct-box/material.js marches it on the GPU, but each texel is read whole where the GPU reads
// between them.

export function rayPick(box_ZYX, texels_XYZ, origin_XYZ, direction_XYZ, { black, white, density, maximum }) {
  let enter = 0;
  let leave = Number.POSITIVE_INFINITY;
  for (let axis = 0; axis < 3; axis++) {
    const near = (-0.5 - origin_XYZ[axis]) / direction_XYZ[axis];
    const far = (0.5 - origin_XYZ[axis]) / direction_XYZ[axis];
    enter = Math.max(enter, Math.min(near, far));
    leave = Math.min(leave, Math.max(near, far));
  }
  if (!(enter < leave)) return null;
  const stride = 0.5 / Math.max(...texels_XYZ.map((texels, axis) => Math.abs(direction_XYZ[axis]) * texels));
  const texelsPerStride = Math.hypot(...texels_XYZ.map((texels, axis) => direction_XYZ[axis] * texels)) * stride;
  const [sizeX, sizeY] = texels_XYZ;
  let brightest = 0;
  let picked = null;
  let transmitted = 1;
  for (let t = enter + stride / 2; t <= leave; t += stride) {
    const [x, y, z] = texels_XYZ.map((texels, axis) =>
      Math.min(texels - 1, Math.max(0, Math.floor((origin_XYZ[axis] + t * direction_XYZ[axis] + 0.5) * texels))),
    );
    const grey = Math.min(1, Math.max(0, (box_ZYX[(z * sizeY + y) * sizeX + x] - black) / (white - black)));
    if (maximum) {
      if (grey > brightest) [brightest, picked] = [grey, t];
    } else if (grey > 0) {
      transmitted *= Math.exp(-density * grey * texelsPerStride);
      if (transmitted < 0.5) return { enter, picked: t };
    }
  }
  return picked === null ? null : { enter, picked };
}
