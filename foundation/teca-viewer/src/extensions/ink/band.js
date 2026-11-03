// Level voxels apart that the band's points are taken along the surface, at most; below one, so no voxel it crosses
// is left between them.
const SPACING = 0.7;
const BLOCK = 16;

export function prepared({ grid_HW, points_zyx_HW3 }) {
  const [height, width] = grid_HW;
  const valid_HW = new Uint8Array(height * width);
  for (let cell = 0; cell < height * width; cell++) {
    const [z, y, x] = points_zyx_HW3.subarray(3 * cell, 3 * cell + 3);
    valid_HW[cell] = Number.isFinite(z + y + x) && (z !== -1 || y !== -1 || x !== -1) ? 1 : 0;
  }
  const blocks_HW = [Math.ceil(height / BLOCK), Math.ceil(width / BLOCK)];
  const blocks_N6 = new Float32Array(6 * blocks_HW[0] * blocks_HW[1]);
  for (let blockRow = 0; blockRow < blocks_HW[0]; blockRow++)
    for (let blockColumn = 0; blockColumn < blocks_HW[1]; blockColumn++) {
      const at = 6 * (blockRow * blocks_HW[1] + blockColumn);
      blocks_N6.fill(Infinity, at, at + 3);
      blocks_N6.fill(-Infinity, at + 3, at + 6);
      for (let row = blockRow * BLOCK; row <= Math.min(height - 1, (blockRow + 1) * BLOCK); row++)
        for (let column = blockColumn * BLOCK; column <= Math.min(width - 1, (blockColumn + 1) * BLOCK); column++) {
          if (valid_HW[row * width + column] === 0) continue;
          for (let axis = 0; axis < 3; axis++) {
            const value = points_zyx_HW3[3 * (row * width + column) + axis];
            if (value < blocks_N6[at + axis]) blocks_N6[at + axis] = value;
            if (value > blocks_N6[at + 3 + axis]) blocks_N6[at + 3 + axis] = value;
          }
        }
    }
  return { grid_HW, points_zyx_HW3, valid_HW, blocks_HW, blocks_N6 };
}

function offsets(steps, apart) {
  const count = Math.ceil((2 * steps) / apart);
  return Float64Array.from({ length: count + 1 }, (_, index) =>
    count === 0 ? 0 : -steps + (2 * steps * index) / count,
  );
}

export function chunkBand(surface, { data_HW, shape_HW }, scale, chunk_ZYX, steps) {
  const { grid_HW, points_zyx_HW3: points, valid_HW, blocks_HW, blocks_N6 } = surface;
  const { shape_ZYX, chunks_ZYX, downsample_ZYX, offset_ZYX } = scale;
  const [height, width] = grid_HW;
  const [imageHeight, imageWidth] = shape_HW;
  const [, chunkY, chunkX] = chunks_ZYX;
  const [dz, dy, dx] = downsample_ZYX;
  const [oz, oy, ox] = offset_ZYX.map((offset) => offset - 0.5);
  const [firstZ, firstY, firstX] = chunk_ZYX.map((chunk, axis) => chunk * chunks_ZYX[axis]);
  const [endZ, endY, endX] = chunk_ZYX.map((chunk, axis) => Math.min(shape_ZYX[axis], (chunk + 1) * chunks_ZYX[axis]));
  const [lowZ, lowY, lowX] = [firstZ * dz, firstY * dy, firstX * dx];
  const [highZ, highY, highX] = [endZ * dz, endY * dy, endX * dx];
  const reach = steps + 1;
  const reaches = (bounds, at) =>
    bounds[at + 3] + reach - oz >= lowZ &&
    bounds[at] - reach - oz < highZ &&
    bounds[at + 4] + reach - oy >= lowY &&
    bounds[at + 1] - reach - oy < highY &&
    bounds[at + 5] + reach - ox >= lowX &&
    bounds[at + 2] - reach - ox < highX;
  const apart = SPACING * Math.min(dz, dy, dx);
  const along = offsets(steps, apart);
  const last = along.length - 1;
  const bounds = new Float64Array(6);
  let data_DHW = null;
  for (let blockRow = 0; blockRow < blocks_HW[0]; blockRow++)
    for (let blockColumn = 0; blockColumn < blocks_HW[1]; blockColumn++) {
      if (!reaches(blocks_N6, 6 * (blockRow * blocks_HW[1] + blockColumn))) continue;
      const lastRow = Math.min(height - 1, (blockRow + 1) * BLOCK);
      const lastColumn = Math.min(width - 1, (blockColumn + 1) * BLOCK);
      for (let row = blockRow * BLOCK; row < lastRow; row++)
        for (let column = blockColumn * BLOCK; column < lastColumn; column++) {
          const a = row * width + column;
          const b = a + 1;
          const c = a + width;
          const d = c + 1;
          if ((valid_HW[a] & valid_HW[b] & valid_HW[c] & valid_HW[d]) === 0) continue;
          for (let axis = 0; axis < 3; axis++) {
            const pa = points[3 * a + axis];
            const pb = points[3 * b + axis];
            const pc = points[3 * c + axis];
            const pd = points[3 * d + axis];
            bounds[axis] = Math.min(pa, pb, pc, pd);
            bounds[3 + axis] = Math.max(pa, pb, pc, pd);
          }
          if (!reaches(bounds, 0)) continue;
          const az = points[3 * a];
          const ay = points[3 * a + 1];
          const ax = points[3 * a + 2];
          const uz = points[3 * b] - az;
          const uy = points[3 * b + 1] - ay;
          const ux = points[3 * b + 2] - ax;
          const vz = points[3 * c] - az;
          const vy = points[3 * c + 1] - ay;
          const vx = points[3 * c + 2] - ax;
          const wz = points[3 * d] - az - uz - vz;
          const wy = points[3 * d + 1] - ay - uy - vy;
          const wx = points[3 * d + 2] - ax - ux - vx;
          // Math.hypot is several times slower than the square root it takes.
          const longest = Math.sqrt(
            Math.max(
              uz * uz + uy * uy + ux * ux,
              vz * vz + vy * vy + vx * vx,
              (uz + wz) ** 2 + (uy + wy) ** 2 + (ux + wx) ** 2,
              (vz + wz) ** 2 + (vy + wy) ** 2 + (vx + wx) ** 2,
            ),
          );
          const parts = Math.max(1, Math.ceil(longest / apart));
          for (let i = 0; i <= parts; i++) {
            const v = i / parts;
            const pixelRow = Math.min(imageHeight - 1, Math.floor(((row + v) * imageHeight) / height)) * imageWidth;
            const suz = uz + v * wz;
            const suy = uy + v * wy;
            const sux = ux + v * wx;
            for (let j = 0; j <= parts; j++) {
              const u = j / parts;
              const value =
                data_HW[pixelRow + Math.min(imageWidth - 1, Math.floor(((column + u) * imageWidth) / width))];
              if (value === 0) continue;
              const svz = vz + u * wz;
              const svy = vy + u * wy;
              const svx = vx + u * wx;
              let nz = suy * svx - sux * svy;
              let ny = sux * svz - suz * svx;
              let nx = suz * svy - suy * svz;
              const norm = Math.sqrt(nz * nz + ny * ny + nx * nx);
              if (norm === 0) continue;
              nz /= norm;
              ny /= norm;
              nx /= norm;
              const pz = az + u * uz + v * svz - oz;
              const py = ay + u * uy + v * svy - oy;
              const px = ax + u * ux + v * svx - ox;
              let from = -steps;
              let to = steps;
              if (nz !== 0) {
                const near = (lowZ - pz) / nz;
                const far = (highZ - pz) / nz;
                from = Math.max(from, Math.min(near, far));
                to = Math.min(to, Math.max(near, far));
              } else if (pz < lowZ || pz >= highZ) continue;
              if (ny !== 0) {
                const near = (lowY - py) / ny;
                const far = (highY - py) / ny;
                from = Math.max(from, Math.min(near, far));
                to = Math.min(to, Math.max(near, far));
              } else if (py < lowY || py >= highY) continue;
              if (nx !== 0) {
                const near = (lowX - px) / nx;
                const far = (highX - px) / nx;
                from = Math.max(from, Math.min(near, far));
                to = Math.min(to, Math.max(near, far));
              } else if (px < lowX || px >= highX) continue;
              if (from > to) continue;
              const start = last === 0 ? 0 : Math.max(0, Math.floor(((from + steps) * last) / (2 * steps)) - 1);
              const stop = last === 0 ? 0 : Math.min(last, Math.ceil(((to + steps) * last) / (2 * steps)) + 1);
              for (let step = start; step <= stop; step++) {
                const t = along[step];
                const gz = Math.floor((pz + t * nz) / dz);
                const gy = Math.floor((py + t * ny) / dy);
                const gx = Math.floor((px + t * nx) / dx);
                if (gz < firstZ || gy < firstY || gx < firstX || gz >= endZ || gy >= endY || gx >= endX) continue;
                data_DHW ??= new Uint8Array(chunks_ZYX[0] * chunkY * chunkX);
                const voxel = ((gz - firstZ) * chunkY + gy - firstY) * chunkX + gx - firstX;
                if (data_DHW[voxel] < value) data_DHW[voxel] = value;
              }
            }
          }
        }
    }
  return data_DHW;
}

export function boxMesh({ grid_HW, points_zyx_HW3: points, valid_HW, blocks_HW, blocks_N6 }, low_ZYX, high_ZYX) {
  const [height, width] = grid_HW;
  const near = (cell) =>
    [0, 1, 2].every((axis) => points[3 * cell + axis] >= low_ZYX[axis] && points[3 * cell + axis] <= high_ZYX[axis]);
  const vertices = new Map();
  const vertex = (cell) => vertices.get(cell) ?? vertices.set(cell, vertices.size).get(cell);
  const triangles = [];
  for (let blockRow = 0; blockRow < blocks_HW[0]; blockRow++)
    for (let blockColumn = 0; blockColumn < blocks_HW[1]; blockColumn++) {
      const at = 6 * (blockRow * blocks_HW[1] + blockColumn);
      if ([0, 1, 2].some((axis) => blocks_N6[at + 3 + axis] < low_ZYX[axis] || blocks_N6[at + axis] > high_ZYX[axis]))
        continue;
      for (let row = blockRow * BLOCK; row < Math.min(height - 1, (blockRow + 1) * BLOCK); row++)
        for (let column = blockColumn * BLOCK; column < Math.min(width - 1, (blockColumn + 1) * BLOCK); column++) {
          const corners = [row * width + column, row * width + column + 1, (row + 1) * width + column];
          corners.push(corners[2] + 1);
          if (corners.some((cell) => valid_HW[cell] === 0) || !corners.some(near)) continue;
          const [a, b, c, d] = corners.map(vertex);
          triangles.push(a, c, b, b, c, d);
        }
    }
  const positions_xyz_N3 = new Float32Array(3 * vertices.size);
  const uvs_N2 = new Float32Array(2 * vertices.size);
  for (const [cell, index] of vertices) {
    for (let axis = 0; axis < 3; axis++) positions_xyz_N3[3 * index + axis] = points[3 * cell + 2 - axis];
    uvs_N2[2 * index] = (cell % width) / width;
    uvs_N2[2 * index + 1] = Math.floor(cell / width) / height;
  }
  return { positions_xyz_N3, uvs_N2, triangles_T3: Uint32Array.from(triangles) };
}
