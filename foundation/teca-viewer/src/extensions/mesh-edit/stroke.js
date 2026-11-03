// A vertex whose x, y and z are all -1, or not finite, is not on the surface. A vertex r of the radius away from the
// one grabbed, measured across the grid in voxels (a grid step spans 1 / scale of them), takes (1 − r²)² of the edit,
// so it is smooth at its rim.

const AXES_HW = [
  [1, 0],
  [0, 1],
];
// The grid's Laplacian applied twice, `[dr, dc, weight]`: zero on a quadratic, so a sheet held by it keeps its curve.
const THIN_PLATE = [
  [0, 0, 20],
  [-1, 0, -8],
  [1, 0, -8],
  [0, -1, -8],
  [0, 1, -8],
  [-1, -1, 2],
  [-1, 1, 2],
  [1, -1, 2],
  [1, 1, 2],
  [-2, 0, 1],
  [2, 0, 1],
  [0, -2, 1],
  [0, 2, 1],
];
// Conjugate-gradient steps a growing takes at most, about twice what a hole of radius 512 voxels at 0.05 needs, and
// the root mean square of the stencil's residual, in voxels, at which it stops.
const THIN_PLATE_STEPS = 2500;
const THIN_PLATE_RESIDUAL = 1e-6;

export function onSurface(points_zyx_N3, index) {
  const [z, y, x] = points_zyx_N3.subarray(3 * index, 3 * index + 3);
  return Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z) && (x !== -1 || y !== -1 || z !== -1);
}

export function onRim(points_zyx_HW3, grid_HW, index) {
  const [height, width] = grid_HW;
  const row = Math.floor(index / width);
  const column = index % width;
  if (!onSurface(points_zyx_HW3, index)) return false;
  for (const [dr, dc] of AXES_HW)
    for (const sign of [-1, 1]) {
      const [r, c] = [row + sign * dr, column + sign * dc];
      if (r >= 0 && r < height && c >= 0 && c < width && !onSurface(points_zyx_HW3, r * width + c)) return true;
    }
  return false;
}

export function nearestVertex(points_zyx_N3, point_ZYX, reach, admits) {
  let nearest = -1;
  let best = reach * reach;
  for (let index = 0; index < points_zyx_N3.length / 3; index++) {
    const dz = points_zyx_N3[3 * index] - point_ZYX[0];
    const dy = points_zyx_N3[3 * index + 1] - point_ZYX[1];
    const dx = points_zyx_N3[3 * index + 2] - point_ZYX[2];
    const distance = dz * dz + dy * dy + dx * dx;
    if (distance < best && admits(index)) {
      best = distance;
      nearest = index;
    }
  }
  return nearest;
}

export function reach(points_zyx_N3, grid_HW, scale_UV, index, radius) {
  const [height, width] = grid_HW;
  const row = Math.floor(index / width);
  const column = index % width;
  const rows = Math.floor(radius * scale_UV[1]);
  const columns = Math.floor(radius * scale_UV[0]);
  const indices = [];
  const weights = [];
  for (let r = Math.max(0, row - rows); r <= Math.min(height - 1, row + rows); r++)
    for (let c = Math.max(0, column - columns); c <= Math.min(width - 1, column + columns); c++) {
      const across = ((r - row) / scale_UV[1]) ** 2 + ((c - column) / scale_UV[0]) ** 2;
      if (across >= radius * radius || !onSurface(points_zyx_N3, r * width + c)) continue;
      indices.push(r * width + c);
      weights.push((1 - across / (radius * radius)) ** 2);
    }
  return { indices: Int32Array.from(indices), weights: Float32Array.from(weights) };
}

export function smoothed(points_zyx_HW3, grid_HW, indices, weights, current_N3, strength) {
  const [height, width] = grid_HW;
  const positions = new Map(Array.from(indices, (index, position) => [index, position]));
  const after_N3 = current_N3.slice();
  for (const [position, index] of indices.entries()) {
    const row = Math.floor(index / width);
    const column = index % width;
    const mean_ZYX = [0, 0, 0];
    let count = 0;
    for (const [dr, dc] of AXES_HW) {
      const pair = [-1, 1].map((sign) => {
        const [r, c] = [row + sign * dr, column + sign * dc];
        if (r < 0 || r >= height || c < 0 || c >= width) return null;
        const at = positions.get(r * width + c);
        if (at !== undefined) return current_N3.subarray(3 * at, 3 * at + 3);
        const neighbour = r * width + c;
        return onSurface(points_zyx_HW3, neighbour) ? points_zyx_HW3.subarray(3 * neighbour, 3 * neighbour + 3) : null;
      });
      if (pair.includes(null)) continue;
      for (let axis = 0; axis < 3; axis++) mean_ZYX[axis] += (pair[0][axis] + pair[1][axis]) / 2;
      count += 1;
    }
    if (count === 0) continue;
    for (let axis = 0; axis < 3; axis++) {
      const value = current_N3[3 * position + axis];
      after_N3[3 * position + axis] = value + weights[position] * strength * (mean_ZYX[axis] / count - value);
    }
  }
  return after_N3;
}

export function grown(points_zyx_HW3, grid_HW, scale_UV, index, radius) {
  const [height, width] = grid_HW;
  const row = Math.floor(index / width);
  const column = index % width;
  const rows = Math.floor(radius * scale_UV[1]);
  const columns = Math.floor(radius * scale_UV[0]);
  let open = [];
  for (let r = Math.max(0, row - rows); r <= Math.min(height - 1, row + rows); r++)
    for (let c = Math.max(0, column - columns); c <= Math.min(width - 1, column + columns); c++) {
      const across = ((r - row) / scale_UV[1]) ** 2 + ((c - column) / scale_UV[0]) ** 2;
      if (across < radius * radius && !onSurface(points_zyx_HW3, r * width + c)) open.push(r * width + c);
    }
  const filled = new Map();
  const on = (r, c) =>
    r >= 0 &&
    r < height &&
    c >= 0 &&
    c < width &&
    (filled.has(r * width + c) || onSurface(points_zyx_HW3, r * width + c));
  const point = (cell) => filled.get(cell) ?? points_zyx_HW3.subarray(3 * cell, 3 * cell + 3);
  for (;;) {
    const ring = [];
    for (const cell of open) {
      const r = Math.floor(cell / width);
      const c = cell % width;
      const sum_ZYX = [0, 0, 0];
      let count = 0;
      for (const [dr, dc] of AXES_HW)
        for (const sign of [-1, 1]) {
          if (!on(r + sign * dr, c + sign * dc) || !on(r + 2 * sign * dr, c + 2 * sign * dc)) continue;
          const near = point((r + sign * dr) * width + c + sign * dc);
          const far = point((r + 2 * sign * dr) * width + c + 2 * sign * dc);
          for (let axis = 0; axis < 3; axis++) sum_ZYX[axis] += 2 * near[axis] - far[axis];
          count += 1;
        }
      if (count > 0) ring.push([cell, Float32Array.from(sum_ZYX, (value) => value / count)]);
    }
    if (ring.length === 0) break;
    for (const [cell, point_ZYX] of ring) filled.set(cell, point_ZYX);
    open = open.filter((cell) => !filled.has(cell));
  }
  // Across a hole of 260 voxels in Paris 4 the plate is 5.5 voxels at most from the files, where the rings alone are
  // 10.0.
  const free = [...filled.keys()].filter((cell) =>
    THIN_PLATE.every(([dr, dc]) => on(Math.floor(cell / width) + dr, (cell % width) + dc)),
  );
  const positions = new Map(free.map((cell, position) => [cell, position]));
  const columns_N13 = new Int32Array(THIN_PLATE.length * free.length).fill(-1);
  for (const [position, cell] of free.entries())
    for (const [entry, [dr, dc]] of THIN_PLATE.entries())
      columns_N13[THIN_PLATE.length * position + entry] = positions.get(cell + dr * width + dc) ?? -1;
  const weights = Float64Array.from(THIN_PLATE, ([, , weight]) => weight);
  const product = (vector, into) => {
    for (let position = 0; position < free.length; position++) {
      let sum = 0;
      for (let entry = 0; entry < THIN_PLATE.length; entry++) {
        const column = columns_N13[THIN_PLATE.length * position + entry];
        if (column !== -1) sum += weights[entry] * vector[column];
      }
      into[position] = sum;
    }
  };
  const dot = (a, b) => {
    let sum = 0;
    for (let position = 0; position < a.length; position++) sum += a[position] * b[position];
    return sum;
  };
  for (let axis = 0; axis < 3; axis++) {
    const x = Float64Array.from(free, (cell) => filled.get(cell)[axis]);
    const residual = Float64Array.from(free, (cell, position) =>
      THIN_PLATE.reduce(
        (sum, [dr, dc, weight], entry) =>
          columns_N13[THIN_PLATE.length * position + entry] === -1
            ? sum - weight * point(cell + dr * width + dc)[axis]
            : sum,
        0,
      ),
    );
    const step = new Float64Array(free.length);
    product(x, step);
    for (let position = 0; position < free.length; position++) residual[position] -= step[position];
    const direction = residual.slice();
    let squared = dot(residual, residual);
    for (let count = 0; count < THIN_PLATE_STEPS && squared > THIN_PLATE_RESIDUAL ** 2 * free.length; count++) {
      product(direction, step);
      const length = squared / dot(direction, step);
      for (let position = 0; position < free.length; position++) {
        x[position] += length * direction[position];
        residual[position] -= length * step[position];
      }
      const next = dot(residual, residual);
      for (let position = 0; position < free.length; position++)
        direction[position] = residual[position] + (next / squared) * direction[position];
      squared = next;
    }
    for (const [position, cell] of free.entries()) filled.get(cell)[axis] = x[position];
  }
  const indices = Int32Array.from(filled.keys());
  const after_N3 = new Float32Array(3 * indices.length);
  for (const [position, point_ZYX] of [...filled.values()].entries()) after_N3.set(point_ZYX, 3 * position);
  return { indices, after_N3 };
}

export function gridEdges(indices, width) {
  const positions = new Map(Array.from(indices, (index, position) => [index, position]));
  const edges_E2 = [];
  for (const [index, position] of positions) {
    const right = (index + 1) % width === 0 ? undefined : positions.get(index + 1);
    const below = positions.get(index + width);
    if (right !== undefined) edges_E2.push(position, right);
    if (below !== undefined) edges_E2.push(position, below);
  }
  return edges_E2;
}
