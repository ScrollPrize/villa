import { SURFACE_FOCUS_CANDIDATE_LIMIT, SURFACE_FOCUS_RADIUS, SURFACE_SEARCH_SLICE_MS } from "../config.js";
import { cellDistanceSquared, DISTANCE_TOLERANCE, degenerateTriangle, nearerTriangle } from "./nearest-triangle.js";
import { yieldTask } from "./yield-task.js";

const PROJECTION_LEAF_BLOCKS = 4;
const FOCUS_SAME_CANDIDATE = 1e-3;
const NEIGHBOR_STEPS_VU = [
  [0, -1],
  [0, 1],
  [-1, 0],
  [1, 0],
];

// The cell is split into two triangles along its diagonal from row, column to row + 1, column + 1.
function nearerCell(nearest, point_ZYX, points, width, row, column) {
  const p00 = (row * width + column) * 3;
  const p10 = p00 + 3;
  const p01 = p00 + width * 3;
  const p11 = p01 + 3;
  nearerTriangle(nearest, point_ZYX, points, p00, p10, p11, column, row, column + 1, row, column + 1, row + 1);
  nearerTriangle(nearest, point_ZYX, points, p00, p11, p01, column, row, column + 1, row + 1, column, row + 1);
}

function combinedBounds(bounds_B6, blocks) {
  const bounds_ZYX = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
  for (const block of blocks) {
    const offset = block * 6;
    for (let axis = 0; axis < 3; axis++) {
      bounds_ZYX[axis] = Math.min(bounds_ZYX[axis], bounds_B6[offset + axis]);
      bounds_ZYX[axis + 3] = Math.max(bounds_ZYX[axis + 3], bounds_B6[offset + axis + 3]);
    }
  }
  return bounds_ZYX;
}

function blockCentroid(bounds_B6, block, axis) {
  const offset = block * 6;
  return (bounds_B6[offset + axis] + bounds_B6[offset + axis + 3]) / 2;
}

function longestCentroidAxis(bounds_B6, blocks) {
  let selectedAxis = 0;
  let selectedSpan = -Infinity;
  for (let axis = 0; axis < 3; axis++) {
    let minimum = Infinity;
    let maximum = -Infinity;
    for (const block of blocks) {
      const centroid = blockCentroid(bounds_B6, block, axis);
      minimum = Math.min(minimum, centroid);
      maximum = Math.max(maximum, centroid);
    }
    if (maximum - minimum > selectedSpan) {
      selectedAxis = axis;
      selectedSpan = maximum - minimum;
    }
  }
  return selectedAxis;
}

function buildTree(bounds_B6, neighborBounds_B24, blocks) {
  if (blocks.length === 0) return null;
  const bounds_ZYX = combinedBounds(bounds_B6, blocks);
  const neighborBounds_D6 = new Float64Array(24);
  for (let i = 0; i < 24; i++) {
    neighborBounds_D6[i] = i % 6 < 3 ? Infinity : -Infinity;
    for (const block of blocks) {
      const value = neighborBounds_B24[block * 24 + i];
      neighborBounds_D6[i] = i % 6 < 3 ? Math.min(neighborBounds_D6[i], value) : Math.max(neighborBounds_D6[i], value);
    }
  }
  if (blocks.length <= PROJECTION_LEAF_BLOCKS)
    return { bounds_ZYX, neighborBounds_D6, blocks, left: null, right: null };
  const axis = longestCentroidAxis(bounds_B6, blocks);
  blocks.sort(
    (first, second) => blockCentroid(bounds_B6, first, axis) - blockCentroid(bounds_B6, second, axis) || first - second,
  );
  const middle = Math.floor(blocks.length / 2);
  return {
    bounds_ZYX,
    neighborBounds_D6,
    blocks: null,
    left: buildTree(bounds_B6, neighborBounds_B24, blocks.slice(0, middle)),
    right: buildTree(bounds_B6, neighborBounds_B24, blocks.slice(middle)),
  };
}

function validSurfacePoint(points, offset) {
  return Number.isFinite(points[offset]) && Number.isFinite(points[offset + 1]) && Number.isFinite(points[offset + 2]);
}

function blockRange(index, block) {
  const rowStart = Math.floor(block / index.blockColumns) * index.blockCells;
  const columnStart = (block % index.blockColumns) * index.blockCells;
  return {
    rowStart,
    rowStop: Math.min(index.H - 1, rowStart + index.blockCells),
    columnStart,
    columnStop: Math.min(index.W - 1, columnStart + index.blockCells),
  };
}

function buildNeighborBounds(points_zyx_HW3, index, validCells) {
  const bounds_B24 = new Float64Array(index.validCells_B.length * 24);
  for (let i = 0; i < bounds_B24.length; i++) bounds_B24[i] = i % 6 < 3 ? Infinity : -Infinity;
  for (let block = 0; block < index.validCells_B.length; block++) {
    if (index.validCells_B[block] === 0) continue;
    const { rowStart, rowStop, columnStart, columnStop } = blockRange(index, block);
    for (let direction = 0; direction < NEIGHBOR_STEPS_VU.length; direction++) {
      const [dv, du] = NEIGHBOR_STEPS_VU[direction];
      const offset = block * 24 + direction * 6;
      let complete = true;
      for (let v = rowStart; v < rowStop && complete; v++) {
        for (let u = columnStart; u < columnStop; u++) {
          if (!validCells[v * index.W + u]) continue;
          if (
            u + du < 0 ||
            u + du >= index.W - 1 ||
            v + dv < 0 ||
            v + dv >= index.H - 1 ||
            validCells[(v + dv) * index.W + u + du] !== 3
          ) {
            complete = false;
            break;
          }
        }
      }
      if (!complete) {
        for (let axis = 0; axis < 3; axis++) {
          bounds_B24[offset + axis] = -Infinity;
          bounds_B24[offset + axis + 3] = Infinity;
        }
        continue;
      }
      const step = (dv * index.W + du) * 3;
      for (let v = Math.max(rowStart, -dv); v <= Math.min(rowStop, index.H - 1 - dv); v++) {
        for (let u = Math.max(columnStart, -du); u <= Math.min(columnStop, index.W - 1 - du); u++) {
          const corner = (v * index.W + u) * 3;
          if (!validSurfacePoint(points_zyx_HW3, corner) || !validSurfacePoint(points_zyx_HW3, corner + step)) continue;
          for (let axis = 0; axis < 3; axis++) {
            const delta = points_zyx_HW3[corner + step + axis] - points_zyx_HW3[corner + axis];
            bounds_B24[offset + axis] = Math.min(bounds_B24[offset + axis], delta);
            bounds_B24[offset + axis + 3] = Math.max(bounds_B24[offset + axis + 3], delta);
          }
        }
      }
    }
  }
  return bounds_B24;
}

function hasCloserNeighbors(query_ZYX, bounds, offset, neighborBounds, neighborOffset) {
  for (let direction = 0; direction < NEIGHBOR_STEPS_VU.length; direction++) {
    const start = neighborOffset + direction * 6;
    if (!Number.isFinite(neighborBounds[start])) continue;
    let upper = 0;
    let magnitude = 0;
    let coordinateRounding = 0;
    for (let axis = 0; axis < 3; axis++) {
      const lo = bounds[offset + axis] - query_ZYX[axis];
      const hi = bounds[offset + axis + 3] - query_ZYX[axis];
      const deltaLo = neighborBounds[start + axis];
      const deltaHi = neighborBounds[start + axis + 3];
      const product = Math.max(lo * deltaLo, lo * deltaHi, hi * deltaLo, hi * deltaHi);
      const square = Math.max(deltaLo * deltaLo, deltaHi * deltaHi);
      upper += 2 * product + square;
      magnitude += 2 * Math.max(Math.abs(lo), Math.abs(hi)) * Math.max(Math.abs(deltaLo), Math.abs(deltaHi)) + square;
      coordinateRounding +=
        Math.max(Math.abs(bounds[offset + axis]), Math.abs(bounds[offset + axis + 3]), Math.abs(query_ZYX[axis])) *
        (Math.max(Math.abs(lo), Math.abs(hi)) + Math.max(Math.abs(deltaLo), Math.abs(deltaHi)));
    }
    // Include rounding of the absolute-coordinate triangle projection, not just this bound.
    if (upper < -DISTANCE_TOLERANCE - magnitude * 1e-12 - 64 * Number.EPSILON * coordinateRounding) return true;
  }
  return false;
}

function hasCloserNeighborCell(point_ZYX, points_zyx_HW3, validCells, grid_HW, row, column, lowerBound) {
  const width = grid_HW[1];
  for (let dv = -1; dv <= 1; dv++) {
    for (let du = -1; du <= 1; du++) {
      const v = row + dv,
        u = column + du;
      if ((du === 0 && dv === 0) || u < 0 || v < 0 || u >= width - 1 || v >= grid_HW[0] - 1) continue;
      if (!validCells[v * width + u]) continue;
      const corner = (v * width + u) * 3;
      for (const offset of [corner, corner + width * 3 + 3]) {
        let upperBound = 0;
        for (let axis = 0; axis < 3; axis++) upperBound += (points_zyx_HW3[offset + axis] - point_ZYX[axis]) ** 2;
        if (upperBound < lowerBound - DISTANCE_TOLERANCE) return true;
      }
    }
  }
  return false;
}

function boundsDistanceSquared(point_ZYX, bounds_ZYX) {
  let total = 0;
  for (let axis = 0; axis < 3; axis++) {
    const distance = Math.max(bounds_ZYX[axis] - point_ZYX[axis], 0, point_ZYX[axis] - bounds_ZYX[axis + 3]);
    total += distance * distance;
  }
  return total;
}

function pointBoundsDistanceSquared(point_ZYX, bounds_B6, offset) {
  let total = 0;
  for (let axis = 0; axis < 3; axis++) {
    const distance = Math.max(
      bounds_B6[offset + axis] - point_ZYX[axis],
      0,
      point_ZYX[axis] - bounds_B6[offset + axis + 3],
    );
    total += distance * distance;
  }
  return total;
}

function subtract(first, second) {
  return first.map((value, axis) => value - second[axis]);
}

function dot(first, second) {
  return first.reduce((total, value, axis) => total + value * second[axis], 0);
}

export class SurfaceProjector {
  constructor(surface) {
    this.surface = surface;
    this.queryStack = [];
    this.focusScratch = null;
    this.build();
  }

  build() {
    const { points_zyx_HW3, grid_HW, sourceStride, surfaceIntersectionIndex: index } = this.surface;
    const blocks = [];
    for (let block = 0; block < index.validCells_B.length; block++) {
      if (index.validCells_B[block] > 0) blocks.push(block);
    }
    this.points_zyx_HW3 = points_zyx_HW3;
    this.grid_HW = grid_HW;
    this.sourceStride = sourceStride;
    this.index = index;
    this.focusValidCells = new Uint8Array(grid_HW[0] * grid_HW[1]);
    for (let row = 0; row < grid_HW[0] - 1; row++) {
      for (let column = 0; column < grid_HW[1] - 1; column++) {
        const a = (row * grid_HW[1] + column) * 3,
          b = a + 3,
          c = a + grid_HW[1] * 3,
          d = c + 3;
        if ([a, b, c, d].every((offset) => validSurfacePoint(points_zyx_HW3, offset))) {
          this.focusValidCells[row * grid_HW[1] + column] =
            (degenerateTriangle(points_zyx_HW3, a, b, d) ? 0 : 1) |
            (degenerateTriangle(points_zyx_HW3, a, d, c) ? 0 : 2);
        }
      }
    }

    this.neighborBounds_B24 = buildNeighborBounds(points_zyx_HW3, index, this.focusValidCells);
    this.root = buildTree(index.bounds_B6, this.neighborBounds_B24, blocks);
  }

  project(point_ZYX) {
    if (this.points_zyx_HW3 !== this.surface.points_zyx_HW3) this.build();
    if (this.root === null) return null;
    const stack = this.queryStack;
    stack.length = 0;
    stack.push(this.root);
    const nearest = {
      point_ZYX: [0, 0, 0],
      preview_UV: [0, 0],
      normal_ZYX: [0, 0, 0],
      distanceSquared: Infinity,
      ambiguous: false,
    };
    while (stack.length > 0) {
      const node = stack.pop();
      if (boundsDistanceSquared(point_ZYX, node.bounds_ZYX) > nearest.distanceSquared + DISTANCE_TOLERANCE) continue;
      if (node.blocks !== null) {
        for (const block of node.blocks) {
          if (
            pointBoundsDistanceSquared(point_ZYX, this.index.bounds_B6, block * 6) >
            nearest.distanceSquared + DISTANCE_TOLERANCE
          ) {
            continue;
          }
          this.projectBlock(point_ZYX, block, nearest);
        }
        continue;
      }
      const leftDistance = boundsDistanceSquared(point_ZYX, node.left.bounds_ZYX);
      const rightDistance = boundsDistanceSquared(point_ZYX, node.right.bounds_ZYX);
      if (leftDistance <= rightDistance) {
        stack.push(node.right, node.left);
      } else {
        stack.push(node.left, node.right);
      }
    }
    if (!Number.isFinite(nearest.distanceSquared)) return null;
    return {
      surface_UV: nearest.preview_UV.map((value) => value * this.sourceStride),
      signedDepth: dot(subtract(point_ZYX, nearest.point_ZYX), nearest.normal_ZYX),
      distanceSquared: nearest.distanceSquared,
      ambiguous: nearest.ambiguous,
    };
  }

  async focusCandidates(point_ZYX, signal) {
    signal.throwIfAborted();
    if (this.points_zyx_HW3 !== this.surface.points_zyx_HW3) this.build();
    const cellCount = this.grid_HW[0] * this.grid_HW[1];
    if (this.root === null || boundsDistanceSquared(point_ZYX, this.root.bounds_ZYX) > SURFACE_FOCUS_RADIUS ** 2)
      return [];
    const scratch = this.focusScratch ?? {
      distances_N: new Float64Array(cellCount).fill(Infinity),
      touched_N: new Uint32Array(cellCount),
      minima_N: new Uint32Array(cellCount),
    };
    const { distances_N, touched_N, minima_N } = scratch;
    this.focusScratch = null;
    let touchedCount = 0;
    let minimaCount = 0;
    let scanned = 0;
    try {
      const nearest = {
        point_ZYX: [0, 0, 0],
        preview_UV: [0, 0],
        normal_ZYX: [0, 0, 0],
        distanceSquared: Infinity,
        ambiguous: false,
      };
      const { points_zyx_HW3, focusValidCells, grid_HW } = this;
      const width = grid_HW[1];
      const projectCell = (row, column) => {
        nearest.distanceSquared = Infinity;
        nearest.ambiguous = false;
        nearerCell(nearest, point_ZYX, points_zyx_HW3, width, row, column);
      };
      const evaluateCell = (row, column) => {
        const cell = row * width + column;
        if (distances_N[cell] !== Infinity || !focusValidCells[cell]) return;
        projectCell(row, column);
        distances_N[cell] = nearest.distanceSquared;
        touched_N[touchedCount++] = cell;
      };
      let deadline = performance.now() + SURFACE_SEARCH_SLICE_MS;
      let visited = 0;
      const stack = [this.root];
      while (stack.length > 0) {
        if ((++visited & 31) === 0 && performance.now() >= deadline) {
          await yieldTask(signal);
          deadline = performance.now() + SURFACE_SEARCH_SLICE_MS;
        }
        const node = stack.pop();
        if (
          boundsDistanceSquared(point_ZYX, node.bounds_ZYX) > SURFACE_FOCUS_RADIUS ** 2 ||
          hasCloserNeighbors(point_ZYX, node.bounds_ZYX, 0, node.neighborBounds_D6, 0)
        )
          continue;
        if (node.blocks === null) {
          stack.push(node.left, node.right);
          continue;
        }
        for (const block of node.blocks) {
          if (
            pointBoundsDistanceSquared(point_ZYX, this.index.bounds_B6, block * 6) > SURFACE_FOCUS_RADIUS ** 2 ||
            hasCloserNeighbors(point_ZYX, this.index.bounds_B6, block * 6, this.neighborBounds_B24, block * 24)
          )
            continue;
          const { rowStart, rowStop, columnStart, columnStop } = blockRange(this.index, block);
          for (let row = rowStart; row < rowStop; row++) {
            for (let column = columnStart; column < columnStop; column++) {
              if ((++scanned & 31) === 0 && performance.now() >= deadline) {
                await yieldTask(signal);
                deadline = performance.now() + SURFACE_SEARCH_SLICE_MS;
              }
              const a = (row * width + column) * 3,
                b = a + 3,
                c = a + width * 3,
                d = c + 3;
              if (!focusValidCells[row * width + column]) continue;
              const lowerBound = cellDistanceSquared(point_ZYX, points_zyx_HW3, a, b, c, d);
              if (lowerBound > SURFACE_FOCUS_RADIUS ** 2) continue;
              if (hasCloserNeighborCell(point_ZYX, points_zyx_HW3, focusValidCells, grid_HW, row, column, lowerBound))
                continue;
              evaluateCell(row, column);
              minima_N[minimaCount++] = row * width + column;
            }
          }
        }
      }
      const candidates = [];
      const minima = minima_N.subarray(0, minimaCount).sort();
      for (let i = 0; i < minima.length; i++) {
        if ((i & 127) === 0 && performance.now() >= deadline) {
          await yieldTask(signal);
          deadline = performance.now() + SURFACE_SEARCH_SLICE_MS;
        }
        const cell = minima[i];
        const distanceSquared = distances_N[cell];
        if (distanceSquared > SURFACE_FOCUS_RADIUS ** 2) continue;
        const row = Math.floor(cell / width),
          column = cell % width;
        let minimum = true;
        for (let dv = -1; dv <= 1 && minimum; dv++)
          for (let du = -1; du <= 1 && minimum; du++) {
            if (column + du < 0 || column + du >= width - 1 || row + dv < 0 || row + dv >= grid_HW[0] - 1) continue;
            evaluateCell(row + dv, column + du);
            const neighbor = distances_N[(row + dv) * width + column + du];
            if (neighbor < distanceSquared - DISTANCE_TOLERANCE) minimum = false;
          }
        if (!minimum) continue;
        projectCell(row, column);
        const surface_UV = nearest.preview_UV.map((value) => value * this.sourceStride);
        if (
          candidates.some(
            (candidate) =>
              Math.hypot(...candidate.surface_UV.map((value, axis) => value - surface_UV[axis])) < FOCUS_SAME_CANDIDATE,
          )
        )
          continue;
        if (candidates.length === SURFACE_FOCUS_CANDIDATE_LIMIT)
          throw new Error("Focus has too many competing surface regions to verify; Surface framing is retained.");
        candidates.push({
          surface_UV,
          signedDepth: dot(subtract(point_ZYX, nearest.point_ZYX), nearest.normal_ZYX),
          distanceSquared,
        });
      }
      return candidates.sort((a, b) => a.distanceSquared - b.distanceSquared);
    } finally {
      for (let i = 0; i < touchedCount; i++) distances_N[touched_N[i]] = Infinity;
      this.focusScratch ??= scratch;
    }
  }

  projectBlock(point_ZYX, block, nearest) {
    const points = this.points_zyx_HW3;
    const width = this.grid_HW[1];
    const { rowStart, rowStop, columnStart, columnStop } = blockRange(this.index, block);
    for (let row = rowStart; row < rowStop; row++) {
      for (let column = columnStart; column < columnStop; column++) {
        const p00 = (row * width + column) * 3;
        const p10 = p00 + 3;
        const p01 = p00 + width * 3;
        const p11 = p01 + 3;
        if (!this.focusValidCells[row * width + column]) continue;
        if (cellDistanceSquared(point_ZYX, points, p00, p10, p01, p11) > nearest.distanceSquared + DISTANCE_TOLERANCE)
          continue;
        nearerCell(nearest, point_ZYX, points, width, row, column);
      }
    }
  }
}
