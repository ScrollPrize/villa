// Pane coordinates are u = dot(p, right_ZYX) and v = dot(p, up_ZYX), both in full-resolution voxels.

import { rotatedPlaneBasis } from "../planes.js";

const SURFACE_INTERSECTION_BLOCK_CELLS = 8;

export function paneBasis(pane, rotation) {
  const { right_XYZ, up_XYZ, normal_XYZ } = rotatedPlaneBasis(pane, rotation);
  return {
    right_ZYX: [right_XYZ[2], right_XYZ[1], right_XYZ[0]],
    up_ZYX: [up_XYZ[2], up_XYZ[1], up_XYZ[0]],
    normal_ZYX: [normal_XYZ[2], normal_XYZ[1], normal_XYZ[0]],
  };
}

function setCellCornerOffsets(corners, row, column, W) {
  corners[0] = (row * W + column) * 3;
  corners[1] = (row * W + column + 1) * 3;
  corners[2] = ((row + 1) * W + column + 1) * 3;
  corners[3] = ((row + 1) * W + column) * 3;
}

function finitePoint(points_zyx_HW3, offset) {
  return (
    Number.isFinite(points_zyx_HW3[offset]) &&
    Number.isFinite(points_zyx_HW3[offset + 1]) &&
    Number.isFinite(points_zyx_HW3[offset + 2])
  );
}

export function buildSurfaceIntersectionIndex(points_zyx_HW3, H, W, blockCells = SURFACE_INTERSECTION_BLOCK_CELLS) {
  if (!(points_zyx_HW3 instanceof Float32Array) || points_zyx_HW3.length !== H * W * 3) {
    throw new Error("Surface intersection points must be a complete float32 [H,W,3] grid.");
  }
  if (!Number.isInteger(H) || H < 2 || !Number.isInteger(W) || W < 2) {
    throw new Error("Surface intersection grid dimensions must be integers of at least two.");
  }
  if (!Number.isInteger(blockCells) || blockCells < 1) {
    throw new Error("Surface intersection block size must be a positive integer.");
  }
  const blockRows = Math.ceil((H - 1) / blockCells);
  const blockColumns = Math.ceil((W - 1) / blockCells);
  const bounds_B6 = new Float32Array(blockRows * blockColumns * 6);
  const validCells_B = new Uint32Array(blockRows * blockColumns);
  const corners = [0, 0, 0, 0];
  let validCells = 0;
  for (let blockRow = 0; blockRow < blockRows; blockRow++) {
    const rowStart = blockRow * blockCells;
    const rowStop = Math.min(H - 1, rowStart + blockCells);
    for (let blockColumn = 0; blockColumn < blockColumns; blockColumn++) {
      const columnStart = blockColumn * blockCells;
      const columnStop = Math.min(W - 1, columnStart + blockCells);
      const block = blockRow * blockColumns + blockColumn;
      const bound = block * 6;
      let minZ = Infinity;
      let minY = Infinity;
      let minX = Infinity;
      let maxZ = -Infinity;
      let maxY = -Infinity;
      let maxX = -Infinity;
      let blockValidCells = 0;
      for (let row = rowStart; row < rowStop; row++) {
        for (let column = columnStart; column < columnStop; column++) {
          setCellCornerOffsets(corners, row, column, W);
          if (!corners.every((offset) => finitePoint(points_zyx_HW3, offset))) continue;
          blockValidCells += 1;
          for (const offset of corners) {
            const z = points_zyx_HW3[offset];
            const y = points_zyx_HW3[offset + 1];
            const x = points_zyx_HW3[offset + 2];
            minZ = Math.min(minZ, z);
            minY = Math.min(minY, y);
            minX = Math.min(minX, x);
            maxZ = Math.max(maxZ, z);
            maxY = Math.max(maxY, y);
            maxX = Math.max(maxX, x);
          }
        }
      }
      validCells_B[block] = blockValidCells;
      validCells += blockValidCells;
      if (blockValidCells > 0) {
        bounds_B6.set([minZ, minY, minX, maxZ, maxY, maxX], bound);
      }
    }
  }
  return {
    H,
    W,
    blockCells,
    blockRows,
    blockColumns,
    bounds_B6,
    validCells_B,
    validCells,
  };
}

function blockCrossesPlane(bounds_B6, offset, normal_ZYX, iso) {
  const [nz, ny, nx] = normal_ZYX;
  const minProjection =
    (nz < 0 ? bounds_B6[offset + 3] : bounds_B6[offset]) * nz +
    (ny < 0 ? bounds_B6[offset + 4] : bounds_B6[offset + 1]) * ny +
    (nx < 0 ? bounds_B6[offset + 5] : bounds_B6[offset + 2]) * nx;
  const maxProjection =
    (nz < 0 ? bounds_B6[offset] : bounds_B6[offset + 3]) * nz +
    (ny < 0 ? bounds_B6[offset + 1] : bounds_B6[offset + 4]) * ny +
    (nx < 0 ? bounds_B6[offset + 2] : bounds_B6[offset + 5]) * nx;
  const tolerance = Math.max(1, Math.abs(iso), Math.abs(minProjection), Math.abs(maxProjection)) * 1e-7;
  return iso >= minProjection - tolerance && iso <= maxProjection + tolerance;
}

export function surfaceIntersectionSegments(
  points_zyx_HW3,
  H,
  W,
  basis,
  iso,
  surfaceIndex = buildSurfaceIntersectionIndex(points_zyx_HW3, H, W),
  evidence = null,
  excluded = null,
) {
  if (surfaceIndex.H !== H || surfaceIndex.W !== W) {
    throw new Error("Surface intersection index does not match its point grid.");
  }
  const [nz, ny, nx] = basis.normal_ZYX;
  const [rz, ry, rx] = basis.right_ZYX;
  const [uz, uy, ux] = basis.up_ZYX;
  const segments_uv = [];
  const crossings_uv = [];
  const corners = [0, 0, 0, 0];
  const signed = [0, 0, 0, 0];
  const crossingBlocks_B = new Uint8Array(surfaceIndex.blockRows * surfaceIndex.blockColumns);
  let intersectedBlocks = 0;
  let testedCells = 0;
  for (let blockRow = 0; blockRow < surfaceIndex.blockRows; blockRow++) {
    for (let blockColumn = 0; blockColumn < surfaceIndex.blockColumns; blockColumn++) {
      const block = blockRow * surfaceIndex.blockColumns + blockColumn;
      if (surfaceIndex.validCells_B[block] === 0) continue;
      if (!blockCrossesPlane(surfaceIndex.bounds_B6, block * 6, basis.normal_ZYX, iso)) continue;
      crossingBlocks_B[block] = 1;
      intersectedBlocks += 1;
    }
  }
  for (let blockRow = 0; blockRow < surfaceIndex.blockRows; blockRow++) {
    const rowStart = blockRow * surfaceIndex.blockCells;
    const rowStop = Math.min(H - 1, rowStart + surfaceIndex.blockCells);
    for (let row = rowStart; row < rowStop; row++) {
      for (let blockColumn = 0; blockColumn < surfaceIndex.blockColumns; blockColumn++) {
        const block = blockRow * surfaceIndex.blockColumns + blockColumn;
        if (crossingBlocks_B[block] === 0) continue;
        const columnStart = blockColumn * surfaceIndex.blockCells;
        const columnStop = Math.min(W - 1, columnStart + surfaceIndex.blockCells);
        for (let col = columnStart; col < columnStop; col++) {
          if (
            excluded?.blocks.has(
              Math.floor((row * excluded.stride) / excluded.blockCells) * excluded.columns +
                Math.floor((col * excluded.stride) / excluded.blockCells),
            )
          )
            continue;
          testedCells += 1;
          setCellCornerOffsets(corners, row, col, W);
          let finite = 0;
          let above = 0;
          let below = 0;
          for (; finite < 4 && finitePoint(points_zyx_HW3, corners[finite]); finite++) {
            const offset = corners[finite];
            signed[finite] =
              points_zyx_HW3[offset] * nz + points_zyx_HW3[offset + 1] * ny + points_zyx_HW3[offset + 2] * nx - iso;
            if (signed[finite] > 0) above += 1;
            if (signed[finite] < 0) below += 1;
          }
          if (finite < 4 || above === 4 || below === 4) continue;
          crossings_uv.length = 0;
          for (let edge = 0; edge < 4; edge++) {
            const a = corners[edge];
            const b = corners[(edge + 1) % 4];
            const sa = signed[edge];
            const sb = signed[(edge + 1) % 4];
            if (sa * sb > 0 || (sa === 0 && sb === 0)) continue;
            const t = sa === sb ? 0.5 : sa / (sa - sb);
            const z = points_zyx_HW3[a] + (points_zyx_HW3[b] - points_zyx_HW3[a]) * t;
            const y = points_zyx_HW3[a + 1] + (points_zyx_HW3[b + 1] - points_zyx_HW3[a + 1]) * t;
            const x = points_zyx_HW3[a + 2] + (points_zyx_HW3[b + 2] - points_zyx_HW3[a + 2]) * t;
            crossings_uv.push(z * rz + y * ry + x * rx, z * uz + y * uy + x * ux);
          }
          for (let pair = 0; pair + 3 < crossings_uv.length; pair += 4) {
            segments_uv.push(
              crossings_uv[pair],
              crossings_uv[pair + 1],
              crossings_uv[pair + 2],
              crossings_uv[pair + 3],
            );
          }
        }
      }
    }
  }
  if (evidence !== null) {
    evidence.totalBlocks = surfaceIndex.blockRows * surfaceIndex.blockColumns;
    evidence.intersectedBlocks = intersectedBlocks;
    evidence.totalValidCells = surfaceIndex.validCells;
    evidence.testedCells = testedCells;
    evidence.segments = segments_uv.length / 4;
  }
  return segments_uv;
}
