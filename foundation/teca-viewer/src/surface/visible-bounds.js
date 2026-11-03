import { SURFACE_CELL_BOUNDS_LIMIT } from "../config.js";
import { ADDRESS_EXPANSION_LIMIT } from "../gpu/addresses.js";
import { SURFACE_BOUNDS_BLOCK_COLUMNS, SURFACE_BOUNDS_BLOCK_ROWS } from "./strip-atlas.js";

function includeRow(bounds_ZYX6, data_HWRGBA, width, v, firstU, lastU) {
  for (let u = firstU; u <= lastU; u++) {
    const offset = (v * width + u) * 4;
    if (data_HWRGBA[offset + 3] < 0.5) continue;
    for (let axis = 0; axis < 3; axis++) {
      bounds_ZYX6[axis] = Math.min(bounds_ZYX6[axis], data_HWRGBA[offset + axis]);
      bounds_ZYX6[axis + 3] = Math.max(bounds_ZYX6[axis + 3], data_HWRGBA[offset + axis]);
    }
  }
}

function* clippedSurfaceBounds({ stripEntries, center_UV, visibleSpan_UV, stripRows, grid_HW }) {
  if (!Number.isSafeInteger(stripRows) || stripRows < 1) {
    throw new Error("Surface strip rows must be a positive integer.");
  }
  if (
    !Array.isArray(grid_HW) ||
    grid_HW.length !== 2 ||
    grid_HW.some((size) => !Number.isSafeInteger(size) || size < 1)
  ) {
    throw new Error("Surface grid dimensions are malformed.");
  }
  const [height, width] = grid_HW;
  const entries = new Map(stripEntries.map((entry) => [entry.stripIndex, entry]));
  const minimum_UV = center_UV.map((value, axis) => Math.floor(value - visibleSpan_UV[axis] / 2));
  const maximum_UV = center_UV.map((value, axis) => Math.ceil(value + visibleSpan_UV[axis] / 2));
  const firstWindowU = Math.max(0, minimum_UV[0]);
  const lastWindowU = Math.min(width - 1, maximum_UV[0]);
  const firstWindowV = Math.max(0, minimum_UV[1]);
  const lastWindowV = Math.min(height - 1, maximum_UV[1]);
  if (
    firstWindowU <= lastWindowU &&
    firstWindowV <= lastWindowV &&
    Math.max(1, lastWindowU - firstWindowU) * Math.max(1, lastWindowV - firstWindowV) <= SURFACE_CELL_BOUNDS_LIMIT
  ) {
    for (let v = firstWindowV; v <= Math.max(firstWindowV, lastWindowV - 1); v++) {
      const upperV = Math.min(v + 1, lastWindowV);
      const lower = entries.get(Math.floor(v / stripRows));
      const upper = entries.get(Math.floor(upperV / stripRows));
      for (let u = firstWindowU; u <= Math.max(firstWindowU, lastWindowU - 1); u++) {
        const rightU = Math.min(u + 1, lastWindowU);
        const bounds_ZYX6 = new Float64Array([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity]);
        if (lower !== undefined) includeRow(bounds_ZYX6, lower.data_HWRGBA, width, v % stripRows, u, rightU);
        if (upper !== undefined) includeRow(bounds_ZYX6, upper.data_HWRGBA, width, upperV % stripRows, u, rightU);
        if (bounds_ZYX6[0] === Infinity) continue;
        const du = (u + 0.5 - center_UV[0]) / SURFACE_BOUNDS_BLOCK_COLUMNS;
        const dv = (v + 0.5 - center_UV[1]) / SURFACE_BOUNDS_BLOCK_ROWS;
        yield { bounds_ZYX6, surfacePriority: Math.hypot(du, dv) };
      }
    }
    return;
  }
  for (const entry of stripEntries) {
    const [blockRows, blockColumns] = entry.boundsGrid_HW;
    const stripStartV = entry.stripIndex * stripRows;
    const lastRow = Math.min(stripRows - 1, height - stripStartV - 1);
    if (lastRow < 0) continue;
    const next = stripStartV + stripRows < height ? entries.get(entry.stripIndex + 1) : undefined;
    const firstU = Math.max(0, minimum_UV[0]);
    const lastU = Math.min(width - 1, maximum_UV[0]);
    const firstV = Math.max(0, minimum_UV[1] - stripStartV);
    const lastV = Math.min(next === undefined ? lastRow : stripRows, maximum_UV[1] - stripStartV);
    if (firstU > lastU || firstV > lastV) continue;
    for (let blockV = Math.floor(firstV / SURFACE_BOUNDS_BLOCK_ROWS); blockV < blockRows; blockV++) {
      const startV = blockV * SURFACE_BOUNDS_BLOCK_ROWS;
      if (startV > lastV) break;
      for (let blockU = Math.floor(firstU / SURFACE_BOUNDS_BLOCK_COLUMNS); blockU < blockColumns; blockU++) {
        const startU = blockU * SURFACE_BOUNDS_BLOCK_COLUMNS;
        if (startU > lastU) break;
        const stopU = Math.min(startU + SURFACE_BOUNDS_BLOCK_COLUMNS, width - 1);
        const stopV = Math.min(startV + SURFACE_BOUNDS_BLOCK_ROWS, lastRow);
        let bounds_ZYX6;
        const fullBlock = firstU <= startU && lastU >= stopU && firstV <= startV && lastV >= stopV;
        const extendsIntoNextStrip = startV + SURFACE_BOUNDS_BLOCK_ROWS >= stripRows && lastV === stripRows;
        if (fullBlock) {
          const offset = (blockV * blockColumns + blockU) * 6;
          const stored = entry.bounds_B6.subarray(offset, offset + 6);
          bounds_ZYX6 = extendsIntoNextStrip ? new Float64Array(stored) : stored;
        } else {
          bounds_ZYX6 = new Float64Array([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity]);
          for (let v = Math.max(firstV, startV); v <= Math.min(lastV, stopV); v++) {
            includeRow(bounds_ZYX6, entry.data_HWRGBA, width, v, Math.max(firstU, startU), Math.min(lastU, stopU));
          }
        }
        // The final cell in a strip interpolates with row zero of the next strip.
        if (extendsIntoNextStrip) {
          includeRow(bounds_ZYX6, next.data_HWRGBA, width, 0, Math.max(firstU, startU), Math.min(lastU, stopU));
        }
        const du = blockU + 0.5 - center_UV[0] / SURFACE_BOUNDS_BLOCK_COLUMNS;
        const dv =
          (stripStartV + (blockV + 0.5) * SURFACE_BOUNDS_BLOCK_ROWS - center_UV[1]) / SURFACE_BOUNDS_BLOCK_ROWS;
        yield { bounds_ZYX6, surfacePriority: Math.hypot(du, dv) };
      }
    }
  }
}

export function visibleSurfaceBounds(options) {
  let cached = null;
  return {
    *[Symbol.iterator]() {
      if (cached !== null) {
        yield* cached;
        return;
      }
      let records = [];
      for (const record of clippedSurfaceBounds(options)) {
        if (records !== null) {
          if (records.length === ADDRESS_EXPANSION_LIMIT) records = null;
          else records.push(record);
        }
        yield record;
      }
      cached = records;
    },
  };
}
