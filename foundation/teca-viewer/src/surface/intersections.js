import {
  SURFACE_CUT_DELAY_MS,
  SURFACE_CUT_DEPTH,
  SURFACE_CUT_REACH,
  SURFACE_INTERSECTION_CACHE_BYTES,
  SURFACE_INTERSECTION_CELL_LIMIT,
} from "../config.js";
import { buildSurfaceIntersectionIndex, surfaceIntersectionSegments } from "./overlay-geometry.js";
import { yieldTask } from "./yield-task.js";

// Primary voxels; matches the projected outline cache precision.
const PLANE_OFFSET_TOLERANCE = 1e-5;

function projectSurfaceIntersections(geometry, registration, pane) {
  const basis = Object.fromEntries(
    Object.entries(pane.basis).map(([key, direction]) => [
      key,
      direction.map((value, axis) => value * registration.scale_ZYX[axis]),
    ]),
  );
  const iso = pane.basis.normal_ZYX.reduce(
    (sum, value, axis) => sum + value * (pane.center_ZYX[axis] - registration.offset_ZYX[axis]),
    0,
  );
  const offset_UV = [pane.basis.right_ZYX, pane.basis.up_ZYX].map((direction) =>
    direction.reduce((sum, value, axis) => sum + value * registration.offset_ZYX[axis], 0),
  );
  const segments_uv = [];
  let testedCells = 0;
  for (const tile of geometry.tiles) {
    const evidence = {};
    const segments = surfaceIntersectionSegments(
      tile.points_zyx_HW3,
      ...tile.grid_HW,
      basis,
      iso,
      tile.index,
      evidence,
    );
    testedCells += evidence.testedCells;
    for (let i = 0; i < segments.length; i++) segments_uv.push(segments[i] + offset_UV[i % 2]);
  }
  return {
    geometry,
    segments_uv,
    evidence: {
      stride: geometry.stride,
      geometryRevision: geometry.revision,
      retainedCells: geometry.cells,
      retainedBytes: geometry.bytes,
      testedCells,
      intersectedBlocks: geometry.tiles.length,
      segments: segments_uv.length / 4,
    },
  };
}

function surfaceIntersectionPlan(source, pyramid, registration, pane) {
  const bounds = pyramid.intersectionBounds_B6;
  const blockCells = pyramid.intersectionBlockCells;
  const blocks = [];
  const directions = [pane.basis.normal_ZYX, pane.basis.right_ZYX, pane.basis.up_ZYX];
  const limits = [
    Math.min(...pane.halfSpan_UV) * SURFACE_CUT_DEPTH,
    ...pane.halfSpan_UV.map((value) => value * SURFACE_CUT_REACH),
  ];
  for (let block = 0; block < bounds.length / 6; block++) {
    if (!Number.isFinite(bounds[block * 6])) continue;
    let visible = true;
    for (let direction = 0; direction < 3; direction++) {
      let middle = 0,
        extent = 0;
      for (let axis = 0; axis < 3; axis++) {
        const low = bounds[block * 6 + axis],
          high = bounds[block * 6 + axis + 3];
        middle +=
          (((low + high) / 2) * registration.scale_ZYX[axis] + registration.offset_ZYX[axis] - pane.center_ZYX[axis]) *
          directions[direction][axis];
        extent += ((high - low) / 2) * Math.abs(registration.scale_ZYX[axis] * directions[direction][axis]);
      }
      if (Math.abs(middle) > extent + limits[direction] + 1e-4) {
        visible = false;
        break;
      }
    }
    if (visible) blocks.push(block);
  }
  const spacing = Math.max(...registration.scale_ZYX.map(Math.abs)) / Math.min(...source.scale_UV);
  let stride = 1;
  while (stride * 2 <= blockCells && stride * 2 * spacing <= pane.voxelsPerPixel) stride *= 2;
  while (blocks.length * (blockCells / stride) ** 2 > SURFACE_INTERSECTION_CELL_LIMIT && stride < blockCells)
    stride *= 2;
  if (blocks.length > SURFACE_INTERSECTION_CELL_LIMIT)
    throw new Error(`Mesh detail budget needs at least ${blocks.length} cells.`);
  return { blocks, stride, cells: blocks.length * (blockCells / stride) ** 2 };
}

export class SurfaceIntersections {
  constructor({ isInteracting }) {
    this.isInteracting = isInteracting;
    this.tiles = new Map();
    this.sourceIds = new WeakMap();
    this.nextSourceId = 0;
    this.bytes = 0;
    this.requests = new Map();
    this.timer = null;
    this.running = false;
    this.version = 0;
    this.geometryRevision = 0;
  }

  request(key, demand, changed) {
    const previous = this.requests.get(key);
    const same = previous?.image === demand.image && previous.revision === demand.pyramid.revision;
    if (same && previous.signature === demand.signature) return previous;
    const planeOffset = demand.pane.center_ZYX.reduce(
      (sum, value, axis) => sum + value * demand.pane.basis.normal_ZYX[axis],
      0,
    );
    // In-plane translations can round the dot product differently at large CT coordinates.
    const planeTolerance = Math.max(
      PLANE_OFFSET_TOLERANCE,
      64 * Number.EPSILON * Math.max(1, ...demand.pane.center_ZYX.map(Math.abs)),
    );
    const samePlane =
      same &&
      previous.planeSignature === demand.planeSignature &&
      Math.abs(previous.planeOffset - planeOffset) <= planeTolerance;
    const geometry = same ? previous.result?.geometry : null;
    previous?.controller.abort();
    const request = {
      ...demand,
      source: demand.image.source,
      revision: demand.pyramid.revision,
      planeOffset: samePlane ? previous.planeOffset : planeOffset,
      changed,
      controller: new AbortController(),
      pending: true,
      result: samePlane
        ? previous.result
        : geometry
          ? projectSurfaceIntersections(geometry, demand.registration, demand.pane)
          : null,
      error: null,
      published: false,
      version: samePlane ? previous.version : ++this.version,
    };
    this.requests.set(key, request);
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    this.schedule();
    return request;
  }

  retain(keys) {
    for (const [key, request] of this.requests) {
      if (keys.has(key)) continue;
      request.controller.abort();
      this.requests.delete(key);
    }
    const sources = new Set([...this.requests.values()].map((request) => request.source));
    for (const [key, tile] of this.tiles) {
      if (sources.has(tile.source)) continue;
      this.bytes -= tile.bytes;
      this.tiles.delete(key);
    }
    if (this.requests.size === 0 && this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  cutting(keys) {
    return [...keys].filter((key) => this.requests.get(key)?.published === false).length;
  }

  schedule() {
    if (this.running || this.timer !== null) return;
    this.timer = setTimeout(() => {
      this.timer = null;
      void this.run();
    }, SURFACE_CUT_DELAY_MS);
  }

  async run() {
    if (this.isInteracting()) {
      this.schedule();
      return;
    }
    this.running = true;
    try {
      for (const [key, request] of this.requests) {
        if (!request.pending) continue;
        request.pending = false;
        try {
          const result = request.prepared ?? (await this.refine(request, request.controller.signal));
          if (this.requests.get(key) !== request || request.pyramid.revision !== request.revision) continue;
          if (this.isInteracting()) {
            request.prepared = result;
            request.pending = true;
            break;
          }
          request.prepared = null;
          request.result = result;
          request.published = true;
          request.version = ++this.version;
          request.changed();
        } catch (error) {
          if (request.controller.signal.aborted) continue;
          if (this.requests.get(key) !== request) continue;
          request.error = error.message;
          request.published = true;
          request.changed();
        }
      }
    } finally {
      this.running = false;
      if ([...this.requests.values()].some((request) => request.pending)) this.schedule();
    }
  }

  async refine({ image, pyramid, registration, pane }, signal) {
    const source = image.source;
    const plan = surfaceIntersectionPlan(source, pyramid, registration, pane);
    let lastYield = performance.now();
    const tiles = [];
    for (const block of plan.blocks) {
      signal.throwIfAborted();
      tiles.push(this.tile(pyramid, source, block, plan.stride));
      if (performance.now() - lastYield > 8) {
        await yieldTask(signal);
        lastYield = performance.now();
      }
    }
    const missing = tiles.filter((tile) => !tile.complete);
    if (plan.stride === 1) {
      const strips = new Map();
      for (const tile of missing) {
        for (let row = 0; row < tile.grid_HW[0]; row++) {
          const v = tile.start_UV[1] + row;
          const index = Math.floor(v / source.stripRows);
          if (!strips.has(index)) strips.set(index, []);
          strips.get(index).push({ tile, row, v });
        }
        if (performance.now() - lastYield > 8) {
          await yieldTask(signal);
          lastYield = performance.now();
        }
      }
      for (const [index, rows] of strips) {
        signal.throwIfAborted();
        const data = image.atlas?.getPresented(index)?.data_HWRGBA ?? (await source.getStrip(index, signal));
        signal.throwIfAborted();
        for (const { tile, row, v } of rows) {
          for (let column = 0; column < tile.grid_HW[1]; column++) {
            const pixel = (v % source.stripRows) * source.grid_HW[1] + tile.start_UV[0] + column;
            for (let axis = 0; axis < 3; axis++)
              tile.points_zyx_HW3[(row * tile.grid_HW[1] + column) * 3 + axis] =
                data[pixel * 4 + 3] < 0.5 ? NaN : data[pixel * 4 + axis];
          }
        }
        await yieldTask(signal);
      }
    }
    for (const tile of missing) {
      this.admit(tile);
      if (performance.now() - lastYield > 8) {
        await yieldTask(signal);
        lastYield = performance.now();
      }
    }
    signal.throwIfAborted();
    const geometry = {
      revision: ++this.geometryRevision,
      tiles,
      stride: plan.stride,
      cells: plan.cells,
      bytes: tiles.reduce((sum, tile) => sum + tile.bytes, 0),
      coverage: {
        blocks: new Set(plan.blocks),
        blockCells: pyramid.intersectionBlockCells,
        columns: pyramid.intersectionGrid_HW[1],
      },
    };
    return projectSurfaceIntersections(geometry, registration, pane);
  }

  tile(pyramid, source, block, stride) {
    if (!this.sourceIds.has(source)) this.sourceIds.set(source, ++this.nextSourceId);
    const key = `${this.sourceIds.get(source)}:${pyramid.revision}:${block}:${stride}`;
    const cached = this.tiles.get(key);
    if (cached !== undefined) {
      this.tiles.delete(key);
      this.tiles.set(key, cached);
      return cached;
    }
    const size = pyramid.intersectionBlockCells;
    const columns = pyramid.intersectionGrid_HW[1];
    const start_UV = [(block % columns) * size, Math.floor(block / columns) * size];
    const stop_UV = start_UV.map((value, axis) => Math.min(source.grid_HW[1 - axis] - 1, value + size));
    const grid_HW = stop_UV.map((value, axis) => Math.ceil((value - start_UV[axis]) / stride) + 1).reverse();
    const points_zyx_HW3 = new Float32Array(grid_HW[0] * grid_HW[1] * 3);
    const level = stride === 1 ? null : pyramid.levels[Math.log2(stride) - 1];
    for (let row = 0; row < grid_HW[0]; row++) {
      const v = Math.min(stop_UV[1], start_UV[1] + row * stride);
      if (level === null) break;
      const data = level.data_HWRGBA;
      for (let column = 0; column < grid_HW[1]; column++) {
        const u = Math.min(stop_UV[0], start_UV[0] + column * stride);
        const pixel = Math.ceil(v / stride) * level.grid_HW[1] + Math.ceil(u / stride);
        for (let axis = 0; axis < 3; axis++)
          points_zyx_HW3[(row * grid_HW[1] + column) * 3 + axis] =
            data[pixel * 4 + 3] < 0.5 ? NaN : data[pixel * 4 + axis];
      }
    }
    return { key, source, grid_HW, points_zyx_HW3, start_UV, complete: false };
  }

  admit(tile) {
    tile.index = buildSurfaceIntersectionIndex(tile.points_zyx_HW3, ...tile.grid_HW);
    tile.bytes = tile.points_zyx_HW3.byteLength + tile.index.bounds_B6.byteLength + tile.index.validCells_B.byteLength;
    tile.complete = true;
    const { key } = tile;
    if (tile.bytes <= SURFACE_INTERSECTION_CACHE_BYTES) {
      while (this.bytes + tile.bytes > SURFACE_INTERSECTION_CACHE_BYTES) {
        const oldest = this.tiles.keys().next().value;
        this.bytes -= this.tiles.get(oldest).bytes;
        this.tiles.delete(oldest);
      }
      this.tiles.set(key, tile);
      this.bytes += tile.bytes;
    }
    return tile;
  }
}
