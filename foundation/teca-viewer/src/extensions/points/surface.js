const SURFACE_MARGIN = 1.5;
const PATCH = 16;
const BINS_A_PATCH = 8;
const QUARTERS = [
  [1, 1],
  [1, -1],
  [-1, 1],
  [-1, -1],
];

export class SurfaceCells {
  constructor({ grid_HW, points_zyx_HW3 }) {
    this.grid_HW = grid_HW;
    this.points_zyx_HW3 = points_zyx_HW3;
    this.patches_HW = grid_HW.map((size) => Math.ceil(size / PATCH));
    this.boxes = new Float32Array(this.patches_HW[0] * this.patches_HW[1] * 6);
    for (let patch = 0; patch < this.boxes.length; patch += 6) {
      this.boxes.fill(Infinity, patch, patch + 3);
      this.boxes.fill(-Infinity, patch + 3, patch + 6);
    }
    this.edges = new Float32Array(this.patches_HW[0] * this.patches_HW[1]);
    for (let row = 0; row < grid_HW[0]; row++) {
      const lastRow = Math.floor(row / PATCH);
      const firstRow = row > 0 && row % PATCH === 0 ? lastRow - 1 : lastRow;
      for (let column = 0; column < grid_HW[1]; column++) {
        const cell = row * grid_HW[1] + column;
        if (!this.on(cell)) continue;
        const lastColumn = Math.floor(column / PATCH);
        const firstColumn = column > 0 && column % PATCH === 0 ? lastColumn - 1 : lastColumn;
        let edge = 0;
        for (const next of [row + 1 < grid_HW[0] ? cell + grid_HW[1] : -1, column + 1 < grid_HW[1] ? cell + 1 : -1])
          if (next >= 0 && this.on(next))
            edge = Math.max(
              edge,
              Math.hypot(...[0, 1, 2].map((axis) => points_zyx_HW3[3 * next + axis] - points_zyx_HW3[3 * cell + axis])),
            );
        for (let patchRow = firstRow; patchRow <= lastRow; patchRow++)
          for (let patchColumn = firstColumn; patchColumn <= lastColumn; patchColumn++) {
            const patch = patchRow * this.patches_HW[1] + patchColumn;
            for (let axis = 0; axis < 3; axis++) {
              const value = points_zyx_HW3[3 * cell + axis];
              if (value < this.boxes[6 * patch + axis]) this.boxes[6 * patch + axis] = value;
              if (value > this.boxes[6 * patch + 3 + axis]) this.boxes[6 * patch + 3 + axis] = value;
            }
            this.edges[patch] = Math.max(this.edges[patch], edge);
          }
      }
    }
    this.index();
  }

  index() {
    const held = [];
    const sides = [];
    this.low_ZYX = [Infinity, Infinity, Infinity];
    const high_ZYX = [-Infinity, -Infinity, -Infinity];
    for (let patch = 0; patch < this.boxes.length / 6; patch++) {
      const box = patch * 6;
      if (this.boxes[box] > this.boxes[box + 3]) continue;
      held.push(patch);
      let side = 0;
      for (let axis = 0; axis < 3; axis++) {
        side = Math.max(side, this.boxes[box + 3 + axis] - this.boxes[box + axis]);
        this.low_ZYX[axis] = Math.min(this.low_ZYX[axis], this.boxes[box + axis]);
        high_ZYX[axis] = Math.max(high_ZYX[axis], this.boxes[box + 3 + axis]);
      }
      sides.push(side);
    }
    sides.sort((a, b) => a - b);
    this.side = Math.max(1, sides[sides.length >> 1] ?? 1);
    const binsAlong = () => high_ZYX.map((high, axis) => (held.length === 0 ? 0 : this.binOf(high, axis) + 1));
    while (binsAlong().reduce((a, b) => a * b) > BINS_A_PATCH * held.length) this.side *= 2;
    this.bins_ZYX = binsAlong();
    const binsOf = (patch, visit) => {
      const box = patch * 6;
      const from_ZYX = [0, 1, 2].map((axis) => this.binOf(this.boxes[box + axis], axis));
      const to_ZYX = [0, 1, 2].map((axis) => this.binOf(this.boxes[box + 3 + axis], axis));
      this.visitBins(from_ZYX, to_ZYX, visit);
    };
    this.binStarts = new Int32Array(this.bins_ZYX[0] * this.bins_ZYX[1] * this.bins_ZYX[2] + 1);
    for (const patch of held) binsOf(patch, (bin) => this.binStarts[bin + 1]++);
    for (let bin = 1; bin < this.binStarts.length; bin++) this.binStarts[bin] += this.binStarts[bin - 1];
    this.binPatches = new Int32Array(this.binStarts.at(-1));
    const filled = this.binStarts.slice(0, -1);
    for (const patch of held) binsOf(patch, (bin) => (this.binPatches[filled[bin]++] = patch));
    this.looked = new Float64Array(this.boxes.length / 6);
    this.looks = 0;
  }

  binOf(value, axis) {
    return Math.floor((value - this.low_ZYX[axis]) / this.side);
  }

  visitBins(from_ZYX, to_ZYX, visit) {
    const [, H, W] = this.bins_ZYX;
    const [z0, y0, x0] = from_ZYX.map((value) => Math.max(0, value));
    const [z1, y1, x1] = to_ZYX.map((value, axis) => Math.min(this.bins_ZYX[axis] - 1, value));
    for (let z = z0; z <= z1; z++)
      for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) visit((z * H + y) * W + x);
  }

  on(cell) {
    const z = this.points_zyx_HW3[3 * cell];
    const y = this.points_zyx_HW3[3 * cell + 1];
    const x = this.points_zyx_HW3[3 * cell + 2];
    return Number.isFinite(z) && Number.isFinite(y) && Number.isFinite(x) && !(z === -1 && y === -1 && x === -1);
  }

  nearest(point_ZYX, depth) {
    const [H, W] = this.grid_HW;
    const points = this.points_zyx_HW3;
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let nearest = null;
    let best = Infinity;
    const look = ++this.looks;
    const from_ZYX = point_ZYX.map((value, axis) => this.binOf(value - depth, axis));
    const to_ZYX = point_ZYX.map((value, axis) => this.binOf(value + depth, axis));
    this.visitBins(from_ZYX, to_ZYX, (bin) => {
      for (let at = this.binStarts[bin]; at < this.binStarts[bin + 1]; at++) {
        const patch = this.binPatches[at];
        if (this.looked[patch] === look) continue;
        this.looked[patch] = look;
        let holds = true;
        for (let axis = 0; axis < 3 && holds; axis++)
          holds =
            point_ZYX[axis] >= this.boxes[6 * patch + axis] - depth &&
            point_ZYX[axis] <= this.boxes[6 * patch + 3 + axis] + depth;
        if (!holds) continue;
        const corner = (depth + this.edges[patch]) ** 2;
        const [row0, column0] = [Math.floor(patch / this.patches_HW[1]) * PATCH, (patch % this.patches_HW[1]) * PATCH];
        for (let row = row0; row <= Math.min(H - 1, row0 + PATCH); row++)
          for (let column = column0; column <= Math.min(W - 1, column0 + PATCH); column++) {
            const vertex = row * W + column;
            if (!this.on(vertex)) continue;
            const d = [0, 1, 2].map((axis) => point_ZYX[axis] - points[3 * vertex + axis]);
            const distance = dot(d, d);
            if (distance > corner) continue;
            if (distance < best) [nearest, best] = [[row, column], distance];
            for (const [dr, dc] of QUARTERS) {
              const [down, across, opposite] = [
                [row + dr, column],
                [row, column + dc],
                [row + dr, column + dc],
              ].map(([r, c]) => (r >= 0 && r < H && c >= 0 && c < W && this.on(r * W + c) ? r * W + c : -1));
              if (down < 0 || across < 0 || opposite < 0) continue;
              const e1 = [0, 1, 2].map((axis) => points[3 * down + axis] - points[3 * vertex + axis]);
              const e2 = [0, 1, 2].map((axis) => points[3 * across + axis] - points[3 * vertex + axis]);
              const [a11, a12, a22, b1, b2] = [dot(e1, e1), dot(e1, e2), dot(e2, e2), dot(e1, d), dot(e2, d)];
              const determinant = a11 * a22 - a12 * a12;
              if (determinant <= 0) continue;
              const s = Math.min(1, Math.max(0, (b1 * a22 - b2 * a12) / determinant));
              const t = Math.min(1, Math.max(0, (b2 * a11 - b1 * a12) / determinant));
              const off = [0, 1, 2].reduce((total, axis) => total + (d[axis] - s * e1[axis] - t * e2[axis]) ** 2, 0);
              if (off < best) [nearest, best] = [[row + s * dr, column + t * dc], off];
            }
          }
      }
    });
    return best <= depth * depth ? nearest : null;
  }
}

function placePiece(cells, { marks, ids_N }, depth, found) {
  const { points_zyx_N3, edges_E2 } = marks;
  const neighbours = marks.neighbours_N2 ?? null;
  const drawn_N = new Int32Array(ids_N.length).fill(-1);
  const cells_N2 = [];
  const from_N = [];
  for (let index = 0; index < ids_N.length; index++) {
    const [z, y, x] = [points_zyx_N3[3 * index], points_zyx_N3[3 * index + 1], points_zyx_N3[3 * index + 2]];
    let place = found.get(ids_N[index]);
    if (place === undefined || place.z !== z || place.y !== y || place.x !== x || place.depth !== depth) {
      place = { z, y, x, depth, cell: cells.nearest([z, y, x], depth) };
      found.set(ids_N[index], place);
    }
    if (place.cell === null) continue;
    drawn_N[index] = from_N.length;
    from_N.push(index);
    cells_N2.push(place.cell[0], place.cell[1]);
  }
  const edges = [];
  for (let edge = 0; edge < edges_E2.length; edge += 2) {
    const [a, b] = [drawn_N[edges_E2[edge]], drawn_N[edges_E2[edge + 1]]];
    if (a >= 0 && b >= 0) edges.push(a, b);
  }
  const neighbours_N2 = neighbours === null ? null : new Int32Array(2 * from_N.length).fill(-1);
  for (let point = 0; point < (neighbours_N2?.length ?? 0); point++) {
    const other = neighbours[2 * from_N[point >> 1] + (point & 1)];
    if (other >= 0) neighbours_N2[point] = drawn_N[other];
  }
  return {
    depth,
    cells_N2: Float32Array.from(cells_N2),
    from_N: Int32Array.from(from_N),
    edges_E2: Uint32Array.from(edges),
    neighbours_N2,
  };
}

function surfaceMarks(url, cells, layer, depth, found, placed) {
  const { marks, ids_N } = layer;
  const starts = layer.starts ?? [0];
  const pieces = (layer.pieces ?? [layer]).map((piece) => {
    if (placed.get(piece)?.depth !== depth) placed.set(piece, placePiece(cells, piece, depth, found));
    return placed.get(piece);
  });
  let [count, ends] = [0, 0];
  for (const piece of pieces) [count, ends] = [count + piece.from_N.length, ends + piece.edges_E2.length];
  const cells_N2 = new Float32Array(2 * count);
  const colors_N3 = new Float32Array(3 * count);
  const from_N = new Int32Array(count);
  const edges_E2 = new Uint32Array(ends);
  const crowding = pieces.some((piece) => piece.neighbours_N2 !== null);
  const neighbours_N2 = crowding ? new Int32Array(2 * count).fill(-1) : null;
  let [at, end] = [0, 0];
  for (const [index, piece] of pieces.entries()) {
    cells_N2.set(piece.cells_N2, 2 * at);
    for (let point = 0; point < piece.from_N.length; point++) {
      const from = starts[index] + piece.from_N[point];
      from_N[at + point] = from;
      for (let channel = 0; channel < 3; channel++)
        colors_N3[3 * (at + point) + channel] = marks.colors_N3[3 * from + channel];
    }
    for (let edge = 0; edge < piece.edges_E2.length; edge++) edges_E2[end + edge] = piece.edges_E2[edge] + at;
    for (let value = 0; value < (piece.neighbours_N2?.length ?? 0); value++)
      if (piece.neighbours_N2[value] >= 0) neighbours_N2[2 * at + value] = piece.neighbours_N2[value] + at;
    [at, end] = [at + piece.from_N.length, end + piece.edges_E2.length];
  }
  if (found.size > 2 * ids_N.length) {
    const drawn = new Set(ids_N);
    for (const id of found.keys()) if (!drawn.has(id)) found.delete(id);
  }
  const selected = marks.selected < 0 ? -1 : from_N.indexOf(marks.selected);
  return { marks: { surface: url, cells_N2, colors_N3, edges_E2, neighbours_N2, selected }, from_N };
}

export class SurfaceNodes {
  constructor(viewer, failed, options) {
    this.viewer = viewer;
    this.failed = failed;
    this.options = options;
    this.surfaces = new Map();
    this.layers = null;
    this.shown = null;
    this.depth = null;
  }

  drop(url) {
    const surface = this.surfaces.get(url);
    surface.controller.abort();
    for (const layer of surface.layers) layer.handle.remove();
    this.surfaces.delete(url);
  }

  volumeChanged() {
    for (const url of [...this.surfaces.keys()]) this.drop(url);
    this.layers = null;
    this.shown = null;
  }

  changed() {
    const listed = this.viewer.listedSurfaces();
    for (const url of [...this.surfaces.keys()]) if (!listed.includes(url)) this.drop(url);
    if (this.layers !== null) this.draw(this.layers, this.shown, this.depth);
  }

  read() {
    if (this.layers === null || this.layers.every(({ ids_N }) => ids_N.length === 0)) return;
    for (const url of this.viewer.surfaces()) {
      if (this.surfaces.has(url)) continue;
      const controller = new AbortController();
      const layers = this.options.map((options) => ({
        handle: this.viewer.addSurfaceMarks(options),
        found: new Map(),
        placed: new WeakMap(),
        drawn: null,
        depth: null,
        from_N: null,
        cells_N2: null,
        shown: null,
      }));
      const surface = { cells: null, layers, controller };
      this.surfaces.set(url, surface);
      this.viewer
        .readSurface(url, controller.signal)
        .then((read) => {
          surface.cells = new SurfaceCells(read);
          this.draw(this.layers, this.shown, this.depth);
        })
        .catch((error) => {
          if (error.name === "AbortError") return;
          if (this.surfaces.get(url) === surface) this.drop(url);
          this.failed(error);
        });
    }
  }

  near(point_ZYX) {
    const cells = this.surfaces.get(this.viewer.surfaceView()?.surface)?.cells ?? null;
    return cells !== null && cells.nearest(point_ZYX, this.depth + SURFACE_MARGIN) !== null;
  }

  nearest(cell_HW, reach) {
    const view = this.viewer.surfaceView();
    const [u, v] = view.scale_UV;
    let nearest = null;
    let best = reach * reach;
    for (const layer of this.surfaces.get(view.surface)?.layers ?? []) {
      for (let point = 0; point < (layer.from_N?.length ?? 0); point++) {
        const from = layer.from_N[point];
        if (layer.shown !== null && layer.shown[from] === 0) continue;
        const rows = (layer.cells_N2[2 * point] - cell_HW[0]) / v;
        const columns = (layer.cells_N2[2 * point + 1] - cell_HW[1]) / u;
        const distance = rows * rows + columns * columns;
        if (distance > best || (distance === best && nearest !== null)) continue;
        best = distance;
        nearest = layer.drawn.ids_N[from];
      }
    }
    return nearest;
  }

  draw(layers, shown, depth) {
    [this.layers, this.shown, this.depth] = [layers, shown, depth];
    this.read();
    const listed = this.viewer.surfaces();
    for (const [url, surface] of this.surfaces) {
      if (surface.cells === null || !listed.includes(url)) continue;
      for (const [index, layer] of surface.layers.entries()) {
        if (layer.drawn !== layers[index] || layer.depth !== depth) {
          const within = depth + SURFACE_MARGIN;
          const placed = surfaceMarks(url, surface.cells, layers[index], within, layer.found, layer.placed);
          layer.handle.set(placed.marks);
          const { from_N, marks } = placed;
          Object.assign(layer, { drawn: layers[index], depth, from_N, cells_N2: marks.cells_N2, shown: null });
        }
        if (layer.shown === shown[index]) continue;
        layer.shown = shown[index];
        const mask = layer.shown === null ? null : new Uint8Array(layer.from_N.length);
        for (let point = 0; point < (mask?.length ?? 0); point++) mask[point] = layer.shown[layer.from_N[point]];
        layer.handle.show(mask);
      }
    }
  }
}
