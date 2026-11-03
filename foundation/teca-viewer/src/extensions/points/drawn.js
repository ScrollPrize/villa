import { bandMarks, commentMark, treeRuns } from "./runs.js";

export function treeMarks(skeleton, tree) {
  const ids = [...(skeleton.members.get(tree) ?? [])].sort((a, b) => a - b);
  const index = new Map();
  const points_zyx_N3 = new Float32Array(3 * ids.length);
  const neighbours_N2 = new Int32Array(2 * ids.length).fill(-1);
  const edges = [];
  const strokes = new Map();
  for (const [at, id] of ids.entries()) {
    const node = skeleton.nodes.get(id);
    index.set(id, at);
    points_zyx_N3.set(node.point_ZYX, 3 * at);
    for (const other of skeleton.links.get(id) ?? []) if (other < id) edges.push(index.get(other), at);
    if (node.stroke === null) continue;
    const before = strokes.get(node.stroke);
    if (before !== undefined) {
      if (!skeleton.links.get(id)?.has(ids[before])) edges.push(before, at);
      [neighbours_N2[2 * before + 1], neighbours_N2[2 * at]] = [at, before];
    }
    strokes.set(node.stroke, at);
  }
  const marks = { points_zyx_N3, edges_E2: Uint32Array.from(edges), neighbours_N2 };
  return { marks, ids_N: Float64Array.from(ids) };
}

function treePart(skeleton, tree, depth) {
  const lines = treeMarks(skeleton, tree);
  const marked = new Map();
  for (const id of lines.ids_N) {
    const { comment } = skeleton.nodes.get(id);
    const mark = comment === null ? null : commentMark(comment);
    if (mark !== null) marked.set(id, mark);
  }
  const bands = bandMarks(skeleton, marked, depth, [lines]);
  return { revision: skeleton.revision, lines, bands, runs: treeRuns(tree, lines.ids_N, marked) };
}

function joinMarks(pieces, colors, depth) {
  let [count, ends] = [0, 0];
  for (const { marks, ids_N } of pieces) [count, ends] = [count + ids_N.length, ends + marks.edges_E2.length];
  const points_zyx_N3 = new Float32Array(3 * count);
  const colors_N3 = new Float32Array(3 * count);
  const crowding = pieces.some(({ marks }) => marks.neighbours_N2 !== undefined);
  const neighbours_N2 = crowding ? new Int32Array(2 * count).fill(-1) : null;
  const edges_E2 = new Uint32Array(ends);
  const ids_N = new Float64Array(count);
  const starts = [];
  let [at, end] = [0, 0];
  for (const [piece, { marks, ids_N: own }] of pieces.entries()) {
    starts.push(at);
    points_zyx_N3.set(marks.points_zyx_N3, 3 * at);
    ids_N.set(own, at);
    const color = colors[piece];
    if (color === null) colors_N3.set(marks.colors_N3, 3 * at);
    else
      for (let value = 3 * at; value < 3 * (at + own.length); value += 3) {
        colors_N3[value] = color[0];
        colors_N3[value + 1] = color[1];
        colors_N3[value + 2] = color[2];
      }
    for (let edge = 0; edge < marks.edges_E2.length; edge++) edges_E2[end + edge] = marks.edges_E2[edge] + at;
    for (let value = 0; value < (marks.neighbours_N2?.length ?? 0); value++)
      if (marks.neighbours_N2[value] >= 0) neighbours_N2[2 * at + value] = marks.neighbours_N2[value] + at;
    [at, end] = [at + own.length, end + marks.edges_E2.length];
  }
  const marks = { points_zyx_N3, colors_N3, edges_E2, neighbours_N2, selected: -1, depth };
  return { marks, ids_N, pieces, starts };
}

function pointOf(joined, piece, id) {
  const end = joined.starts[piece + 1] ?? joined.ids_N.length;
  let [low, high] = [joined.starts[piece], end];
  while (low < high) {
    const middle = (low + high) >> 1;
    if (joined.ids_N[middle] < id) low = middle + 1;
    else high = middle;
  }
  return low < end && joined.ids_N[low] === id ? low : -1;
}

export class DrawnTrees {
  constructor() {
    this.skeleton = null;
    this.parts = new Map();
    this.joined = null;
    this.shown = null;
    this.listed = null;
  }

  join(skeleton, colorOf, bright, depth) {
    let changed = skeleton !== this.skeleton || this.joined.bright !== bright || this.joined.depth !== depth;
    if (skeleton !== this.skeleton) [this.skeleton, this.parts] = [skeleton, new Map()];
    for (const id of this.parts.keys()) {
      if (skeleton.trees.has(id)) continue;
      this.parts.delete(id);
      changed = true;
    }
    for (const id of skeleton.trees.keys()) {
      if (this.parts.get(id)?.revision >= (skeleton.touched.get(id) ?? 0)) continue;
      this.parts.set(id, treePart(skeleton, id, depth));
      changed = true;
    }
    if (!changed) return this.joined;
    const trees = [...skeleton.trees.values()];
    const parts = trees.map((tree) => this.parts.get(tree.id));
    const lines = joinMarks(
      parts.map((part) => part.lines),
      trees.map(colorOf),
      depth,
    );
    const bands = joinMarks(
      parts.map((part) => part.bands),
      trees.map(() => null),
      depth,
    );
    const pieces = new Map(trees.map((tree, piece) => [tree.id, piece]));
    this.joined = { lines, bands, pieces, bright, depth };
    return this.joined;
  }

  hidden(shown, key, pressed) {
    const last = this.shown;
    const joined = this.joined;
    if (last?.joined === joined && last.key === key && last.pressed === pressed) return last;
    const at = { lines: [], bands: [] };
    for (const id of pressed ?? []) {
      const piece = joined.pieces.get(this.skeleton.nodes.get(id).tree);
      for (const layer of ["lines", "bands"]) {
        const point = piece === undefined ? -1 : pointOf(joined[layer], piece, id);
        if (point >= 0) at[layer].push(point);
      }
    }
    const pressedAt = `${at.lines}:${at.bands}`;
    if (last?.joined === joined && last.key === key && last.pressedAt === pressedAt)
      return Object.assign(last, { pressed });
    const masks = {};
    for (const layer of ["lines", "bands"]) {
      const { ids_N, starts } = joined[layer];
      let mask = null;
      const hide = (from, to) => {
        mask ??= new Uint8Array(ids_N.length).fill(1);
        mask.fill(0, from, to);
      };
      for (const [id, piece] of joined.pieces)
        if (!shown(this.skeleton.trees.get(id))) hide(starts[piece], starts[piece + 1] ?? ids_N.length);
      for (const point of at[layer]) hide(point, point + 1);
      masks[layer] = mask;
    }
    this.shown = { joined, key, pressed, pressedAt, ...masks };
    return this.shown;
  }

  runs(shown, key) {
    if (this.listed?.joined === this.joined && this.listed.key === key) return this.listed.runs;
    const trees = [...this.skeleton.trees.values()].filter(shown).sort((a, b) => a.id - b.id);
    const runs = trees.flatMap((tree) => this.parts.get(tree.id).runs);
    this.listed = { joined: this.joined, key, runs };
    return runs;
  }
}
