import * as THREE from "../../../vendor/three.module.js";

const LEAF = 8;
const DASHES = 12;

function same(a, b) {
  return a === b || (a.length === b.length && a.every((value, at) => value === b[at]));
}

export function treeLinks({ points_zyx_N3, edges_E2 }) {
  const count = points_zyx_N3.length / 3;
  const root = Int32Array.from({ length: count }, (_, point) => point);
  const find = (point) => {
    while (root[point] !== point) point = root[point] = root[root[point]];
    return point;
  };
  let pieces = count;
  for (let end = 0; end < edges_E2.length; end += 2) {
    const [a, b] = [find(edges_E2[end]), find(edges_E2[end + 1])];
    if (a !== b) [root[a], pieces] = [b, pieces - 1];
  }
  const order = Int32Array.from({ length: count }, (_, point) => point);
  const leaves = [];
  const build = (first, last) => {
    const box = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
    for (let at = first; at < last; at++)
      for (let axis = 0; axis < 3; axis++) {
        const value = points_zyx_N3[3 * order[at] + axis];
        if (value < box[axis]) box[axis] = value;
        if (value > box[3 + axis]) box[3 + axis] = value;
      }
    const node = { first, last, box, children: null, piece: -1 };
    if (last - first > LEAF) {
      const widths = [0, 1, 2].map((axis) => box[3 + axis] - box[axis]);
      const axis = widths.indexOf(Math.max(...widths));
      const split = (box[axis] + box[3 + axis]) / 2;
      let middle = first;
      for (let at = first; at < last; at++)
        if (points_zyx_N3[3 * order[at] + axis] < split) {
          [order[at], order[middle]] = [order[middle], order[at]];
          middle++;
        }
      if (middle === first) middle = (first + last) >> 1;
      node.children = [build(first, middle), build(middle, last)];
    } else leaves.push(node);
    return node;
  };
  const top = count === 0 ? null : build(0, count);
  const pieceOf = new Int32Array(count);
  const label = (node) => {
    if (node.children === null) {
      const piece = pieceOf[order[node.first]];
      let one = true;
      for (let at = node.first + 1; at < node.last && one; at++) one = pieceOf[order[at]] === piece;
      node.piece = one ? piece : -1;
    } else {
      const [a, b] = node.children.map(label);
      node.piece = a === b ? a : -1;
    }
    return node.piece;
  };
  const gap = (a, b) => {
    let squared = 0;
    for (let axis = 0; axis < 3; axis++) {
      const apart = Math.max(a[axis] - b[3 + axis], 0, b[axis] - a[3 + axis]);
      squared += apart * apart;
    }
    return squared;
  };
  const best = new Float64Array(count);
  const from = new Int32Array(count);
  const to = new Int32Array(count);
  const reachOf = (leaf) => {
    let reach = 0;
    for (let at = leaf.first; at < leaf.last; at++) reach = Math.max(reach, best[pieceOf[order[at]]]);
    return reach;
  };
  const links = [];
  const stack = [];
  const gaps = [];
  while (pieces > 1) {
    for (let point = 0; point < count; point++) pieceOf[point] = find(point);
    label(top);
    best.fill(Infinity);
    for (const leaf of leaves) {
      let reach = reachOf(leaf);
      const look = (node, apart) => {
        if (apart >= reach || (node.piece !== -1 && node.piece === leaf.piece)) return;
        stack.push(node);
        gaps.push(apart);
      };
      look(top, 0);
      while (stack.length > 0) {
        const node = stack.pop();
        if (gaps.pop() >= reach) continue;
        if (node.children === null) {
          for (let at = leaf.first; at < leaf.last; at++) {
            const point = order[at];
            const piece = pieceOf[point];
            let outside = 0;
            for (let axis = 0; axis < 3; axis++) {
              const value = points_zyx_N3[3 * point + axis];
              const apart = Math.max(node.box[axis] - value, 0, value - node.box[3 + axis]);
              outside += apart * apart;
            }
            if (outside >= best[piece]) continue;
            for (let near = node.first; near < node.last; near++) {
              const other = order[near];
              if (pieceOf[other] === piece) continue;
              const dz = points_zyx_N3[3 * point] - points_zyx_N3[3 * other];
              const dy = points_zyx_N3[3 * point + 1] - points_zyx_N3[3 * other + 1];
              const dx = points_zyx_N3[3 * point + 2] - points_zyx_N3[3 * other + 2];
              const squared = dz * dz + dy * dy + dx * dx;
              if (squared < best[piece]) [best[piece], from[piece], to[piece]] = [squared, point, other];
            }
          }
          reach = reachOf(leaf);
          continue;
        }
        const [a, b] = node.children;
        const [toA, toB] = [gap(leaf.box, a.box), gap(leaf.box, b.box)];
        const aFirst = toA <= toB;
        look(aFirst ? b : a, aFirst ? toB : toA);
        look(aFirst ? a : b, aFirst ? toA : toB);
      }
    }
    for (let piece = 0; piece < count; piece++) {
      if (pieceOf[piece] !== piece) continue;
      const [a, b] = [find(from[piece]), find(to[piece])];
      if (a === b) continue;
      [root[a], pieces] = [b, pieces - 1];
      links.push(Math.min(from[piece], to[piece]), Math.max(from[piece], to[piece]));
    }
  }
  return Uint32Array.from(links);
}

export class Links3d {
  constructor(viewer) {
    this.viewer = viewer;
    this.drawn = [];
    this.of = null;
    this.links = new Map();
  }

  draw(joined, shown, key) {
    if (joined !== null && this.of?.[0] === joined && this.of[1] === key) return;
    for (const [object, handle] of this.drawn) {
      handle.remove();
      object.geometry.dispose();
      object.material.dispose();
    }
    [this.drawn, this.of] = [[], null];
    if (joined === null) return;
    this.of = [joined, key];
    const { marks, pieces, starts } = joined.lines;
    const { points_zyx_N3, colors_N3 } = marks;
    const positions_xyz_N3 = new Float32Array(points_zyx_N3.length);
    for (let value = 0; value < points_zyx_N3.length; value += 3)
      for (let axis = 0; axis < 3; axis++) positions_xyz_N3[value + axis] = points_zyx_N3[value + 2 - axis];
    for (const tree of this.links.keys()) if (!joined.pieces.has(tree)) this.links.delete(tree);
    const [strokes, ends] = [[], []];
    for (const [tree, piece] of joined.pieces) {
      if (!shown(tree)) continue;
      const { points_zyx_N3: own_zyx_N3, edges_E2 } = pieces[piece].marks;
      let last = this.links.get(tree);
      if (last === undefined || !same(last.points_zyx_N3, own_zyx_N3) || !same(last.edges_E2, edges_E2)) {
        last = { points_zyx_N3: own_zyx_N3, edges_E2, links: treeLinks({ points_zyx_N3: own_zyx_N3, edges_E2 }) };
        this.links.set(tree, last);
      }
      for (const point of edges_E2) strokes.push(starts[piece] + point);
      for (const point of last.links) ends.push(starts[piece] + point);
    }
    const scribbles = new THREE.BufferGeometry();
    scribbles.setAttribute("position", new THREE.BufferAttribute(positions_xyz_N3, 3));
    scribbles.setAttribute("color", new THREE.BufferAttribute(colors_N3, 3));
    scribbles.setIndex(strokes);
    const linkPositions_xyz_M3 = new Float32Array(3 * ends.length);
    const linkColors_M3 = new Float32Array(3 * ends.length);
    for (const [end, point] of ends.entries()) {
      linkPositions_xyz_M3.set(positions_xyz_N3.subarray(3 * point, 3 * point + 3), 3 * end);
      linkColors_M3.set(colors_N3.subarray(3 * point, 3 * point + 3), 3 * end);
    }
    const joins = new THREE.BufferGeometry();
    joins.setAttribute("position", new THREE.BufferAttribute(linkPositions_xyz_M3, 3));
    joins.setAttribute("color", new THREE.BufferAttribute(linkColors_M3, 3));
    const along_M = Float32Array.from(ends, (_, end) => (end % 2) * 2 * DASHES);
    joins.setAttribute("lineDistance", new THREE.BufferAttribute(along_M, 1));
    this.drawn = [
      new THREE.LineSegments(scribbles, new THREE.LineBasicMaterial({ vertexColors: true })),
      new THREE.LineSegments(joins, new THREE.LineDashedMaterial({ vertexColors: true, dashSize: 1, gapSize: 1 })),
    ].map((object) => [object, this.viewer.add3d(object)]);
  }
}
