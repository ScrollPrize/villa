// Records are never changed in place; an edit replaces them.

import { clamp } from "../../math.js";

const HISTORY = 100;
// Finest voxels either side of a slice pane's plane on it: as the plane steps by slices, it holds each node once.
const PLANE_HALF = 0.5;

export function emptySkeleton() {
  return {
    groups: new Map(),
    trees: new Map(),
    nodes: new Map(),
    links: new Map(),
    branchpoints: [],
    selected: null,
    done: [],
    undone: [],
    members: new Map(),
    strokes: new Map(),
    revision: 0,
    touched: new Map(),
    regrouped: 0,
  };
}

export function emptyPatch() {
  return { groups: [], trees: [], place: null, nodes: [], edges: [], branchpoints: null, selected: null };
}

function largest(ids) {
  let most = 0;
  for (const id of ids) most = Math.max(most, id);
  return most;
}

function put(records, id, record) {
  if (record === null) records.delete(id);
  else records.set(id, record);
}

function putNode(skeleton, id, node) {
  const before = skeleton.nodes.get(id);
  if (before !== undefined) {
    const members = skeleton.members.get(before.tree);
    members.delete(id);
    if (members.size === 0) skeleton.members.delete(before.tree);
    const strokes = skeleton.strokes.get(before.tree);
    const stroke = strokes.get(before.stroke ?? id);
    stroke.ids.delete(id);
    if (stroke.ids.size === 0) strokes.delete(before.stroke ?? id);
    else if (id === stroke.first || id === stroke.last) [stroke.first, stroke.last] = [null, null];
    if (strokes.size === 0) skeleton.strokes.delete(before.tree);
    skeleton.touched.set(before.tree, skeleton.revision);
  }
  put(skeleton.nodes, id, node);
  if (node === null) return;
  if (!skeleton.members.has(node.tree)) skeleton.members.set(node.tree, new Set());
  skeleton.members.get(node.tree).add(id);
  if (!skeleton.strokes.has(node.tree)) skeleton.strokes.set(node.tree, new Map());
  const strokes = skeleton.strokes.get(node.tree);
  if (!strokes.has(node.stroke ?? id)) strokes.set(node.stroke ?? id, { ids: new Set(), first: id, last: id });
  const stroke = strokes.get(node.stroke ?? id);
  stroke.ids.add(id);
  if (stroke.first !== null) [stroke.first, stroke.last] = [Math.min(stroke.first, id), Math.max(stroke.last, id)];
  skeleton.touched.set(node.tree, skeleton.revision);
}

export function apply(skeleton, patch, backwards = false) {
  const side = backwards ? 1 : 2;
  skeleton.revision += 1;
  for (const change of patch.groups) put(skeleton.groups, change[0], change[side]);
  if (patch.groups.length > 0) skeleton.regrouped = skeleton.revision;
  for (const change of patch.trees) {
    put(skeleton.trees, change[0], change[side]);
    skeleton.touched.set(change[0], skeleton.revision);
  }
  if (backwards && patch.place !== null)
    for (const [records, places] of [
      [skeleton.groups, patch.place.groups],
      [skeleton.trees, patch.place.trees],
    ]) {
      const back = new Set(places.map(([id]) => id));
      const order = [...records].filter(([id]) => !back.has(id));
      for (const [id, at] of places) order.splice(at, 0, [id, records.get(id)]);
      records.clear();
      for (const [id, record] of order) records.set(id, record);
    }
  for (const change of patch.nodes) putNode(skeleton, change[0], change[side]);
  for (const change of patch.edges) {
    const [a, b] = change[0];
    const end = skeleton.nodes.get(a) ?? skeleton.nodes.get(b);
    if (end !== undefined) skeleton.touched.set(end.tree, skeleton.revision);
    for (const [from, to] of [
      [a, b],
      [b, a],
    ]) {
      const linked = skeleton.links.get(from) ?? new Set();
      if (change[side]) linked.add(to);
      else linked.delete(to);
      put(skeleton.links, from, linked.size > 0 ? linked : null);
    }
  }
  if (patch.branchpoints !== null) skeleton.branchpoints = patch.branchpoints[side - 1];
  if (patch.selected !== null) skeleton.selected = patch.selected[side - 1];
  if (!skeleton.nodes.has(skeleton.selected)) skeleton.selected = null;
}

export function record(skeleton, patch) {
  skeleton.done.push(patch);
  if (skeleton.done.length > HISTORY) skeleton.done.shift();
  skeleton.undone = [];
}

export function commit(skeleton, patch) {
  apply(skeleton, patch);
  record(skeleton, patch);
}

export function undo(skeleton) {
  const patch = skeleton.done.pop();
  if (patch === undefined) return false;
  apply(skeleton, patch, true);
  skeleton.undone.push(patch);
  return true;
}

export function redo(skeleton) {
  const patch = skeleton.undone.pop();
  if (patch === undefined) return false;
  apply(skeleton, patch);
  skeleton.done.push(patch);
  return true;
}

export function selectNode(skeleton, id) {
  skeleton.selected = id;
}

export function addNodes(skeleton, tree, points_ZYX, { from, connected, stroke, comment, view }) {
  const first = largest(skeleton.nodes.keys()) + 1;
  const patch = { ...emptyPatch(), selected: [skeleton.selected, first + points_ZYX.length - 1] };
  for (const [index, point_ZYX] of points_ZYX.entries()) {
    const id = first + index;
    patch.nodes.push([id, null, { id, tree, point_ZYX: [...point_ZYX], radius: 1, stroke, comment, view }]);
    const previous = index === 0 ? from : id - 1;
    if (connected && previous !== null) patch.edges.push([[previous, id], false, true]);
  }
  return patch;
}

export function moveNode(skeleton, id, point_ZYX) {
  const node = skeleton.nodes.get(id);
  return { ...emptyPatch(), nodes: [[id, node, { ...node, point_ZYX: [...point_ZYX] }]] };
}

export function setComment(skeleton, id, comment) {
  const node = skeleton.nodes.get(id);
  return { ...emptyPatch(), nodes: [[id, node, { ...node, comment }]] };
}

// The tree keeps the rest of its nodes, joined or not, since a tree is one fibre or surface whatever breaks it.
export function deleteNode(skeleton, id) {
  const node = skeleton.nodes.get(id);
  const neighbours = [...(skeleton.links.get(id) ?? [])];
  const patch = {
    ...emptyPatch(),
    nodes: [[id, node, null]],
    edges: neighbours.map((other) => [[id, other], true, false]),
  };
  if (skeleton.branchpoints.includes(id))
    patch.branchpoints = [skeleton.branchpoints, skeleton.branchpoints.filter((other) => other !== id)];
  if (skeleton.selected === id) {
    const before = stepNode(skeleton, id, -1);
    const next = before !== id ? before : stepNode(skeleton, id, 1);
    patch.selected = [id, neighbours[0] ?? (next !== id ? next : null)];
  }
  return patch;
}

function strokeEnds(stroke) {
  if (stroke.first === null) {
    let [first, last] = [Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY];
    for (const id of stroke.ids) [first, last] = [Math.min(first, id), Math.max(last, id)];
    [stroke.first, stroke.last] = [first, last];
  }
  return [stroke.first, stroke.last];
}

function strokeRecord(skeleton, id) {
  const { stroke, tree } = skeleton.nodes.get(id);
  return skeleton.strokes.get(tree).get(stroke ?? id);
}

export function strokeOf(skeleton, id) {
  return [...strokeRecord(skeleton, id).ids].sort((a, b) => a - b);
}

export function strokeCount(skeleton, tree) {
  return skeleton.strokes.get(tree)?.size ?? 0;
}

export function strokePlace(skeleton, id) {
  const record = strokeRecord(skeleton, id);
  const [begun] = strokeEnds(record);
  const strokes = skeleton.strokes.get(skeleton.nodes.get(id).tree);
  let place = 0;
  for (const stroke of strokes.values()) if (strokeEnds(stroke)[0] <= begun) place += 1;
  return { place, count: strokes.size, nodes: record.ids.size };
}

export function stepStroke(skeleton, tree, id, step) {
  const from = id === null ? null : strokeEnds(strokeRecord(skeleton, id))[0];
  let found = null;
  for (const stroke of skeleton.strokes.get(tree)?.values() ?? []) {
    const [first] = strokeEnds(stroke);
    const ahead = from === null || (step > 0 ? first > from : first < from);
    if (ahead && (found === null || (step > 0 ? first < found : first > found))) found = first;
  }
  return found === null ? null : strokeOf(skeleton, found);
}

export function lastNode(skeleton, tree) {
  let latest = null;
  for (const stroke of skeleton.strokes.get(tree)?.values() ?? []) {
    const ends = strokeEnds(stroke);
    if (latest === null || ends[0] > latest[0]) latest = ends;
  }
  return latest?.[1] ?? null;
}

export function deleteStroke(skeleton, id) {
  const ids = strokeOf(skeleton, id);
  const gone = new Set(ids);
  const patch = { ...emptyPatch(), nodes: ids.map((each) => [each, skeleton.nodes.get(each), null]) };
  for (const each of ids)
    for (const other of skeleton.links.get(each) ?? []) {
      if (gone.has(other) && other < each) continue;
      patch.edges.push([[each, other], true, false]);
    }
  const kept = skeleton.branchpoints.filter((other) => !gone.has(other));
  if (kept.length < skeleton.branchpoints.length) patch.branchpoints = [skeleton.branchpoints, kept];
  if (gone.has(skeleton.selected)) {
    let [before, after] = [null, null];
    for (const stroke of skeleton.strokes.get(skeleton.nodes.get(id).tree).values()) {
      const ends = strokeEnds(stroke);
      if (ends[0] < ids[0] && (before === null || ends[0] > before[0])) before = ends;
      if (ends[0] > ids[0] && (after === null || ends[0] < after[0])) after = ends;
    }
    patch.selected = [skeleton.selected, before?.[1] ?? after?.[0] ?? null];
  }
  return patch;
}

export function moveNodes(skeleton, ids, tree) {
  const moved = new Set(ids);
  const taken = new Set([...(skeleton.members.get(tree) ?? [])].map((id) => skeleton.nodes.get(id).stroke));
  const strokes = new Map();
  const patch = emptyPatch();
  for (const id of ids) {
    const node = skeleton.nodes.get(id);
    if (node.stroke !== null && !strokes.has(node.stroke)) {
      let stroke = node.stroke;
      for (let n = 2; taken.has(stroke); n++) stroke = `${node.stroke}~${n}`;
      strokes.set(node.stroke, stroke);
      taken.add(stroke);
    }
    patch.nodes.push([id, node, { ...node, tree, stroke: strokes.get(node.stroke) ?? null }]);
    for (const other of skeleton.links.get(id) ?? [])
      if (!moved.has(other)) patch.edges.push([[id, other], true, false]);
  }
  return patch;
}

export function newTree(skeleton, ids) {
  const id = largest(skeleton.trees.keys()) + 1;
  const tree = { id, name: `Class ${id}`, color_RGB: null, group: null };
  const selected = ids.length === 0 ? [skeleton.selected, null] : null;
  return { ...moveNodes(skeleton, ids, id), trees: [[id, null, tree]], selected };
}

export function deleteTree(skeleton, id, next) {
  const place = { groups: [], trees: [[id, [...skeleton.trees.keys()].indexOf(id)]] };
  const patch = { ...emptyPatch(), trees: [[id, skeleton.trees.get(id), null]], place };
  const gone = new Set(skeleton.members.get(id));
  for (const node of gone) patch.nodes.push([node, skeleton.nodes.get(node), null]);
  for (const node of gone)
    for (const other of skeleton.links.get(node) ?? [])
      if (node < other) patch.edges.push([[node, other], true, false]);
  const kept = skeleton.branchpoints.filter((node) => !gone.has(node));
  if (kept.length < skeleton.branchpoints.length) patch.branchpoints = [skeleton.branchpoints, kept];
  patch.selected = [skeleton.selected, next === null ? null : lastNode(skeleton, next)];
  return patch;
}

export function flagBranchpoint(skeleton, id) {
  const flagged = skeleton.branchpoints.includes(id);
  const after = flagged ? skeleton.branchpoints.filter((other) => other !== id) : [...skeleton.branchpoints, id];
  return { ...emptyPatch(), branchpoints: [skeleton.branchpoints, after] };
}

// The edit back to the latest branchpoint, selecting it and taking it off the list, as webKnossos does, or null.
export function jumpBack(skeleton) {
  const last = skeleton.branchpoints.at(-1);
  return last === undefined ? null : { ...flagBranchpoint(skeleton, last), selected: [skeleton.selected, last] };
}

export function sliceDepth(depth) {
  return depth + PLANE_HALF;
}

export function nearestNode(skeleton, point_ZYX, normal_ZYX, depth, reach, shown) {
  const within = sliceDepth(depth);
  let nearest = null;
  let best = reach * reach;
  for (const node of skeleton.nodes.values()) {
    const dz = node.point_ZYX[0] - point_ZYX[0];
    const dy = node.point_ZYX[1] - point_ZYX[1];
    const dx = node.point_ZYX[2] - point_ZYX[2];
    const across = dz * normal_ZYX[0] + dy * normal_ZYX[1] + dx * normal_ZYX[2];
    if (Math.abs(across) > within) continue;
    const along = dz * dz + dy * dy + dx * dx - across * across;
    if (along > best || (along === best && nearest !== null) || !shown(skeleton.trees.get(node.tree))) continue;
    best = along;
    nearest = node.id;
  }
  return nearest;
}

export function stepNode(skeleton, id, step) {
  const ids = [...skeleton.members.get(skeleton.nodes.get(id).tree)].sort((a, b) => a - b);
  return ids[clamp(ids.indexOf(id) + step, 0, ids.length - 1)];
}
