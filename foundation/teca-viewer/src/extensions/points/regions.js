import { emptyPatch, moveNodes } from "./skeleton.js";
import {
  childGroup,
  gapName,
  gapOf,
  groupWinding,
  placeOf,
  roleTrees,
  treeName,
  treeRole,
  windingName,
  withRole,
} from "./targets.js";

const REGION_WINDINGS = 100;

export function regionOf(winding) {
  return winding === null ? null : Math.floor((winding + REGION_WINDINGS / 2 - 1) / REGION_WINDINGS);
}

function regionEnds(winding) {
  const middle = regionOf(winding) * REGION_WINDINGS;
  return [middle - REGION_WINDINGS / 2 + 1, middle + REGION_WINDINGS / 2];
}

export function regionLabel(winding) {
  return regionEnds(winding).map(windingName).join(" to ");
}

export function insertWinding(skeleton, winding) {
  const last = regionEnds(winding)[1];
  const patch = emptyPatch();
  for (const group of skeleton.groups.values()) {
    const gap = gapOf(skeleton.groups, group.id);
    const number = gap === null ? groupWinding(group) : gap.winding;
    const what = gap === null ? "the last winding" : "the air above the last winding";
    if (number === last)
      return { patch: null, said: `Insert would push ${what} of ${regionLabel(winding)} out of that region.` };
    if (number !== null && number >= winding && number < last) {
      const name = gap === null ? windingName(number + 1) : gapName(number + 1);
      patch.groups.push([group.id, group, { ...group, name }]);
    }
  }
  return { patch, said: null };
}

export function newRegion(skeleton, target) {
  let top = null;
  for (const tree of skeleton.trees.values()) {
    const place = placeOf(skeleton.groups, tree.group) ?? gapOf(skeleton.groups, tree.group);
    const region = place === null ? null : regionOf(place.winding);
    if (region !== null) top = Math.max(top ?? region, region);
  }
  const winding = top === null ? 0 : (top + 1) * REGION_WINDINGS;
  return { ...withRole(skeleton, { ...target, winding }, target.role), air: target.air };
}

export function joinRegions(skeleton, from, to) {
  const [fromRegion, toRegion] = [regionOf(from), regionOf(to)];
  if (to === null) return { patch: null, said: "W? is in no region; choose the winding to join to." };
  if (fromRegion === toRegion)
    return {
      patch: null,
      said: `${windingName(from)} and ${windingName(to)} are both in ${regionLabel(to)}; join to another region.`,
    };
  const shift = to - from;
  const moving = new Map();
  for (const tree of skeleton.trees.values()) {
    const place = placeOf(skeleton.groups, tree.group);
    const gap = gapOf(skeleton.groups, tree.group);
    if (place !== null && regionOf(place.winding) === fromRegion)
      moving.set(skeleton.groups.get(tree.group).parent, place.winding);
    if (gap !== null && regionOf(gap.winding) === fromRegion) moving.set(tree.group, gap.winding);
  }
  const windings = [...new Set(moving.values())].sort((a, b) => a - b);
  if (windings.length === 0) return { patch: null, said: `${regionLabel(from)} holds no class to join.` };
  const out = windings.find((winding) => regionOf(winding + shift) !== toRegion);
  if (out !== undefined)
    return {
      patch: null,
      said: `${windingName(out)} would be ${windingName(out + shift)}, outside ${regionLabel(to)}.`,
    };
  const patch = { ...emptyPatch(), place: { groups: [], trees: [] } };
  const at = (records) => new Map([...records.keys()].map((id, index) => [id, index]));
  const [groupAt, treeAt] = [at(skeleton.groups), at(skeleton.trees)];
  const trees = (group) => [...skeleton.trees.values()].filter((tree) => tree.group === group);
  const children = (group) => [...skeleton.groups.values()].filter((child) => child.parent === group);
  const takeOut = (group) => {
    patch.groups.push([group.id, group, null]);
    patch.place.groups.push([group.id, groupAt.get(group.id)]);
  };
  const merge = (source, there) => {
    const roles = roleTrees(skeleton, there);
    const highest = Math.max(0, ...roles.filter(({ role }) => role === "fiber").map(({ fiber }) => fiber));
    let left = children(source).length;
    for (const tree of trees(source)) {
      const role = treeRole(skeleton.groups, tree);
      const same = roles.find((other) => role !== null && role.role !== "fiber" && other.role === role.role);
      if (role === null) left += 1;
      else if (same !== undefined) {
        patch.nodes.push(...moveNodes(skeleton, [...(skeleton.members.get(tree.id) ?? [])], same.tree.id).nodes);
        patch.trees.push([tree.id, tree, null]);
        patch.place.trees.push([tree.id, treeAt.get(tree.id)]);
      } else {
        const name = role.role === "fiber" ? treeName("fiber", highest + role.fiber) : tree.name;
        patch.trees.push([tree.id, tree, { ...tree, group: there, name }]);
      }
    }
    return left;
  };
  for (const [id, winding] of moving) {
    const group = skeleton.groups.get(id);
    const gap = gapOf(skeleton.groups, id) !== null;
    const name = gap ? gapName(winding + shift) : windingName(winding + shift);
    const into = childGroup(skeleton, null, name);
    if (into === null) {
      patch.groups.push([id, group, { ...group, name }]);
      continue;
    }
    if (gap) {
      if (merge(id, into) === 0) takeOut(group);
      continue;
    }
    let kept = trees(id).length;
    for (const side of children(id)) {
      const there = childGroup(skeleton, into, side.name);
      if (placeOf(skeleton.groups, side.id) === null) {
        kept += 1;
        continue;
      }
      if (there === null) {
        patch.groups.push([side.id, side, { ...side, parent: into }]);
        continue;
      }
      if (merge(side.id, there) === 0) takeOut(side);
      else kept += 1;
    }
    if (kept === 0) takeOut(group);
  }
  for (const places of Object.values(patch.place)) places.sort((a, b) => a[1] - b[1]);
  const range = (list) =>
    list.length === 1 ? windingName(list[0]) : `${windingName(list[0])} to ${windingName(list.at(-1))}`;
  return { patch, said: `Joined ${range(windings)} as ${range(windings.map((winding) => winding + shift))}.` };
}
