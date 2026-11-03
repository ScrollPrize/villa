import { emptyPatch } from "./skeleton.js";

const WINDING = /^W(\?|0|[+-][1-9]\d*)$/;
const GAP = /^W(\?|0|[+-][1-9]\d*)–W(?:\?|0|[+-][1-9]\d*)$/;
const SIDE = /^(recto|verso|side \?)(?: S(\?|[+-][1-9]\d*))?$/;
const ROLE = /^(?:fiber ([1-9]\d*)|ink|no ink)$/;
const AIR = "air";
const SIDES = ["recto", "verso"];
const ROLES = ["fiber", "ink", "no ink"];
const MARKS = ["kollesis"];
const ROLE_COLORS = { ink: [1, 0.1, 0.1], "no ink": [0.1, 1, 0.2], air: [0.05, 0.7, 1] };
const START = { winding: 0, side: "recto", sheet: 0, role: "fiber", fiber: 1, tree: null, mark: null, air: false };
const SHEETS_BESIDE = 1;

function signed(value) {
  if (value === null) return "?";
  return value > 0 ? `+${value}` : String(value);
}

function index(text) {
  return text === "?" ? null : Number(text);
}

export function windingName(winding) {
  return `W${signed(winding)}`;
}

export function gapName(winding) {
  return `${windingName(winding)}–${windingName(winding === null ? null : winding + 1)}`;
}

function sideName(side, sheet) {
  return `${side ?? "side ?"}${sheet === 0 ? "" : ` S${signed(sheet)}`}`;
}

export function treeName(role, fiber) {
  return role === "fiber" ? `fiber ${fiber}` : role;
}

export function groupWinding(group) {
  const match = group.parent === null ? WINDING.exec(group.name) : null;
  return match === null ? null : index(match[1]);
}

export function gapOf(groups, group) {
  const gapGroup = group === null ? undefined : groups.get(group);
  const match = gapGroup === undefined || gapGroup.parent !== null ? null : GAP.exec(gapGroup.name);
  return match === null || gapName(index(match[1])) !== gapGroup.name ? null : { winding: index(match[1]) };
}

export function startTarget() {
  return { ...START };
}

export function placeOf(groups, group) {
  const sideGroup = group === null ? undefined : groups.get(group);
  const side = sideGroup === undefined ? null : SIDE.exec(sideGroup.name);
  if (side === null || sideGroup.parent === null) return null;
  const windingGroup = groups.get(sideGroup.parent);
  const winding = WINDING.exec(windingGroup.name);
  if (winding === null || windingGroup.parent !== null) return null;
  return {
    winding: index(winding[1]),
    side: side[1] === "side ?" ? null : side[1],
    sheet: side[2] === undefined ? 0 : index(side[2]),
  };
}

function roleOf(name) {
  const match = ROLE.exec(name);
  if (match === null) return null;
  return match[1] === undefined ? { role: name, fiber: null } : { role: "fiber", fiber: Number(match[1]) };
}

export function treeRole(groups, tree) {
  if (gapOf(groups, tree.group) !== null) return tree.name === AIR ? { role: AIR, fiber: null } : null;
  return placeOf(groups, tree.group) === null ? null : roleOf(tree.name);
}

export function treeTarget(skeleton, id, from) {
  const tree = skeleton.trees.get(id);
  const role = treeRole(skeleton.groups, tree);
  if (role === null) return { ...from, tree: id };
  if (role.role === AIR) return { ...from, winding: gapOf(skeleton.groups, tree.group).winding, tree: null, air: true };
  return { ...placeOf(skeleton.groups, tree.group), ...role, tree: null, mark: from.mark, air: false };
}

export function childGroup(skeleton, parent, name) {
  for (const group of skeleton.groups.values()) if (group.parent === parent && group.name === name) return group.id;
  return null;
}

function sideGroupOf(skeleton, target) {
  const winding = childGroup(skeleton, null, windingName(target.winding));
  return winding === null ? null : childGroup(skeleton, winding, sideName(target.side, target.sheet));
}

export function roleTrees(skeleton, group) {
  const found = [];
  for (const tree of skeleton.trees.values()) {
    const role = tree.group === group ? treeRole(skeleton.groups, tree) : null;
    if (role !== null) found.push({ tree, ...role });
  }
  return found;
}

function targetRole(target) {
  return target.air ? AIR : target.role;
}

export function targetTree(skeleton, target) {
  if (target.tree !== null) return skeleton.trees.has(target.tree) ? target.tree : null;
  const group = target.air ? childGroup(skeleton, null, gapName(target.winding)) : sideGroupOf(skeleton, target);
  if (group === null) return null;
  const name = treeName(targetRole(target), target.fiber);
  return roleTrees(skeleton, group).find(({ tree }) => tree.name === name)?.tree.id ?? null;
}

export function makeTarget(skeleton, target) {
  const patch = emptyPatch();
  let groupId = Math.max(0, ...skeleton.groups.keys());
  const group = (parent, name) => {
    const found = parent === null || skeleton.groups.has(parent) ? childGroup(skeleton, parent, name) : null;
    if (found !== null) return found;
    groupId += 1;
    patch.groups.push([groupId, null, { id: groupId, name, parent }]);
    return groupId;
  };
  const into = target.air
    ? group(null, gapName(target.winding))
    : group(group(null, windingName(target.winding)), sideName(target.side, target.sheet));
  const id = Math.max(0, ...skeleton.trees.keys()) + 1;
  const tree = {
    id,
    name: treeName(targetRole(target), target.fiber),
    color_RGB: ROLE_COLORS[targetRole(target)] ?? null,
    group: into,
  };
  patch.trees.push([id, null, tree]);
  return { patch, tree: id };
}

export function targetLabel(skeleton, target) {
  const mark = target.mark === null ? "" : ` › ${target.mark}`;
  if (target.tree !== null) return skeleton.trees.get(target.tree).name + mark;
  const place = target.air
    ? [gapName(target.winding)]
    : [windingName(target.winding), sideName(target.side, target.sheet)];
  return [...place, treeName(targetRole(target), target.fiber)].join(" › ") + mark;
}

export function treeLabel(skeleton, id) {
  return targetLabel(skeleton, treeTarget(skeleton, id, START));
}

export function classList(skeleton) {
  const targets = new Map();
  const other = [];
  for (const id of skeleton.trees.keys()) {
    const target = treeTarget(skeleton, id, START);
    if (target.tree === null) targets.set(id, target);
    else other.push(id);
  }
  const rank = ({ winding, side, sheet, role, fiber, air }) => [
    winding ?? Infinity,
    air ? 1 : 0,
    [...SIDES, null].indexOf(side),
    sheet ?? Infinity,
    ROLES.indexOf(role),
    fiber ?? 0,
  ];
  const ranks = new Map([...targets].map(([id, target]) => [id, rank(target)]));
  const order = (a, b) => {
    const [x, y] = [ranks.get(a), ranks.get(b)];
    const at = x.findIndex((value, i) => value !== y[i]);
    return at === -1 ? 0 : x[at] - y[at];
  };
  const list = [];
  for (const id of [...targets.keys()].sort(order)) {
    const name = windingName(targets.get(id).winding);
    if (list.at(-1)?.name !== name) list.push({ name, ids: [] });
    list.at(-1).ids.push(id);
  }
  return other.length === 0 ? list : [...list, { name: "Other", ids: other }];
}

export function classNeighbour(skeleton, id) {
  const ids = classList(skeleton).flatMap((group) => group.ids);
  const at = ids.indexOf(id);
  return ids[at - 1] ?? ids[at + 1] ?? null;
}

export function classStep(skeleton, id, step) {
  const holding = (tree) => tree === id || (skeleton.members.get(tree)?.size ?? 0) > 0;
  const ids = classList(skeleton).flatMap((group) => group.ids.filter(holding));
  const at = ids.indexOf(id);
  if (at === -1) return ids.at(step > 0 ? 0 : -1) ?? null;
  return ids[at + step] ?? null;
}

export function treeColor(tree, labelColor) {
  return tree.color_RGB ?? labelColor(tree.id);
}

export function targetColor(skeleton, target, labelColor) {
  const id = targetTree(skeleton, target);
  if (id !== null) return treeColor(skeleton.trees.get(id), labelColor);
  return ROLE_COLORS[targetRole(target)] ?? labelColor(Math.max(0, ...skeleton.trees.keys()) + 1);
}

export function readTarget(value) {
  const whole = (field, least) => Number.isSafeInteger(value?.[field]) && value[field] >= least;
  const checks = {
    winding: value?.winding === null || whole("winding", -Infinity),
    side: [null, ...SIDES].includes(value?.side),
    sheet: value?.sheet === null || whole("sheet", -Infinity),
    role: ROLES.includes(value?.role),
    fiber: value?.role === "fiber" ? whole("fiber", 1) : value?.fiber === null,
    tree: value?.tree === null || whole("tree", 0),
    mark: [undefined, null, ...MARKS].includes(value?.mark),
    air: [undefined, false, true].includes(value?.air),
  };
  for (const [field, good] of Object.entries(checks))
    if (!good) throw new Error(`The address's Points class has ${field} ${JSON.stringify(value?.[field])}.`);
  const fields = Object.fromEntries(Object.keys(START).map((field) => [field, value[field]]));
  return { ...fields, mark: value.mark ?? null, air: value.air ?? false };
}

function lastFiber(skeleton, place) {
  const group = sideGroupOf(skeleton, place);
  const fibers = group === null ? [] : roleTrees(skeleton, group).filter(({ role }) => role === "fiber");
  if (fibers.length === 0) return 1;
  const latest = new Map();
  for (const node of skeleton.nodes.values()) latest.set(node.tree, Math.max(latest.get(node.tree) ?? 0, node.id));
  let best = fibers[0];
  for (const fiber of fibers) {
    const [a, b] = [latest.get(fiber.tree.id) ?? 0, latest.get(best.tree.id) ?? 0];
    if (a > b || (a === b && fiber.fiber > best.fiber)) best = fiber;
  }
  return best.fiber;
}

function sideFibers(skeleton, target) {
  const group = sideGroupOf(skeleton, target);
  const fibers = group === null ? [] : roleTrees(skeleton, group).filter(({ role }) => role === "fiber");
  return [...new Set(fibers.map(({ fiber }) => fiber))].sort((a, b) => a - b);
}

export function withRole(skeleton, target, role, keepFiber = false) {
  const place = { ...target, role, tree: null, air: false };
  if (role !== "fiber") return { ...place, fiber: null };
  if (!keepFiber || target.tree !== null) return { ...place, fiber: lastFiber(skeleton, place) };
  const taken = sideFibers(skeleton, place);
  return { ...place, fiber: taken.includes(target.fiber) ? Math.max(...taken) + 1 : target.fiber };
}

export function newFiber(skeleton, target) {
  const fiber = Math.max(0, ...sideFibers(skeleton, target)) + 1;
  return { ...target, role: "fiber", tree: null, fiber, air: false };
}

export function otherSide(skeleton, target, keepFiber = false) {
  const side = target.side === "recto" ? "verso" : "recto";
  return withRole(skeleton, { ...target, side }, target.role, keepFiber);
}

export function windingStep(skeleton, target, step, keepFiber = false) {
  if (target.winding === null) return null;
  const place = { ...target, winding: target.winding + step };
  return { ...withRole(skeleton, place, target.role, keepFiber), air: target.air };
}

export function switchAir(target) {
  if (target.tree === null && target.air) return { ...target, air: false };
  return { ...target, tree: null, air: true };
}

export function targetChoices(skeleton, target) {
  const own = target.sheet ?? 0;
  const sheets = [];
  for (let sheet = Math.min(0, own) - SHEETS_BESIDE; sheet <= Math.max(0, own) + SHEETS_BESIDE; sheet++)
    sheets.push([sheet, `S${signed(sheet)}`]);
  const made = sideFibers(skeleton, target);
  const next = Math.max(0, ...made) + 1;
  const onFiber = targetRole(target) === "fiber" && target.tree === null;
  const targeted = onFiber && target.fiber !== next ? [target.fiber] : [];
  const fibers = [...new Set([...made, ...targeted])].sort((a, b) => a - b).map((fiber) => [fiber, `fiber ${fiber}`]);
  return {
    winding: windingName(target.winding),
    sheets: [...sheets, [null, `S${signed(null)}`]],
    fibers: [...fibers, [next, `new fiber ${next}`]],
    tree: skeleton.trees.get(target.tree)?.name ?? null,
  };
}

export function pickerTarget(skeleton, target, part, keepFiber = false) {
  if (part.fiber !== undefined) return { ...target, role: "fiber", fiber: part.fiber, tree: null, air: false };
  if (part.air) return target.tree === null && target.air ? target : switchAir(target);
  const air = target.air && part.winding !== undefined;
  const place = { ...target, ...part };
  const same = Object.keys(part).every((field) => place[field] === target[field]);
  if (target.tree === null && target.air === air && same) return target;
  return { ...withRole(skeleton, place, place.role, keepFiber && part.role === undefined), air };
}

export function inWinding(skeleton, id, winding) {
  const { group } = skeleton.trees.get(id);
  const place = placeOf(skeleton.groups, group) ?? gapOf(skeleton.groups, group);
  return place !== null && place.winding === winding;
}
