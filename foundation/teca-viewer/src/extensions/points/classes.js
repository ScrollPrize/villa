import { emptyPatch, moveNodes, newTree, strokeCount, strokeOf } from "./skeleton.js";
import { makeTarget, roleTrees, targetLabel, targetTree, treeLabel, treeName, treeRole } from "./targets.js";
import { counted } from "./view.js";

function scribblesSaid(scribbles, nodes, label) {
  if (scribbles === 1) return `Scribble of ${counted(nodes, "node")} is ${label}.`;
  return `${scribbles} scribbles, ${counted(nodes, "node")}, are ${label}.`;
}

export function moveScribbles(skeleton, ids, target) {
  const tree = targetTree(skeleton, target);
  const label = targetLabel(skeleton, { ...target, mark: null });
  const moving = ids.filter((id) => skeleton.nodes.get(id).tree !== tree);
  if (moving.length === 0)
    return { patch: null, said: `${ids.length === 1 ? "This scribble is" : "These scribbles are"} ${label} already.` };
  const made = tree === null ? makeTarget(skeleton, target) : { patch: emptyPatch(), tree };
  const patch = {
    ...moveNodes(
      skeleton,
      moving.flatMap((id) => strokeOf(skeleton, id)),
      made.tree,
    ),
    groups: made.patch.groups,
    trees: made.patch.trees,
  };
  return { patch, said: scribblesSaid(moving.length, patch.nodes.length, label) };
}

export function moveClass(skeleton, id, target) {
  const into = targetTree(skeleton, target);
  const label = targetLabel(skeleton, { ...target, mark: null });
  if (into === id) return { patch: null, said: `This class is ${label} already.` };
  const tree = skeleton.trees.get(id);
  const ids = [...(skeleton.members.get(id) ?? [])];
  const moved = `Class of ${counted(strokeCount(skeleton, id), "scribble")}, ${counted(ids.length, "node")},`;
  if (into === null) {
    const { patch } = makeTarget(skeleton, target);
    const own = treeRole(skeleton.groups, tree);
    const fiber = target.tree === null && !target.air && target.role === "fiber" && own?.role === "fiber";
    const holder = fiber && own.fiber !== target.fiber ? targetTree(skeleton, { ...target, fiber: own.fiber }) : null;
    const why = holder === null || holder === id ? "" : ` (${treeName("fiber", own.fiber)} is taken there)`;
    const made = { ...patch, trees: [[id, tree, { ...patch.trees[0][2], id }]] };
    return { patch: made, said: `${moved} is ${label}${why}.` };
  }
  const place = { groups: [], trees: [[id, [...skeleton.trees.keys()].indexOf(id)]] };
  const patch = { ...moveNodes(skeleton, ids, into), trees: [[id, tree, null]], place };
  return { patch, said: `${moved} joins ${label}.` };
}

export function newClass(skeleton, ids) {
  const patch = newTree(
    skeleton,
    ids.flatMap((id) => strokeOf(skeleton, id)),
  );
  const [[tree, , made]] = patch.trees;
  return { patch, tree, said: ids.length === 0 ? null : scribblesSaid(ids.length, patch.nodes.length, made.name) };
}

export function nameClass(skeleton, id, name) {
  const tree = skeleton.trees.get(id);
  const other = roleTrees(skeleton, tree.group).find((role) => role.tree.id !== id && role.tree.name === name);
  if (other !== undefined)
    return { patch: null, said: `${treeLabel(skeleton, other.tree.id)} is another class already.` };
  return { patch: { ...emptyPatch(), trees: [[id, tree, { ...tree, name }]] }, said: null };
}
