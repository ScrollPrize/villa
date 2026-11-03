import { commentTags, runOf } from "./runs.js";
import { emptySkeleton, strokePlace } from "./skeleton.js";
import { classList, targetChoices, targetColor, targetLabel, targetTree, treeLabel, windingName } from "./targets.js";

export function counted(count, noun, nouns = `${noun}s`) {
  return `${count} ${count === 1 ? noun : nouns}`;
}

export function pointsView(
  { skeleton, watched, drawn, aim, mode, active, unsaved, kept, clear, depth, runs, labelColor },
  message,
) {
  const ready = skeleton !== null;
  const { target } = aim;
  const tree = ready ? targetTree(skeleton, target) : null;
  const node = ready && skeleton.selected !== null ? skeleton.nodes.get(skeleton.selected) : null;
  const color_RGB = ready ? targetColor(skeleton, target, labelColor) : [1, 1, 1];
  const count = (id) => skeleton.members.get(id)?.size ?? 0;
  let note = message;
  if (note === "") {
    if (mode.joining !== null)
      note = `Join ${windingName(mode.joining)}: choose the winding it is, then Join again; Escape cancels.`;
    else if (ready && skeleton.trees.size > 0) {
      const classes = counted(skeleton.trees.size, "class", "classes");
      const made =
        tree === null ? `; ${targetLabel(skeleton, { ...target, mark: null })} is made with its first node` : "";
      const nodes = counted(skeleton.nodes.size, "node");
      note = `${classes}, ${nodes}${unsaved ? ", not saved" : ""}, ${kept} in this browser${made}.`;
    } else if (watched !== null) {
      const shown = `${counted(watched.nodes.size, "node")} kept ${watched.nodes.size === 1 ? "shows" : "show"}`;
      note = `Another tab has this CT's nodes open; the ${shown} here, to edit once it lets go of them.`;
    } else if (active && !mode.editing) note = "Click a slice pane to place a node, drag to scribble; Space pans.";
  }
  const view = {
    ready,
    active,
    undo: ready && skeleton.done.length > 0,
    redo: ready && skeleton.undone.length > 0,
    clear,
    insert: ready && target.winding !== null,
    joining: mode.joining !== null,
    editing: mode.editing,
    target,
    choices: targetChoices(skeleton ?? emptySkeleton(), target),
    color_RGB,
    only: aim.only,
    follow: aim.following,
    links: aim.linked,
    spacing: aim.spacing(),
    depth,
    tree: tree === null ? null : { id: tree, name: skeleton.trees.get(tree).name },
    mark: target.mark,
    tags: node === null ? null : commentTags(node.comment),
    scribble: node === null ? null : strokePlace(skeleton, node.id),
    empty: node === null && tree !== null && count(tree) === 0,
    classes: (ready ? classList(skeleton) : []).map(({ name, ids }) => ({
      name,
      trees: ids.map((id) => ({ id, label: `${treeLabel(skeleton, id)} (${counted(count(id), "node")})` })),
    })),
    runs,
    runRow: ({ tree, mark, ids }) => ({
      label: `${treeLabel(drawn, tree)} › ${mark}`,
      count: counted(ids.length, "node"),
    }),
    currentRun: node === null ? -1 : runOf(runs, skeleton, node.id),
    note,
  };
  if (!ready) return { view, label: null };
  const where = targetLabel(skeleton, target);
  const text = !mode.editing
    ? where
    : mode.whole
      ? `Edit class: ${where}`
      : mode.picked() !== null
        ? `Edit${mode.kept.length > 0 ? ` ${mode.kept.length} kept` : ""}: ${where}`
        : "Edit: click a scribble";
  return { view, label: { text, color_RGB } };
}
