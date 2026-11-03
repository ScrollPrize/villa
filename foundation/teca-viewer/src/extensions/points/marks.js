import { DrawnTrees } from "./drawn.js";
import { Links3d } from "./links.js";
import { bandMarks, commentMark } from "./runs.js";
import { sliceDepth } from "./skeleton.js";
import { SurfaceNodes } from "./surface.js";
import { inWinding, treeColor } from "./targets.js";

// Finest voxels across the band under marked nodes the lines join, and the dot under one they join to none: a
// fiber's width. Across the XY pane about the checks' segment at zoom 4 and 8, 53 of 69 bright runs are 3 to 20
// voxels across, with a median of 7.8 and quartiles of 5.3 and 13.6.
const BAND_VOXELS = 8;
// CSS pixels the band is at least, twice its line's 2, so zoomed out it shows beside the line and the bands of fibers
// 20 voxels apart part while a voxel is over 0.2 px; and at most, reached past 4 px a voxel, so zoomed in the fiber's
// edges show beside it. The dot is a ring about a node's disc once wider than its 8.
const BAND_WIDTH = [4, 32];
const DIMMED = 0.5;

export function pressMarks(skeleton, colorOf, depth, shown, pressed) {
  const joins = [];
  for (const id of pressed)
    for (const other of skeleton.links.get(id) ?? []) if (id < other || !pressed.has(other)) joins.push(id, other);
  const strokes = new Map();
  for (const id of pressed) {
    const { tree, stroke } = skeleton.nodes.get(id);
    if (stroke === null) continue;
    if (!strokes.has(tree)) strokes.set(tree, new Set());
    strokes.get(tree).add(stroke);
  }
  for (const [tree, pressedStrokes] of strokes) {
    const before = new Map();
    const ids = [...skeleton.members.get(tree)].filter((id) => pressedStrokes.has(skeleton.nodes.get(id).stroke));
    for (const id of ids.sort((a, b) => a - b)) {
      const { stroke } = skeleton.nodes.get(id);
      const other = before.get(stroke);
      if (other !== undefined && (pressed.has(id) || pressed.has(other)) && !skeleton.links.get(id)?.has(other))
        joins.push(other, id);
      before.set(stroke, id);
    }
  }
  const colors = new Map();
  const colorFor = (tree) => {
    if (!colors.has(tree)) colors.set(tree, shown(skeleton.trees.get(tree)) ? colorOf(skeleton.trees.get(tree)) : null);
    return colors.get(tree);
  };
  const candidates = new Set([...pressed, ...joins]);
  if (skeleton.selected !== null) candidates.add(skeleton.selected);
  const drawn = [...candidates].filter((id) => colorFor(skeleton.nodes.get(id).tree) !== null);
  const index = new Map(drawn.map((id, at) => [id, at]));
  const points_zyx_N3 = new Float32Array(drawn.length * 3);
  const colors_N3 = new Float32Array(drawn.length * 3);
  for (const [at, id] of drawn.entries()) {
    const node = skeleton.nodes.get(id);
    points_zyx_N3.set(node.point_ZYX, 3 * at);
    colors_N3.set(colorFor(node.tree), 3 * at);
  }
  const edges = [];
  for (let at = 0; at < joins.length; at += 2) {
    const [a, b] = [index.get(joins[at]), index.get(joins[at + 1])];
    if (a !== undefined && b !== undefined) edges.push(a, b);
  }
  const selected = index.get(skeleton.selected) ?? -1;
  const marks = { points_zyx_N3, colors_N3, edges_E2: Uint32Array.from(edges), selected, depth };
  return { marks, ids_N: Float64Array.from(drawn) };
}

export class TreeMarks {
  constructor(viewer, failed) {
    const band = { diameter: 0, width: BAND_WIDTH, voxels: BAND_VOXELS };
    this.settledBands = viewer.addMarks(band);
    this.liveBands = viewer.addMarks(band);
    this.settledMarks = viewer.addMarks();
    this.liveMarks = viewer.addMarks();
    this.onSurfaces = new SurfaceNodes(viewer, failed, [band, band, {}, {}]);
    this.trees = new DrawnTrees();
    this.inSpace = new Links3d(viewer);
    this.settled = null;
    this.hidden = null;
    this.runs = [];
    this.live = null;
    this.pressBands = null;
    this.liveOf = null;
    this.liveSelected = null;
    this.labelColor = viewer.labelColor;
    this.shown = () => true;
  }

  draw(drawn, { bright, only, winding, depth, picked, linked }, pressed, rejoin) {
    const colorOf = (tree) => {
      const color_RGB = treeColor(tree, this.labelColor);
      return tree.id === bright ? color_RGB : color_RGB.map((value) => value * DIMMED);
    };
    const shown = only ? (tree) => inWinding(drawn, tree.id, winding) : () => true;
    this.shown = shown;
    const view = only ? `winding ${winding} of ${drawn.regrouped}` : "all";
    const slab = sliceDepth(depth);
    if (rejoin || pressed === null) {
      const joined = this.trees.join(drawn, colorOf, bright, slab);
      if (joined !== this.settled) {
        this.settled = joined;
        this.settledMarks.set(joined.lines.marks);
        this.settledBands.set(joined.bands.marks);
      }
      this.runs = this.trees.runs(shown, view);
      this.inSpace.draw(linked ? joined : null, (id) => shown(drawn.trees.get(id)), view);
    }
    const hiding = this.trees.hidden(shown, view, pressed);
    if (hiding !== this.hidden) {
      this.hidden = hiding;
      this.settledMarks.show(hiding.lines);
      this.settledBands.show(hiding.bands);
    }
    // A class of 50,000 nodes picked whole takes 45 to 80 ms to build, at a pick or an edit but not at a step.
    const of = [drawn, drawn.revision, picked, bright, view, slab];
    const inPick = (id) => picked?.includes(id) ?? false;
    const { selected } = drawn;
    const kept =
      pressed === null &&
      of.every((each, at) => each === this.liveOf?.[at]) &&
      (selected === this.liveSelected || (inPick(selected) && inPick(this.liveSelected)));
    if (kept && selected !== this.liveSelected) {
      const { marks, ids_N } = this.live;
      this.live = { marks: { ...marks, selected: ids_N.indexOf(selected) }, ids_N };
      this.liveMarks.set(this.live.marks);
      this.liveSelected = selected;
    } else if (!kept) {
      const pressing = picked === null ? (pressed ?? new Set()) : new Set(picked);
      this.live = pressMarks(drawn, colorOf, slab, shown, pressing);
      this.liveMarks.set(this.live.marks);
      this.liveSelected = selected;
      this.liveOf = null;
      if (rejoin || [...(pressed ?? [])].some((id) => commentMark(drawn.nodes.get(id).comment) !== null)) {
        const marked = new Map();
        for (const id of this.live.ids_N) {
          const mark = commentMark(drawn.nodes.get(id).comment);
          if (mark !== null) marked.set(id, mark);
        }
        for (const id of picked ?? []) marked.set(id, "picked");
        this.pressBands = bandMarks(drawn, marked, slab, [this.live]);
        this.liveBands.set(this.pressBands.marks);
        if (pressed === null) this.liveOf = of;
      }
    }
    const { settled, hidden } = this;
    this.onSurfaces.draw(
      [settled.bands, this.pressBands, settled.lines, this.live],
      [hidden.bands, null, hidden.lines, null],
      depth,
    );
  }
}
