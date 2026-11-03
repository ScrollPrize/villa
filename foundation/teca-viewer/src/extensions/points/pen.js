import { chordSteps } from "../../math.js";
import { inside } from "./nml.js";
import { markComment } from "./runs.js";
import { addNodes, apply, commit, emptyPatch, moveNode, nearestNode, record, redo, undo } from "./skeleton.js";
import { makeTarget, targetTree } from "./targets.js";

const REACH = 12;
const SLOP = 4;
const SURFACE_SAG = 1;
// At 8 a cell, 2.5 voxels apart at 20 voxels a cell, nodes stay within 1.03 voxels of the surface; at 1, 2.19.
const SURFACE_SAMPLES = 8;

export function spaced(from_ZYX, to_ZYX, carried, spacing) {
  const length = Math.hypot(...to_ZYX.map((value, axis) => value - from_ZYX[axis]));
  const points_ZYX = [];
  let offset = carried;
  for (; offset <= length; offset += spacing)
    points_ZYX.push(from_ZYX.map((value, axis) => value + ((to_ZYX[axis] - value) * offset) / length));
  return { points_ZYX, carried: offset - length };
}

export class Pen {
  constructor(points) {
    this.points = points;
    this.radius = null;
    this.label = null;
    this.surface = true;
    this.drag = null;
  }

  show(label) {
    this.label = label;
  }

  busy() {
    return this.drag !== null;
  }

  pressed() {
    const drag = this.drag;
    return drag?.moved && drag.skeleton === this.points.skeleton() ? drag.ids : null;
  }

  commit(patch) {
    if (this.busy()) return false;
    commit(this.points.skeleton(), patch);
    this.points.edited();
    return true;
  }

  undo() {
    if (this.busy() || !undo(this.points.skeleton())) return false;
    this.points.edited();
    return true;
  }

  redo() {
    if (this.busy() || !redo(this.points.skeleton())) return false;
    this.points.edited();
    return true;
  }

  place(drag, points_ZYX, view) {
    const skeleton = drag.skeleton;
    const within = points_ZYX.filter((point_ZYX) => inside(this.points.level(), point_ZYX));
    if (within.length === 0) return;
    if (drag.tree === null) {
      drag.tree = targetTree(skeleton, this.points.target());
      if (drag.tree === null) {
        const made = makeTarget(skeleton, this.points.target());
        apply(skeleton, made.patch);
        drag.patch.groups.push(...made.patch.groups);
        drag.patch.trees.push(...made.patch.trees);
        drag.tree = made.tree;
      }
    }
    const selected = skeleton.selected === null ? null : skeleton.nodes.get(skeleton.selected);
    const patch = addNodes(skeleton, drag.tree, within, {
      from: this.points.joined() && selected?.tree === drag.tree ? selected.id : null,
      connected: this.points.joined(),
      stroke: drag.stroke,
      comment: markComment(this.points.target().mark),
      view,
    });
    apply(skeleton, patch);
    for (const [id] of patch.nodes) drag.ids.add(id);
    drag.patch.nodes.push(...patch.nodes);
    drag.patch.edges.push(...patch.edges);
    drag.patch.selected[1] = skeleton.selected;
  }

  placing(drag, stroke) {
    const selected = drag.skeleton.selected;
    Object.assign(drag, { stroke, tree: null, patch: { ...emptyPatch(), selected: [selected, selected] } });
  }

  down(at, ctrl) {
    if (at.point_ZYX === null) return;
    const points = this.points;
    const skeleton = points.skeleton();
    const reach = REACH * at.pixel;
    const node =
      at.pane === "surface"
        ? points.surfaceNode(at.cell_HW, reach)
        : nearestNode(skeleton, at.point_ZYX, at.normal_ZYX, points.depth(), reach, points.shown());
    const ids = new Set(node === null ? [] : [node]);
    const editing = points.editing();
    this.drag = {
      skeleton,
      node,
      ids,
      start_ZYX: at.point_ZYX,
      cell_HW: at.cell_HW,
      moved: false,
      view: { pane: at.pane, normal_ZYX: [...at.normal_ZYX] },
      editing,
    };
    if (!editing && node !== null) this.drag.before = skeleton.nodes.get(node);
    if (editing || node !== null) points.selected(node, editing && ctrl);
  }

  move(at) {
    const drag = this.drag;
    if (drag === null || drag.editing || at.point_ZYX === null) return;
    const far = Math.hypot(...at.point_ZYX.map((value, axis) => value - drag.start_ZYX[axis])) > SLOP * at.pixel;
    if (!drag.moved && !far) return;
    const skeleton = drag.skeleton;
    if (drag.node !== null) {
      drag.moved = true;
      if (!inside(this.points.level(), at.point_ZYX)) return;
      apply(skeleton, moveNode(skeleton, drag.node, at.point_ZYX));
      this.points.drawn();
      return;
    }
    if (!drag.moved) {
      drag.moved = true;
      this.placing(drag, crypto.randomUUID());
      this.place(drag, [drag.start_ZYX], drag.view);
      Object.assign(drag, { last_ZYX: drag.start_ZYX, carried: this.points.spacing() });
    }
    const ends = at.pane === "surface" ? this.along(drag.cell_HW, drag.last_ZYX, at) : [at];
    const points_ZYX = [];
    for (const { point_ZYX } of ends) {
      const next = spaced(drag.last_ZYX, point_ZYX, drag.carried, this.points.spacing());
      points_ZYX.push(...next.points_ZYX);
      Object.assign(drag, { last_ZYX: point_ZYX, carried: next.carried });
    }
    drag.cell_HW = at.cell_HW;
    this.place(drag, points_ZYX, { pane: at.pane, normal_ZYX: [...at.normal_ZYX] });
    this.points.drawn();
  }

  along(from_HW, from_ZYX, at) {
    const surfaceAt = this.points.surfacePoints();
    const cellAt = (t) => from_HW.map((value, axis) => value + (at.cell_HW[axis] - value) * t);
    const samples = SURFACE_SAMPLES * Math.hypot(...at.cell_HW.map((value, axis) => value - from_HW[axis]));
    return chordSteps((t) => surfaceAt(cellAt(t))?.point_ZYX ?? null, from_ZYX, at.point_ZYX, SURFACE_SAG, samples);
  }

  up() {
    this.end(true);
  }

  end(release) {
    const drag = this.drag;
    if (drag === null) return;
    this.drag = null;
    const skeleton = drag.skeleton;
    if (drag.editing) {
      if (release && drag.node !== null) this.points.released(drag.node, drag.view.pane);
      return;
    }
    if (drag.node !== null) {
      if (drag.moved)
        record(skeleton, { ...emptyPatch(), nodes: [[drag.node, drag.before, skeleton.nodes.get(drag.node)]] });
      else this.points.clicked(drag.node);
    } else if (drag.moved) {
      if (drag.patch.nodes.length === 0) return;
      record(skeleton, drag.patch);
    } else {
      this.placing(drag, null);
      this.place(drag, [drag.start_ZYX], drag.view);
      if (drag.patch.nodes.length === 0) {
        this.points.said("That point is outside the CT.");
        return;
      }
      record(skeleton, drag.patch);
    }
    if (drag.node === null || drag.moved) this.points.edited();
    if (release) this.points.released(drag.node ?? skeleton.selected, drag.view.pane);
  }

  putAway() {
    if (this.drag?.moved) this.end(false);
    this.drag = null;
    this.points.putAway();
  }
}
