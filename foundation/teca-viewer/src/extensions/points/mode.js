import { moveClass, moveScribbles, nameClass, newClass } from "./classes.js";
import { joinRegions } from "./regions.js";
import { stepStroke, strokeCount, strokeOf } from "./skeleton.js";
import { targetLabel, targetTree, treeLabel, treeTarget } from "./targets.js";
import { counted } from "./view.js";

const PICK_FIRST = "Click a scribble to change its class; Escape goes back to Draw.";

export class Mode {
  constructor(aim) {
    this.aim = aim;
    this.editing = false;
    this.pick = null;
    this.pickedTree = null;
    this.whole = false;
    this.kept = [];
    this.joining = null;
    this.ids = null;
  }

  pickNode(skeleton, id, whole = false) {
    this.pick = skeleton.nodes.get(id);
    this.pickedTree = this.pick.tree;
    this.whole = whole;
    this.kept = [];
    this.aim.to(treeTarget(skeleton, this.pick.tree, this.aim.target));
  }

  stepTo(skeleton, id) {
    const kept = this.kept;
    this.pickNode(skeleton, id);
    this.kept = kept;
  }

  keep(skeleton) {
    if (this.pick === null) return PICK_FIRST;
    if (this.whole) return "The whole class is picked; Shift+E picks the scribble, which Enter then keeps.";
    const scribble = new Set(strokeOf(skeleton, this.pick.id));
    const kept = this.kept.filter((id) => !scribble.has(id));
    this.kept = kept.length < this.kept.length ? kept : [...kept, this.pick.id];
    return null;
  }

  togglePick(skeleton, id) {
    const before = this.whole || this.kept.length > 0 ? null : this.pick;
    this.stepTo(skeleton, id);
    if (before !== null) this.kept = [before.id];
    this.keep(skeleton);
  }

  picked() {
    return this.pick?.id ?? null;
  }

  scribbles() {
    return this.kept.length > 0 ? this.kept : [this.pick.id];
  }

  // The same array while the pick and `skeleton` stay as they are, so the band under it is built again only when they
  // change (marks.js draw).
  pickedIds(skeleton) {
    if (this.pick === null) return null;
    const scribble = this.whole ? null : (this.pick.stroke ?? this.pick.id);
    const of = [skeleton, skeleton.revision, this.pick.tree, scribble, this.kept.join()];
    if (!of.every((each, at) => each === this.ids?.of[at])) {
      const ids = this.whole
        ? [...skeleton.members.get(this.pick.tree)]
        : this.scribbles().flatMap((id) => strokeOf(skeleton, id));
      this.ids = { of, ids };
    }
    return this.ids.ids;
  }

  letGo() {
    this.pick = null;
    this.whole = false;
    this.kept = [];
  }

  follow(skeleton) {
    const [pick, selected] = [this.pick, skeleton.nodes.get(skeleton.selected)];
    this.kept = this.kept.filter((id) => skeleton.nodes.has(id));
    const stays =
      pick !== null &&
      selected !== undefined &&
      (selected.id === pick.id ||
        (selected.tree === pick.tree && (this.whole || (pick.stroke !== null && selected.stroke === pick.stroke))));
    if (!stays) {
      if (selected === undefined || this.kept.length === 0) this.letGo();
      else this.stepTo(skeleton, selected.id);
      return;
    }
    this.pick = selected;
    if (this.whole || !skeleton.trees.has(this.pickedTree)) this.pickedTree = selected.tree;
    this.aim.to(treeTarget(skeleton, selected.tree, this.aim.target));
  }

  setEditing(editing, skeleton, whole = false) {
    this.editing = editing;
    this.letGo();
    if (!editing) return;
    this.joining = null;
    if (skeleton.selected !== null) this.pickNode(skeleton, skeleton.selected, whole);
  }

  switchMode(skeleton) {
    this.setEditing(!this.editing, skeleton);
  }

  switchPick(skeleton) {
    this.setEditing(true, skeleton, !this.whole);
  }

  release() {
    this.joining = null;
    if (this.aim.target.mark !== null) this.aim.setMark(null);
    this.setEditing(false);
  }

  volumeChanged() {
    this.joining = null;
    this.letGo();
  }

  move(skeleton, next) {
    if (this.pick === null) return { patch: null, said: PICK_FIRST };
    if (this.whole) return moveClass(skeleton, this.pick.tree, next);
    return moveScribbles(skeleton, this.scribbles(), next);
  }

  makeClass(skeleton) {
    if (this.editing && this.pick === null) return { patch: null, said: PICK_FIRST };
    if (this.whole)
      return { patch: null, said: `${treeLabel(skeleton, this.pick.tree)} is a class of its own already.` };
    return newClass(skeleton, this.pick === null ? [] : this.scribbles());
  }

  name(skeleton, id, name) {
    if (this.editing && this.pick === null) return { patch: null, said: PICK_FIRST };
    if (this.kept.length > 1) {
      const kept = counted(this.kept.length, "scribble");
      return { patch: null, said: `With ${kept} kept, C gives them a class of their own; a click picks one.` };
    }
    const scribbles = this.pick === null || this.whole ? 1 : strokeCount(skeleton, id);
    if (scribbles > 1) {
      const what = "C gives this scribble a class of its own, or Escape and Name renames them all";
      return { patch: null, said: `This class holds ${scribbles} scribbles; ${what}.` };
    }
    return nameClass(skeleton, id, name);
  }

  scribbleStep(skeleton, step) {
    const picked = this.pick !== null;
    const tree = picked ? this.pickedTree : targetTree(skeleton, this.aim.target);
    const inClass = picked || skeleton.nodes.get(skeleton.selected)?.tree === tree;
    const from = inClass ? skeleton.selected : null;
    if (step === 0 && from !== null) return { node: from, on: from };
    const ids = tree === null ? null : stepStroke(skeleton, tree, from, step === 0 ? -1 : step);
    if (ids !== null) return { node: ids[0], on: ids[Math.floor(ids.length / 2)] };
    const place = tree === null ? this.aim.target : treeTarget(skeleton, tree, this.aim.target);
    const label = targetLabel(skeleton, { ...place, mark: null });
    if (from === null) return { node: null, said: `${label} has no nodes yet.` };
    return { node: null, said: `${label} has no scribble ${step > 0 ? "after" : "before"} this one.` };
  }

  join(skeleton) {
    const winding = this.aim.target.winding;
    if (this.joining === null && winding === null)
      return { patch: null, said: "W? is in no region; choose a winding to join." };
    if (this.joining === null || this.joining === winding) {
      this.joining = this.joining === null ? winding : null;
      return { patch: null, said: "" };
    }
    const joined = joinRegions(skeleton, this.joining, winding);
    if (joined.patch !== null) this.joining = null;
    return joined;
  }
}
