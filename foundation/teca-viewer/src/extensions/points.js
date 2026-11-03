import { ctName, saveFile } from "../storage.js";
import { Aim } from "./points/aim.js";
import { TreeMarks } from "./points/marks.js";
import { Mode } from "./points/mode.js";
import { readNml, readSaved, savedZip, writeNml } from "./points/nml.js";
import { ICON, PointsPanel } from "./points/panel.js";
import { Pen } from "./points/pen.js";
import { insertWinding, newRegion } from "./points/regions.js";
import { commentTags, stepRun, tagsComment } from "./points/runs.js";
import {
  deleteNode,
  deleteStroke,
  deleteTree,
  emptySkeleton,
  flagBranchpoint,
  jumpBack,
  lastNode,
  selectNode,
  setComment,
  stepNode,
  stepStroke,
} from "./points/skeleton.js";
import { Keeping } from "./points/storage.js";
import {
  classNeighbour,
  classStep,
  newFiber,
  otherSide,
  pickerTarget,
  switchAir,
  targetTree,
  treeTarget,
  windingStep,
  withRole,
} from "./points/targets.js";
import { counted, pointsView } from "./points/view.js";

const DEFAULT_DEPTH = 0;

export default function createPoints(viewer, entry) {
  const marks = new TreeMarks(viewer, failed);
  let ct = null;
  let skeleton = null;
  let watched = null;
  let active = false;
  let unsaved = false;
  let depth = DEFAULT_DEPTH;
  let filed = entry.opened;
  const empty = emptySkeleton();
  const keeping = new Keeping(failed, () => describe());
  const aim = new Aim(entry.saved, entry.save, failed);
  const mode = new Mode(aim);
  const pen = new Pen({
    skeleton: () => skeleton,
    level: () => ct.scales[0],
    target: () => aim.target,
    spacing: () => aim.spacing(),
    depth: () => depth,
    surfaceNode: (cell_HW, reach) => marks.onSurfaces.nearest(cell_HW, reach),
    joined: () => panel.joined(),
    surfacePoints: viewer.surfacePoints,
    editing: () => mode.editing,
    shown: () => marks.shown,
    drawn: () => draw(false),
    edited,
    selected: (id, adding) => {
      if (adding) {
        if (id === null) return;
        mode.togglePick(skeleton, id);
        selectNode(skeleton, id);
      } else {
        selectNode(skeleton, id);
        if (mode.editing && id !== null) mode.pickNode(skeleton, id);
      }
      changed();
    },
    clicked: (id) => (aim.to(treeTarget(skeleton, skeleton.nodes.get(id).tree, aim.target)) ? changed() : describe()),
    released: (id, pane) => {
      if (aim.following) centre(id, pane);
    },
    said: (text) => describe(text),
    putAway: () => {
      active = false;
      mode.setEditing(false, skeleton);
      changed();
    },
  });

  function failed(error) {
    console.error(error);
    viewer.setStatus(`Points: ${error.message}`);
  }

  function drawn() {
    return skeleton ?? watched ?? empty;
  }

  function describe(message = "") {
    const kept = skeleton === null ? null : keeping.state(ct.rootUrl);
    const clear = ct !== null && keeping.holds(ct.rootUrl);
    const shown = { skeleton, watched, aim, mode, active, unsaved, kept, clear, depth, runs: marks.runs };
    const { view, label } = pointsView({ ...shown, drawn: drawn(), labelColor: viewer.labelColor }, message);
    panel.show(view);
    pen.show(label);
    if (active) viewer.setTool(pen);
  }

  function draw(rejoin = true) {
    const bright = skeleton === null ? null : targetTree(skeleton, aim.target);
    const how = {
      bright,
      only: aim.only,
      winding: aim.target.winding,
      depth,
      picked: mode.pickedIds(skeleton),
      linked: aim.linked,
    };
    marks.draw(drawn(), how, pen.pressed(), rejoin);
  }

  function nml() {
    return skeleton.trees.size === 0 ? null : writeNml(skeleton, ctName(ct.rootUrl), ct.scales[0]);
  }

  function changed(message = "") {
    aim.follow(skeleton);
    mode.follow(skeleton);
    draw();
    describe(message);
  }

  function edited() {
    unsaved = true;
    keeping.soon(ct.rootUrl, nml);
    changed();
  }

  function aimAt(next) {
    if (pen.busy()) return;
    if (next === null) describe("The winding is unknown; Winding's ? pressed again sets W0.");
    else if (!mode.editing) {
      if (aim.to(next)) changed();
      else describe();
    } else {
      const moved = mode.move(skeleton, next);
      if (moved.patch === null) describe(moved.said);
      else if (pen.commit(moved.patch)) viewer.setStatus(moved.said);
    }
  }

  function centre(id, pane) {
    const point_ZYX = drawn().nodes.get(id).point_ZYX;
    viewer.focus(point_ZYX, { surface: pane === "surface" || marks.onSurfaces.near(point_ZYX) });
  }

  function select(id, on = id, pick = false, surface = false) {
    if (pen.busy()) return;
    selectNode(skeleton, id);
    if (pick) mode.stepTo(skeleton, id);
    changed();
    centre(on, surface ? "surface" : null);
  }

  function toRun(run) {
    if (run === null) describe("No marked run lies that way.");
    else if (skeleton === null) centre(run.ids[Math.floor(run.ids.length / 2)], null);
    else select(run.ids[0], run.ids[Math.floor(run.ids.length / 2)]);
  }

  function editSelected(patchOf) {
    if (skeleton.selected !== null) pen.commit(patchOf(skeleton.selected));
  }

  function removeClass(id) {
    if (pen.busy()) return;
    const next = classNeighbour(skeleton, id);
    const target = next === null ? aim.target : treeTarget(skeleton, next, aim.target);
    const patch = { ...deleteTree(skeleton, id, next), target: [aim.target, target] };
    aim.to(target);
    pen.commit(patch);
  }

  function changeMode(change) {
    if (pen.busy()) return;
    change();
    changed();
  }

  function toDraw() {
    if (!mode.editing) return;
    mode.setEditing(false, skeleton);
    changed();
  }

  function join() {
    if (pen.busy()) {
      describe("Join waits for the press under way to end.");
      return;
    }
    toDraw();
    const joined = mode.join(skeleton);
    if (joined.patch === null) describe(joined.said);
    else if (pen.commit(joined.patch)) viewer.setStatus(joined.said);
  }

  function history(forward) {
    const patch = forward ? skeleton.undone.at(-1) : skeleton.done.at(-1);
    if (pen.busy() || patch === undefined) return;
    if (patch.target !== undefined) aim.to(patch.target[forward ? 1 : 0]);
    if (forward) pen.redo();
    else pen.undo();
  }

  function replace(read, name) {
    if (
      skeleton.nodes.size > 0 &&
      !window.confirm(`Replace the ${counted(skeleton.nodes.size, "node")} here with ${name}?`)
    )
      return;
    skeleton = read;
    mode.letGo();
    keeping.now(ct.rootUrl, nml);
    unsaved = false;
    changed();
  }

  async function open(chosen) {
    const opened = ct;
    try {
      const bytes = new Uint8Array(await chosen.arrayBuffer());
      const read = await readSaved(bytes, chosen.name, opened.scales, opened.metadata.physicalUnits_ZYX);
      if (ct === opened && skeleton !== null) replace(read, chosen.name);
    } catch (error) {
      failed(error);
      describe(error.message);
    }
  }

  const panel = new PointsPanel(
    {
      tool: () => {
        if (active) viewer.setTool(null);
        else if (skeleton !== null) {
          active = true;
          describe();
        }
      },
      mode: (editing) => changeMode(() => mode.setEditing(editing, skeleton)),
      switchMode: () => changeMode(() => mode.switchMode(skeleton)),
      switchPick: () => changeMode(() => mode.switchPick(skeleton)),
      undo: () => history(false),
      redo: () => history(true),
      insert: () => {
        if (pen.busy()) return;
        toDraw();
        const { patch, said } = insertWinding(skeleton, aim.target.winding);
        if (patch === null) describe(said);
        else if (patch.groups.length > 0) pen.commit(patch);
      },
      region: () => {
        if (!pen.busy()) toDraw();
        aimAt(newRegion(skeleton, aim.target));
      },
      join,
      newClass: () => {
        const made = mode.makeClass(skeleton);
        if (made.patch === null) describe(made.said);
        else if (!pen.busy()) {
          aim.to({ ...aim.target, tree: made.tree });
          pen.commit(made.patch);
          if (made.said !== null) viewer.setStatus(made.said);
        }
      },
      deleteClass: () => removeClass(targetTree(skeleton, aim.target)),
      save: async () => {
        const name = ctName(ct.rootUrl);
        const zip = savedZip(skeleton, name, ct.scales, ct.metadata.physicalUnits_ZYX);
        unsaved = false;
        describe();
        saveFile(await zip, `${name}-points.zip`);
      },
      open: (chosen) => void open(chosen),
      clear: () => {
        if (!window.confirm("Forget every node on this CT, here and in what the browser keeps of it?")) return;
        if (active) viewer.setTool(null);
        skeleton = emptySkeleton();
        unsaved = false;
        keeping.now(ct.rootUrl, nml);
        changed();
      },
      choose: (part) => aimAt(pickerTarget(skeleton, aim.target, part, mode.whole)),
      only: (checked) => {
        aim.showOnly(checked);
        draw();
        describe();
      },
      follow: (checked) => {
        aim.setFollowing(checked);
        describe();
      },
      links: (checked) => {
        aim.setLinked(checked);
        draw();
        describe();
      },
      comment: (text) =>
        editSelected((id) =>
          setComment(skeleton, id, tagsComment({ ...commentTags(skeleton.nodes.get(id).comment), text })),
        ),
      chooseClass: (id) => aimAt(treeTarget(skeleton, id, aim.target)),
      name: (name) => {
        const tree = targetTree(skeleton, aim.target);
        if (tree === null || name === skeleton.trees.get(tree).name || pen.busy()) return false;
        const { patch, said } = mode.name(skeleton, tree, name);
        if (patch === null) describe(said);
        else if (pen.commit(patch) && aim.to(treeTarget(skeleton, tree, aim.target))) changed();
        return patch !== null;
      },
      spacing: (spacing) => aim.setSpacing(spacing),
      depth: (value) => {
        depth = value >= 0 ? value : DEFAULT_DEPTH;
        draw();
        return depth;
      },
      run: (index) => toRun(marks.runs[index]),
      mark: (mark) => {
        if (pen.busy()) return;
        aim.setMark(mark === aim.target.mark ? null : mark);
        describe();
      },
      release: () => changeMode(() => mode.release()),
      branchpoint: () => editSelected((id) => flagBranchpoint(skeleton, id)),
      jump: () => {
        const patch = jumpBack(skeleton);
        if (patch !== null && pen.commit(patch)) centre(skeleton.selected, null);
      },
      step: (places) => {
        if (skeleton.selected !== null) select(stepNode(skeleton, skeleton.selected, places));
      },
      stepRun: (step) => toRun(stepRun(marks.runs, skeleton, skeleton.selected, step)),
      stepScribble: (step) => {
        const went = mode.scribbleStep(skeleton, step);
        if (went.node === null) describe(went.said);
        else select(went.node, went.on, mode.editing && went.node !== skeleton.selected, step === 0);
      },
      stepClass: (step) => {
        const next = classStep(skeleton, targetTree(skeleton, aim.target), step);
        if (next === null) {
          describe(`No class with nodes lies ${step > 0 ? "after" : "before"} this one in the Class list.`);
          return;
        }
        if (!mode.editing) aim.to(treeTarget(skeleton, next, aim.target));
        const ids = stepStroke(skeleton, next, null, -1);
        select(ids[0], ids[Math.floor(ids.length / 2)], mode.editing);
      },
      keep: () => {
        const why = mode.editing ? mode.keep(skeleton) : "Enter keeps a scribble in Edit, which E goes to.";
        if (why === null) changed();
        else describe(why);
      },
      remove: () => {
        const tree = targetTree(skeleton, aim.target);
        if (skeleton.selected !== null) pen.commit(deleteStroke(skeleton, skeleton.selected));
        else if (tree !== null && lastNode(skeleton, tree) === null) removeClass(tree);
      },
      removeNode: () => editSelected((id) => deleteNode(skeleton, id)),
      previous: () => aimAt({ ...aim.previous, mark: aim.target.mark }),
      role: (role) => aimAt(withRole(skeleton, aim.target, role)),
      newFiber: () => aimAt(newFiber(skeleton, aim.target)),
      otherSide: () => aimAt(otherSide(skeleton, aim.target, mode.whole)),
      winding: (step) => aimAt(windingStep(skeleton, aim.target, step, mode.whole)),
      air: () => aimAt(switchAir(aim.target)),
    },
    pen,
  );

  async function restore(opened) {
    await keeping.hold(opened.rootUrl, async () => {
      const text = await keeping.read(opened.rootUrl);
      if (ct !== opened) return;
      watched = text === null ? emptySkeleton() : readNml(text, opened.scales[0]);
      draw();
      describe();
    });
    if (ct !== opened) return;
    watched = null;
    const text = await keeping.read(opened.rootUrl);
    if (ct !== opened || skeleton !== null) return;
    skeleton = text === null ? emptySkeleton() : readNml(text, opened.scales[0]);
    changed(text === null ? "" : `Brought back ${counted(skeleton.nodes.size, "node")} placed here before.`);
    const fromFile = filed;
    filed = null;
    if (fromFile === null || fromFile.nml === text) return;
    if (fromFile.ct !== opened.rootUrl || typeof fromFile.nml !== "string")
      throw new Error(
        `The session file's nodes are {ct, nml} of ${opened.rootUrl}, not ${JSON.stringify(fromFile).slice(0, 120)}.`,
      );
    replace(readNml(fromFile.nml, opened.scales[0]), "the session file's nodes");
  }

  async function kept() {
    if (ct === null) return null;
    const text = skeleton === null ? await keeping.read(ct.rootUrl) : nml();
    return text === null ? null : { ct: ct.rootUrl, nml: text };
  }

  function volumeChanged(opened) {
    if (active) viewer.setTool(null);
    keeping.now();
    ct = opened;
    skeleton = null;
    watched = null;
    unsaved = false;
    mode.volumeChanged();
    marks.onSurfaces.volumeChanged();
    draw();
    describe();
    void restore(opened).catch((error) => {
      failed(error);
      describe(error.message);
    });
  }

  describe();
  return {
    panel: panel.element,
    place: { title: "Points", icon: ICON },
    keys: panel.keys(),
    volumeChanged,
    surfacesChanged: () => marks.onSurfaces.changed(),
    kept,
  };
}
