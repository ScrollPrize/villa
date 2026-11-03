import { lock, saveFile } from "../storage.js";
import { zipped } from "../zip.js";
import { surfaceFiles } from "./mesh-edit/file.js";
import { extent, gathered, stepped } from "./mesh-edit/log.js";
import { MeshEditPanel } from "./mesh-edit/panel.js";
import { forgetLog, readLog, writeLog } from "./mesh-edit/storage.js";
import { gridEdges, grown, nearestVertex, onRim, onSurface, reach, smoothed } from "./mesh-edit/stroke.js";

const ICON = `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4h12v12H4zM4 10l6-3.5 6 3.5M10 4v12"/><circle cx="10" cy="6.5" r="1.6" fill="currentColor"/></svg>`;
const DEFAULT_RADIUS = 128;
const SMOOTH_STRENGTH = 0.25;
const MODES = [
  { kind: "move", name: "Move", code: "KeyM", rgb: [1, 0.55, 0.1] },
  { kind: "smooth", name: "Smooth", code: "KeyS", rgb: [0.3, 0.8, 1] },
  { kind: "erase", name: "Erase", code: "KeyE", rgb: [1, 0.2, 0.2] },
  { kind: "grow", name: "Grow", code: "KeyW", rgb: [0.35, 1, 0.35] },
];
const ELSEWHERE = "Another tab edits this surface; it shows here as that tab keeps it, to edit once it lets go of it.";
const NO_MARKS = {
  points_zyx_N3: new Float32Array(0),
  colors_N3: new Float32Array(0),
  edges_E2: new Uint32Array(0),
  selected: -1,
  depth: 0,
};

export default function createMeshEdit(viewer, entry) {
  const preview = viewer.addMarks();

  let ct = null;
  let surface = null;
  let drawn = null;
  let edits = [];
  let done = 0;
  let grab = null;
  let active = false;
  let mode = MODES[0];
  let busy = false;
  let reading = new AbortController();
  let owed = 0;
  let unkept = null;
  let keeping = Promise.resolve();
  let hold = null;
  let watching = false;

  function failed(error) {
    console.error(error);
    viewer.setStatus(`Mesh edits: ${error.message}`);
  }

  function describe(message = "") {
    let note = message;
    if (note === "" && surface !== null) {
      const [height, width] = surface.grid_HW;
      const log = `${done} edit${done === 1 ? "" : "s"} to undo, ${edits.length - done} to redo`;
      const stored = owed > 0 ? "being kept" : unkept === surface ? "not kept" : "kept";
      const kept = edits.length > 0 ? `, ${stored} in this browser` : "";
      note = `${height} × ${width} vertices, ${log}${kept}.`;
    }
    panel.show({
      busy,
      ct: ct !== null,
      edited: surface !== null,
      editable: surface !== null && !watching,
      active,
      mode,
      undo: done !== 0 && grab === null && !watching,
      redo: done !== edits.length && grab === null && !watching,
      note,
    });
  }

  function holdOn(volumeUrl, url) {
    if (hold?.volumeUrl === volumeUrl && hold.url === url) return hold;
    letGo(hold);
    hold = Object.assign(lock(`teca-mesh-edit ${volumeUrl} ${url}`), { volumeUrl, url });
    return hold;
  }

  function letGo(held) {
    if (held === null || held !== hold) return;
    hold = null;
    void keeping.then(held.release);
  }

  function keep(write) {
    const mine = surface;
    owed += 1;
    keeping = keeping
      .then(write)
      .then(
        () => {
          if (unkept === mine) unkept = null;
        },
        (error) => {
          unkept = mine;
          failed(error);
        },
      )
      .finally(() => {
        owed -= 1;
        describe();
      });
    return keeping;
  }

  function step(edit, forward, kept) {
    stepped(surface.points_zyx_HW3, edit, forward);
    if (drawn !== null) void drawn.changed(...extent(edit.indices, surface.grid_HW[1])).catch(failed);
    const mine = surface;
    const [log, count] = [edits, done];
    keep(() =>
      unkept === mine
        ? writeLog(mine.volumeUrl, mine.url, count, log, 0)
        : writeLog(mine.volumeUrl, mine.url, count, kept ? [edit] : [], count - 1),
    );
  }

  function show() {
    const points_zyx_N3 = mode.kind === "erase" ? grab.before_N3 : grab.after_N3;
    preview.set({
      points_zyx_N3,
      colors_N3: points_zyx_N3.map((_, cell) => mode.rgb[cell % 3]),
      edges_E2: grab.edges_E2,
      selected: grab.grabbed,
      // Half a grid step, so each line of the grid shows once where it crosses a pane.
      depth: 0.5 / Math.min(...surface.scale_UV),
    });
  }

  function write(values_N3) {
    for (const [position, index] of grab.indices.entries())
      surface.points_zyx_HW3.set(values_N3.subarray(3 * position, 3 * position + 3), 3 * index);
  }

  function moved(at) {
    const { from } = grab;
    if (mode.kind === "move") {
      const delta_ZYX =
        from.pane === "surface"
          ? from.normal_ZYX.map((value) => (value * (from.cell_HW[0] - at.cell_HW[0])) / surface.scale_UV[1])
          : at.point_ZYX.map((value, axis) => value - from.point_ZYX[axis]);
      grab.after_N3 = grab.before_N3.map(
        (value, cell) => value + grab.weights[Math.floor(cell / 3)] * delta_ZYX[cell % 3],
      );
    } else if (mode.kind === "smooth") {
      const { indices, weights, after_N3 } = grab;
      grab.after_N3 = smoothed(surface.points_zyx_HW3, surface.grid_HW, indices, weights, after_N3, SMOOTH_STRENGTH);
    } else return;
    show();
    if (from.pane !== "surface") return;
    write(grab.after_N3);
    grab.written = true;
    if (drawn !== null) void drawn.changed(...extent(grab.indices, surface.grid_HW[1])).catch(failed);
  }

  const tool = {
    radius: DEFAULT_RADIUS,
    label: null,
    surface: true,
    down(at) {
      const { points_zyx_HW3: points, grid_HW, scale_UV } = surface;
      let index = -1;
      if (at.pane !== "surface") {
        // A growing starts at the edge of a hole: the vertex nearest the pointer may be on another turn of the scroll.
        const admits =
          mode.kind === "grow" ? (cell) => onRim(points, grid_HW, cell) : (cell) => onSurface(points, cell);
        index = nearestVertex(points, at.point_ZYX, tool.radius, admits);
      } else if (at.edited === surface) {
        const [row, column] = at.cell_HW.map(Math.round);
        const cell = row * grid_HW[1] + column;
        const inside = row >= 0 && row < grid_HW[0] && column >= 0 && column < grid_HW[1];
        const grabbable = inside && onSurface(points, cell) && (mode.kind !== "move" || at.normal_ZYX !== null);
        if ((inside && mode.kind === "grow") || grabbable) index = cell;
      }
      if (index === -1) {
        describe(
          at.pane !== "surface"
            ? `No vertex of the surface${mode.kind === "grow" ? " beside a hole" : ""} is inside the ring.`
            : at.edited === surface
              ? "No cell of the surface is under the pointer."
              : "The Surface pane shows another surface than the one edited.",
        );
        return;
      }
      const { indices, weights, after_N3 } =
        mode.kind === "grow"
          ? { ...grown(points, grid_HW, scale_UV, index, tool.radius), weights: null }
          : { ...reach(points, grid_HW, scale_UV, index, tool.radius), after_N3: null };
      if (indices.length === 0) {
        describe("No cell off the surface inside the ring is next to two on it in a row or a column.");
        return;
      }
      const before_N3 = gathered(points, indices);
      grab = {
        from: at,
        indices,
        weights,
        before_N3,
        after_N3: mode.kind === "erase" ? new Float32Array(before_N3.length).fill(-1) : (after_N3 ?? before_N3),
        edges_E2: gridEdges(indices, grid_HW[1]),
        grabbed: indices.indexOf(index),
        written: false,
      };
      show();
      describe();
    },
    move(at) {
      if (grab !== null) moved(at);
    },
    up(at) {
      if (grab === null) return;
      if (mode.kind === "move") moved(at);
      const { indices, before_N3, after_N3, written } = grab;
      if (written) write(before_N3);
      grab = null;
      preview.set(NO_MARKS);
      if (after_N3.every((value, cell) => Object.is(value, before_N3[cell]))) {
        describe();
        return;
      }
      const edit = { indices, before_N3, after_N3 };
      edits = [...edits.slice(0, done), edit];
      done = edits.length;
      step(edit, true, true);
      describe();
    },
    putAway() {
      active = false;
      describe();
    },
  };

  function resize(radius) {
    tool.radius = panel.radius(radius);
    if (active) viewer.setTool(tool);
    describe();
  }

  async function open(url, kept) {
    const opened = ct;
    busy = true;
    if (drawn !== null && surface.url !== url) {
      void drawn.remove().catch(failed);
      surface = null;
      drawn = null;
      edits = [];
      done = 0;
    }
    const mine = holdOn(opened.rootUrl, url);
    describe(`Reading ${url}…`);
    let message = "";
    try {
      const watch = !(await mine.free) && !mine.granted;
      const log = await keeping.then(() => readLog(opened.rootUrl, url));
      if (kept && log.edits.length === 0) return;
      const read = { url, volumeUrl: opened.rootUrl, ...(await viewer.readSurface(url, reading.signal)) };
      if (ct !== opened) return;
      for (const edit of log.edits.slice(0, log.done)) stepped(read.points_zyx_HW3, edit, true);
      surface = read;
      edits = log.edits;
      done = log.done;
      watching = watch;
      const handle = await viewer.editSurface(url, surface);
      if (ct !== opened) return;
      drawn = handle;
      if (drawn === null) message = "The surface was not drawn; the status bar says why.";
      else if (watching) message = ELSEWHERE;
      else if (edits.length > 0) message = "Brought back the edits kept here before; Discard forgets them.";
      if (watching)
        void mine.ready.then(() => {
          if (hold === mine && surface?.url === url) void open(url, false);
        });
    } catch (error) {
      if (error.name !== "AbortError") {
        failed(error);
        message = error.message;
      }
    } finally {
      if (ct === opened) {
        busy = false;
        if (surface?.url !== url) letGo(mine);
        describe(message);
      }
    }
  }

  function listSurfaces() {
    panel.list(viewer.surfaces(), surface?.url);
    describe();
  }

  function edit(url) {
    if (url === "" || busy || ct === null) return;
    if (active) viewer.setTool(null);
    entry.save({ url });
    void open(url, false);
  }

  function take(next) {
    if (busy || surface === null || grab !== null || watching) return;
    if (active && mode === next) {
      viewer.setTool(null);
      return;
    }
    mode = next;
    active = true;
    viewer.setTool(tool);
    describe();
  }

  function undo() {
    if (busy || done === 0 || grab !== null || watching) return;
    done -= 1;
    step(edits[done], false, false);
    describe();
  }

  function redo() {
    if (busy || done === edits.length || grab !== null || watching) return;
    done += 1;
    step(edits[done - 1], true, false);
    describe();
  }

  async function save() {
    if (surface === null) return;
    const directory = surface.url.replace(/\/+$/, "").split("/").at(-1);
    const name = `${directory.replace(/\.tifxyz$/, "")}-edited`;
    saveFile(await zipped(surfaceFiles(surface, name), async () => {}, false), `${name}.zip`);
  }

  async function discard(chosen) {
    const url = surface?.url ?? chosen;
    if (busy || ct === null || url === "") return;
    const opened = ct;
    const mine = holdOn(opened.rootUrl, url);
    const elsewhere = !(await mine.free) && !mine.granted;
    if (ct !== opened) return;
    if (elsewhere || !window.confirm("Forget every edit of this surface, here and in what the browser keeps of it?")) {
      if (surface?.url !== url) letGo(mine);
      if (elsewhere) describe(ELSEWHERE);
      return;
    }
    if (active) viewer.setTool(null);
    const volumeUrl = opened.rootUrl;
    const handle = drawn;
    surface = null;
    drawn = null;
    edits = [];
    done = 0;
    entry.save(null);
    describe();
    void keep(() => Promise.all([handle?.remove(), forgetLog(volumeUrl, url)])).then(() =>
      describe("Forgot the edits; the surface is drawn from its files."),
    );
    letGo(mine);
  }

  function volumeChanged(opened) {
    if (active) viewer.setTool(null);
    reading.abort();
    reading = new AbortController();
    ct = opened;
    surface = null;
    drawn = null;
    edits = [];
    done = 0;
    grab = null;
    busy = false;
    letGo(hold);
    watching = false;
    preview.set(NO_MARKS);
    listSurfaces();
    const url = entry.saved?.url;
    if (url === undefined) return;
    if (viewer.surfaces().includes(url)) panel.choose(url);
    void open(url, true);
  }

  const panel = new MeshEditPanel(
    { tool: () => take(mode), take, undo, redo, save, edit, discard: (chosen) => void discard(chosen), resize },
    MODES,
    tool,
  );
  listSurfaces();
  return {
    panel: panel.element,
    place: { title: "Mesh edit", icon: ICON },
    keys: panel.keys(),
    volumeChanged,
    surfacesChanged: listSurfaces,
  };
}
