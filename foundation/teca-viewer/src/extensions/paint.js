import { ctName, saveFile } from "../storage.js";
import { zipped } from "../zip.js";
import { Brush } from "./paint/brush.js";
import { Gate } from "./paint/gate.js";
import { Holding } from "./paint/holding.js";
import { AXES, DEFAULT_RADIUS, ICON, MAX_RADIUS, paintPanel } from "./paint/panel.js";
import { Spares } from "./paint/spares.js";
import { Keeping } from "./paint/storage.js";

// uint8 labels; 0 is unlabelled.
const MAX_CLASS = 255;
const DEFAULT_CLASSES = 2;
const RESIZE = 1.25;
const SWEEPERS = 3;

export default function createPaint(viewer) {
  const panel = paintPanel();
  const element = (id) => panel.querySelector(`#paint-${id}`);
  const classList = element("classes");
  const [radiusInput, radiusNumber, axesInput, radiiRow, eraseInput, fillInput, gateInput, gateValue, fileInput] = [
    "radius",
    "radius-number",
    "axes",
    "radii",
    "erase",
    "fill",
    "gate",
    "gate-value",
    "file",
  ].map(element);
  const axisInputs = AXES.map((axis) => element(`radius-${axis}`));
  const [toolButton, undoButton, redoButton, clearButton, saveButton, loadButton] =
    panel.querySelectorAll(".button-row button");
  const note = element("note");
  const hereButton = element("here");

  let active = false;
  let current = 1;
  let classes = DEFAULT_CLASSES;

  function failed(error) {
    console.error(error);
    viewer.setStatus(`Paint: ${error.message}`);
  }

  // Paint's IndexedDB writes are made by a worker, as the page's own thread hears of IndexedDB only after its next
  // frame once an input comes, and its reads by others, which gunzip the chunks.
  const storageWorker = new URL("./paint/storage.worker.js", import.meta.url);
  const writer = new Worker(storageWorker, { type: "module" });
  const keeping = new Keeping(
    (message, transfer) => writer.postMessage(message, transfer),
    (message, receive) => {
      const reader = new Worker(storageWorker, { type: "module" });
      reader.onmessage = ({ data }) => receive(data);
      reader.onerror = (event) => receive({ error: event.message });
      reader.postMessage(message);
      return () => reader.terminate();
    },
    failed,
    () => describe(),
  );
  writer.onmessage = ({ data }) => keeping.answered(data);
  writer.onerror = (event) => {
    const error = new Error(`Paint's storage worker: ${event.message}`);
    failed(error);
    keeping.stop(error);
  };

  const sweepWorker = new URL("./paint/surface-sweep.worker.js", import.meta.url);
  const sweepers = Array.from({ length: SWEEPERS }, () => {
    const worker = new Worker(sweepWorker, { type: "module" });
    const replies = [];
    let stopped = null;
    worker.onmessage = ({ data }) => replies.shift().resolve(data);
    worker.onerror = (event) => {
      worker.terminate();
      stopped = new Error(`Paint's sweep worker: ${event.message}`);
      for (const reply of replies.splice(0)) reply.reject(stopped);
    };
    return (job) => {
      const reply = Promise.withResolvers();
      if (stopped !== null) reply.reject(stopped);
      else {
        replies.push(reply);
        worker.postMessage(job, [job.points_N9.buffer]);
      }
      return reply.promise;
    };
  });
  const sparing = new Worker(new URL("./paint/spares.worker.js", import.meta.url), { type: "module" });
  const spares = new Spares((count, length) => sparing.postMessage({ count, length }));
  sparing.onmessage = ({ data }) => spares.received(data.length, data.buffers);
  sparing.onerror = (event) => failed(new Error(`Paint's spares worker: ${event.message}`));
  const brush = new Brush(
    [DEFAULT_RADIUS, DEFAULT_RADIUS, DEFAULT_RADIUS],
    failed,
    viewer.setStatus,
    viewer.surfacePoints,
    sweepers,
  );
  const holding = new Holding(viewer, keeping, brush, (length) => spares.zeroed(length), {
    describe,
    failed,
    brushSet: () => active,
    setBrush: () => {
      active = true;
      viewer.setTool(tool);
    },
    raise: (largest) => {
      classes = Math.min(MAX_CLASS, Math.max(classes, largest));
    },
    listClasses,
  });

  function describe(message = "") {
    const { ct, labels, busy, restoring, unrestored, waiting } = holding;
    for (const [index, button] of [...classList.children].slice(0, classes).entries())
      button.classList.toggle("active", index + 1 === current && !eraseInput.checked);
    toolButton.classList.toggle("active", active);
    toolButton.disabled = busy || restoring || ct === null;
    undoButton.disabled = busy || labels === null || labels.history.done.length === 0;
    redoButton.disabled = busy || labels === null || labels.history.undone.length === 0;
    clearButton.disabled = busy || restoring || labels === null;
    saveButton.disabled = busy || restoring || labels === null || labels.unsaved.size === 0;
    loadButton.disabled = busy || restoring || ct === null;
    hereButton.hidden = waiting === null;
    if (document.activeElement !== radiusNumber) radiusNumber.value = radiusInput.value;
    radiiRow.hidden = !axesInput.checked;
    for (const [axis, input] of axisInputs.entries())
      if (document.activeElement !== input) input.value = String(brush.radii_ZYX[axis]);
    if (message !== "") note.textContent = message;
    else if (unrestored !== null) note.textContent = unrestored;
    else if (waiting !== null) note.textContent = waiting.note;
    else if (labels !== null && labels.held.size > 0) {
      const { chunks_ZYX } = labels.volume.scales[0];
      const mib = Math.ceil((labels.held.size * chunks_ZYX[0] * chunks_ZYX[1] * chunks_ZYX[2]) / 2 ** 20);
      const unsaved = labels.unsaved.size > 0 ? `, ${labels.unsaved.size} not saved` : "";
      const held = `${labels.held.size} chunk${labels.held.size === 1 ? "" : "s"}`;
      note.textContent = `${held}, ${mib} MiB${unsaved}, ${keeping.state(labels.url)} in this browser.`;
    } else note.textContent = active ? "Drag on a slice pane or the Surface to paint; hold Space to pan." : "";
  }

  function choose(value) {
    current = value;
    eraseInput.checked = false;
    describe();
  }

  function listClasses() {
    const buttons = Array.from({ length: classes }, (_, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.id = `paint-class-${index + 1}`;
      const swatch = document.createElement("span");
      swatch.className = "overlay-swatch";
      const color_RGB = viewer.labelColor(index + 1).map((value) => Math.round(value * 255));
      swatch.style.background = `rgb(${color_RGB.join()})`;
      button.append(swatch, ` ${index + 1}`);
      button.addEventListener("click", () => choose(index + 1));
      return button;
    });
    const add = document.createElement("button");
    add.type = "button";
    add.id = "paint-class-add";
    add.textContent = "+";
    add.disabled = classes === MAX_CLASS;
    add.addEventListener("click", () => {
      classes += 1;
      listClasses();
      choose(classes);
    });
    classList.replaceChildren(...buttons, add);
    describe();
  }

  // The gate on the CT at the panel's threshold, or null while it is off. Once another volume opens, the chunks the
  // panes hold are its, so the gate reads its own.
  function gate() {
    if (!gateInput.checked) return null;
    const threshold = Math.min(255, Math.max(0, Math.round(Number(gateValue.value))));
    const gated = holding.ct;
    const held = (level, chunk_ZYX) => (holding.ct === gated ? viewer.heldCT(level, chunk_ZYX) : null);
    return new Gate(gated, held, viewer.openVolume, threshold);
  }

  const tool = {
    radius: DEFAULT_RADIUS,
    label: null,
    surface: true,
    down(at) {
      const value = eraseInput.checked ? 0 : current;
      if (!fillInput.checked) {
        keeping.owe(brush.begin(holding.labels, gate(), value));
        brush.to(at, false);
      } else if (at.pane === "surface") viewer.setStatus("Paint fills on the slice panes, not on the Surface.");
      else {
        const mine = holding.labels;
        keeping.owe(brush.fill(mine, gate(), at, value).then(() => describe(), failed));
        holding.keep(mine);
      }
    },
    move(at) {
      if (brush.stroke !== null) brush.to(at, false);
    },
    up(at) {
      if (brush.stroke === null) return;
      const mine = holding.labels;
      brush.to(at, true);
      void brush.end().then(() => describe(), failed);
      holding.keep(mine);
    },
    putAway() {
      active = false;
      describe();
    },
  };

  function resize(radii_ZYX) {
    const whole_ZYX = radii_ZYX.map((radius) => Math.min(MAX_RADIUS, Math.max(1, Math.round(radius))));
    brush.resize(axesInput.checked ? whole_ZYX : [whole_ZYX[0], whole_ZYX[0], whole_ZYX[0]]);
    radiusInput.value = String(Math.max(...brush.radii_ZYX));
    tool.radius = Math.max(...brush.radii_ZYX);
    if (active) viewer.setTool(tool);
    describe();
  }

  function scale(by) {
    resize(
      brush.radii_ZYX.map((radius) =>
        by > 1 ? Math.max(radius + 1, Math.round(radius * by)) : Math.min(radius - 1, Math.round(radius * by)),
      ),
    );
  }

  function turn(back) {
    const mine = holding.labels;
    if (holding.busy || mine === null) return;
    const asked = Promise.withResolvers();
    keeping.owe(asked.promise);
    void holding
      .transfer(async () => {
        const turned = mine.turn(back, brush.pause);
        asked.resolve();
        if (!(await turned)) return;
        brush.follow(mine);
        holding.keep(mine);
      })
      .finally(asked.resolve);
  }

  toolButton.addEventListener("click", () => {
    if (holding.busy || holding.restoring || holding.ct === null) return;
    if (active) {
      viewer.setTool(null);
      return;
    }
    try {
      holding.make();
    } catch (error) {
      describe(error.message);
      return;
    }
    active = true;
    viewer.setTool(tool);
    describe();
  });

  undoButton.addEventListener("click", () => turn(true));

  redoButton.addEventListener("click", () => turn(false));

  clearButton.addEventListener("click", () => {
    if (holding.busy || holding.labels === null) return;
    if (!window.confirm("Forget everything painted on this CT, here and in what the browser keeps of it?")) return;
    if (active) viewer.setTool(null);
    holding.clear();
    describe();
  });

  saveButton.addEventListener("click", () => {
    const mine = holding.labels;
    if (mine === null) return;
    void holding.transfer(async () => {
      const name = ctName(mine.url);
      await holding.packing;
      saveFile(await zipped(await mine.savedEntries(), brush.pause, false), `${name}-labels.zarr.zip`);
    });
  });

  hereButton.addEventListener("click", () => holding.take());

  loadButton.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", () => {
    const [chosen] = fileInput.files;
    // So the same file can be chosen again.
    fileInput.value = "";
    if (chosen !== undefined) holding.load(chosen);
  });

  radiusInput.addEventListener("input", () => resize(AXES.map(() => Number(radiusInput.value))));
  radiusNumber.addEventListener("change", () => resize(AXES.map(() => Number(radiusNumber.value))));
  axesInput.addEventListener("change", () => resize(brush.radii_ZYX));
  for (const input of axisInputs)
    input.addEventListener("change", () => resize(axisInputs.map((each) => Number(each.value))));
  for (const input of [eraseInput, fillInput]) input.addEventListener("change", () => describe());

  listClasses();
  const plain = { shift: false, command: false, alt: false };
  const set = { ...plain, tool };
  const stepping = { ...set, repeats: true };
  const toggle = (input) => () => {
    input.checked = !input.checked;
    describe();
  };
  return {
    panel,
    place: { title: "Paint", icon: ICON },
    keys: [
      { description: "Paint: take the brush or put it away", code: "Digit7", ...plain, run: () => toolButton.click() },
      { description: "Paint: erase or paint", code: "KeyE", ...set, run: toggle(eraseInput) },
      { description: "Paint: fill with a click or paint", code: "KeyB", ...set, run: toggle(fillInput) },
      { description: "Paint: next class", code: "KeyC", ...set, run: () => choose((current % classes) + 1) },
      { description: "Paint: smaller brush", code: "BracketLeft", ...stepping, run: () => scale(1 / RESIZE) },
      { description: "Paint: larger brush", code: "BracketRight", ...stepping, run: () => scale(RESIZE) },
      {
        description: "Paint: take the last edit back",
        code: "KeyZ",
        ...stepping,
        command: true,
        run: () => turn(true),
      },
      {
        description: "Paint: do again the last edit taken back",
        code: "KeyZ",
        ...stepping,
        shift: true,
        command: true,
        run: () => turn(false),
      },
    ],
    volumeChanged: (opened) => holding.open(opened),
  };
}
