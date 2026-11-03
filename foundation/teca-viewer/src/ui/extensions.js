import { ORTHOGONAL_PLANES } from "../config.js";
import { labelColor } from "../gpu/label-colors.js";
import { presentationScale } from "../gpu/presentation-bricks.js";
import { MadeVolume } from "../volume/made-volume.js";
import { MemoryVolume } from "../volume/memory-volume.js";

const MARK_OPTIONS = { keys: ["diameter", "width", "voxels"], methods: { set: ["neighbours_N2"], show: [] } };
const SEAMS = {
  editSurface: { keys: [], methods: { changed: [] } },
  addLayer: { keys: [], methods: { setStyle: [], changed: [] } },
  addSurfaceImage: { keys: [], methods: { setStyle: [], setDetail: [] } },
  addMarks: MARK_OPTIONS,
  addSurfaceMarks: MARK_OPTIONS,
  add3d: { keys: [], methods: { changed: [] } },
  focus: { keys: ["surface"], methods: {} },
};
const STYLE = ["color_RGB", "opacity", "range"];
const MARKS = ["points_zyx_N3", "colors_N3", "edges_E2", "selected", "depth"];
const SURFACE_MARKS = ["surface", "cells_N2", "colors_N3", "edges_E2", "selected"];

export function taken(seam, given, needs, method = null) {
  const { keys, methods } = SEAMS[seam];
  const allowed = [...needs, ...(method === null ? keys : methods[method])];
  const other = Object.keys(given).find((key) => !allowed.includes(key));
  if (other !== undefined)
    throw new Error(`${method === null ? seam : `${seam}'s ${method}()`} takes ${allowed.join(", ")}, not ${other}.`);
  return given;
}

export function layerReaders(readers) {
  const { slices, surface } = taken("addLayer", readers, ["slices", "surface"]);
  if (surface instanceof MadeVolume)
    throw new Error(`Made volume ${surface.rootUrl} is drawn on the slices alone: hand addLayer surface: null.`);
  return { slices, surface };
}

export function drawnStyle(seam, given, method = null) {
  const { color_RGB, range } = taken(seam, given, STYLE, method);
  if (color_RGB === null && range !== undefined)
    throw new Error(`Labels take no range, being drawn whatever their value: not ${JSON.stringify(range)}.`);
  const [low, high] = Array.isArray(range) && range.length === 2 ? range : [];
  const bytes = Number.isInteger(low) && Number.isInteger(high) && 0 <= low && low <= high && high <= 255;
  if (color_RGB !== null && !bytes)
    throw new Error(`A range is [low, high] display bytes, 0 ≤ low ≤ high ≤ 255, not ${JSON.stringify(range)}.`);
  return given;
}

export function handleOf(seam, methods) {
  const names = ["remove", ...Object.keys(SEAMS[seam].methods)];
  const other = Object.keys(methods).find((name) => !names.includes(name));
  if (other !== undefined) throw new Error(`A handle of ${seam} has ${names.join("(), ")}(), not ${other}().`);
  return methods;
}

export function createExtensions({
  state,
  opened,
  view,
  marks,
  surfaceMarkers,
  dataPanels,
  viewPanels,
  rightTabs,
  infoTab,
  infoPanel,
  openVolume,
  showVolume,
  addSurface,
  readSurface,
  heldSurfaceChunk,
  setEditedSurface,
  surfaceChanged,
  setTool,
  setFocus,
  addKeys,
  markViewDirty,
  scheduleRender,
  queueOrthogonalRedraw,
  publishSession,
  setStatus,
}) {
  const attached = [];

  function addLayer(readers, style) {
    if (state.volume === null) throw new Error("Open a volume before adding a layer.");
    const layer = view.addLayer(layerReaders(readers), state.volume, drawnStyle("addLayer", style));
    markViewDirty();
    scheduleRender();
    return handleOf("addLayer", {
      setStyle: (next) => {
        view.setLayerStyle(layer, drawnStyle("addLayer", next, "setStyle"));
        queueOrthogonalRedraw();
      },
      changed: (changes) => view.refreshLayer(layer, changes),
      remove: () => {
        view.removeLayer(layer);
        queueOrthogonalRedraw();
      },
    });
  }

  function slicedLevels() {
    if (state.volume === null) throw new Error("Open a volume before making a volume on its levels.");
    return state.volume.scales.map((scale) => ({
      shape_ZYX: scale.shape_ZYX,
      chunks_ZYX: presentationScale(scale).chunks_ZYX,
      downsample_ZYX: scale.downsample_ZYX,
      offset_ZYX: scale.offset_ZYX,
    }));
  }

  function memoryVolume(name, zeroed) {
    return new MemoryVolume(name, slicedLevels(), zeroed);
  }

  function madeVolume(name, make) {
    return new MadeVolume(name, slicedLevels(), make);
  }

  function addSurfaceImage(url, image, style) {
    if (state.volume === null) throw new Error("Open a volume before adding a surface image.");
    if (style.color_RGB === null) throw new Error("A surface image is drawn in a colour, not as labels.");
    const drawn = view.addSurfaceImage(
      url,
      taken("addSurfaceImage", image, ["data_HW", "shape_HW"]),
      drawnStyle("addSurfaceImage", style),
    );
    queueOrthogonalRedraw(["surface"]);
    return handleOf("addSurfaceImage", {
      setStyle: (next) => {
        view.setSurfaceImageStyle(drawn, drawnStyle("addSurfaceImage", next, "setStyle"));
        queueOrthogonalRedraw(["surface"]);
      },
      setDetail: (detail) => {
        if (detail !== null)
          taken("addSurfaceImage", detail, ["data_HW", "shape_HW", "start_HW", "downsample"], "setDetail");
        const whole =
          detail === null ||
          (detail.start_HW.every(Number.isInteger) && Number.isInteger(detail.downsample) && detail.downsample >= 1);
        if (!whole)
          throw new Error(
            `A detail starts at a whole full pixel, a whole number of them a pixel: not [${detail.start_HW}] and ` +
              `${detail.downsample}.`,
          );
        view.setSurfaceImageDetail(drawn, detail);
        queueOrthogonalRedraw(["surface"]);
      },
      remove: () => {
        view.removeSurfaceImage(drawn);
        queueOrthogonalRedraw(["surface"]);
      },
    });
  }

  function markOptions(seam, options) {
    const { diameter, width, voxels } = taken(seam, options ?? {}, []);
    if (diameter !== undefined && !(diameter >= 0 && Number.isFinite(diameter)))
      throw new Error(`A mark's disc is 0 or more CSS pixels across, not ${diameter}.`);
    if (voxels === undefined && width !== undefined && !(width > 0 && Number.isFinite(width)))
      throw new Error(`A mark's edge is a positive number of CSS pixels across, not ${width}.`);
    if (voxels !== undefined && !(voxels > 0 && Number.isFinite(voxels)))
      throw new Error(`A mark's edge is a positive number of voxels across, not ${voxels}.`);
    if (voxels !== undefined && !(Array.isArray(width) && width.length === 2 && width[0] > 0 && width[0] <= width[1]))
      throw new Error(
        `A mark's edge of voxels is within [narrowest, widest] CSS pixels, not ${JSON.stringify(width)}.`,
      );
    return { diameter, width, voxels };
  }

  function addMarks(options) {
    const added = marks.addMarks(markOptions("addMarks", options));
    return handleOf("addMarks", {
      set: (next) => {
        added.set(taken("addMarks", next, MARKS, "set"));
        queueOrthogonalRedraw(ORTHOGONAL_PLANES);
      },
      show: (shown_N) => {
        added.show(shown_N);
        queueOrthogonalRedraw(ORTHOGONAL_PLANES);
      },
      remove: () => {
        added.remove();
        queueOrthogonalRedraw(ORTHOGONAL_PLANES);
      },
    });
  }

  function addSurfaceMarks(options) {
    const added = surfaceMarkers.addMarks(markOptions("addSurfaceMarks", options));
    return handleOf("addSurfaceMarks", {
      set: (next) => {
        added.set(taken("addSurfaceMarks", next, SURFACE_MARKS, "set"));
        queueOrthogonalRedraw(["surface"]);
      },
      show: (shown_N) => {
        added.show(shown_N);
        queueOrthogonalRedraw(["surface"]);
      },
      remove: () => {
        added.remove();
        queueOrthogonalRedraw(["surface"]);
      },
    });
  }

  function add3d(object) {
    view.add3d(object);
    queueOrthogonalRedraw(["3d"]);
    return handleOf("add3d", {
      changed: () => queueOrthogonalRedraw(["3d"]),
      remove: () => {
        view.remove3d(object);
        queueOrthogonalRedraw(["3d"]);
      },
    });
  }

  function filterCT(passes, reach) {
    if (!Number.isInteger(reach) || reach < 0)
      throw new Error(`A filter reach is a whole number of taps, not ${reach}.`);
    view.filterCT(passes, reach);
    markViewDirty();
    scheduleRender();
  }

  async function editSurface(url, surface) {
    setEditedSurface(url, surface);
    if (!(await addSurface(url))) return null;
    return handleOf("editSurface", {
      changed: async (start_HW, stop_HW) => {
        await surfaceChanged(url, surface, start_HW, stop_HW);
        queueOrthogonalRedraw();
      },
      remove: () => {
        setEditedSurface(url, null);
        return addSurface(url);
      },
    });
  }

  function heldCT(level, chunk_ZYX) {
    const volume = state.volume;
    const channel = state.selectedChannel;
    return (
      volume.heldChunk(...chunk_ZYX, channel, volume.scales[level]) ??
      heldSurfaceChunk(volume.rootUrl, level, chunk_ZYX, channel)
    );
  }

  function focus(point_ZYX, options) {
    setFocus(point_ZYX, taken("focus", options ?? {}, []));
  }

  function focusPoint() {
    return state.center_ZYX.slice();
  }

  function surfaces() {
    return state.overlays
      .filter((descriptor) => descriptor.primarySource === state.activeSource && descriptor.visible)
      .map((descriptor) => descriptor.path);
  }

  function surfaceView() {
    return surfaceMarkers.view();
  }

  function surfacePoints() {
    return surfaceMarkers.points();
  }

  function listedSurfaces() {
    return state.overlays
      .filter((descriptor) => descriptor.primarySource === state.activeSource)
      .map((descriptor) => descriptor.path);
  }

  const viewer = {
    openVolume,
    showVolume,
    addSurface,
    surfaces,
    listedSurfaces,
    surfaceView,
    surfacePoints,
    readSurface,
    editSurface,
    addLayer,
    addSurfaceImage,
    memoryVolume,
    madeVolume,
    heldCT,
    addMarks,
    addSurfaceMarks,
    add3d,
    filterCT,
    setTool,
    focus,
    focusPoint,
    labelColor,
    setStatus,
  };

  function placePanel(name, panel, place) {
    if (place === "data" || place === "view") {
      (place === "data" ? dataPanels : viewPanels).append(panel);
      return null;
    }
    if (typeof place?.title !== "string" || !/^<svg[\s>]/.test(place.icon))
      throw new Error(`Extension ${name} places its panel in ${JSON.stringify(place)}, not "data", "view" or a tab.`);
    const tab = Object.assign(document.createElement("button"), {
      type: "button",
      className: "rtab",
      innerHTML: place.icon,
    });
    tab.firstElementChild.classList.add("icon");
    tab.append(Object.assign(document.createElement("span"), { textContent: place.title }));
    Object.assign(tab.dataset, { rtab: name });
    tab.title = place.title;
    tab.setAttribute("role", "tab");
    const tabPanel = Object.assign(document.createElement("div"), { className: "rpanel", hidden: true });
    tabPanel.dataset.rpanel = name;
    tabPanel.append(panel);
    rightTabs.insertBefore(tab, infoTab);
    infoPanel.before(tabPanel);
    return tab;
  }

  function markToolTab() {
    for (const extension of attached)
      extension.tab?.classList.toggle("tool-set", extension.tool !== null && extension.tool === state.tool);
  }

  function extend(name, create) {
    if (!/^[a-z]+(?:-[a-z]+)*$/.test(name)) throw new Error(`Extension name ${name} is not kebab-case.`);
    if (attached.some((extension) => extension.name === name)) throw new Error(`Extension ${name} is attached twice.`);
    const extension = { name, tool: null };
    const own = {
      ...viewer,
      setTool: (tool) => {
        if (tool !== null) extension.tool = tool;
        setTool(tool);
        markToolTab();
      },
    };
    const { panel, place, keys, volumeChanged, surfacesChanged, focusChanged, surfaceViewChanged, kept } = create(own, {
      saved: state.extensions[name] ?? null,
      save: (value) => {
        state.extensions = { ...state.extensions, [name]: value };
        return publishSession() === null;
      },
      opened: opened[name] ?? null,
    });
    addKeys(keys ?? []);
    const tab = placePanel(name, panel, place);
    attached.push(
      Object.assign(extension, { tab, volumeChanged, surfacesChanged, focusChanged, surfaceViewChanged, kept }),
    );
  }

  async function kept() {
    const parts = await Promise.all(
      attached.map(async (extension) => [extension.name, (await extension.kept?.()) ?? null]),
    );
    return Object.fromEntries(parts.filter(([, part]) => part !== null));
  }

  function volumeChanged(volume) {
    for (const extension of attached) extension.volumeChanged?.(volume);
  }

  function surfacesChanged() {
    for (const extension of attached) extension.surfacesChanged?.();
  }

  let told_ZYX = null;

  function focusSettled() {
    if (told_ZYX !== null && state.center_ZYX.every((value, axis) => value === told_ZYX[axis])) return;
    told_ZYX = state.center_ZYX.slice();
    for (const extension of attached) extension.focusChanged?.();
  }

  let toldView = "null";

  function surfaceViewDrawn() {
    const shown = JSON.stringify(surfaceMarkers.view());
    if (shown === toldView) return;
    toldView = shown;
    for (const extension of attached) extension.surfaceViewChanged?.();
  }

  return { extend, volumeChanged, surfacesChanged, focusSettled, surfaceViewDrawn, kept };
}
