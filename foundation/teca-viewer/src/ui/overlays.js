import { OVERLAY_LINE_WIDTH_RANGE } from "../config.js";
import { overlayColor } from "../surface/overlay-store.js";
import { bindRangeAndNumber } from "./dom.js";
import { layerRowMenu, wireLayerRowMenu } from "./layer-row.js";

export function createOverlayControls({
  state,
  view,
  elements,
  wirePopover,
  loadOverlay,
  syncOverlayLoads,
  removeOverlay,
  setOverlayVisible,
  setOverlayLineWidth,
  surfaceController,
  surfaceControls,
  setActiveViewport,
  onLayoutChange,
  queueOrthogonalRedraw,
  publishSession,
  surfacesChanged,
  setStatus,
}) {
  function redraw() {
    publishSession();
    queueOrthogonalRedraw();
  }

  function reportStatus(message, error = false) {
    elements.overlayStatus.textContent = message;
    elements.overlayStatus.hidden = message === "" || (!error && state.overlays.length > 0);
    elements.overlayStatus.classList.toggle("error", error);
    setStatus(message);
  }

  function setFormLoading(busy) {
    elements.overlayForm.setAttribute("aria-busy", String(busy));
    elements.overlayPath.readOnly = busy;
    elements.overlayAdd.disabled = busy;
    elements.overlayAdd.classList.toggle("loading", busy);
    elements.overlayAdd.querySelector(".button-label").textContent = busy ? "Loading…" : "Add";
  }

  function awaited(path) {
    return state.fullscreenViewport === "surface" && state.surfaceView.path === path;
  }

  function reportLoading(path, message) {
    reportStatus(message);
    if (awaited(path)) surfaceControls.setRestoreStatus(message);
  }

  const report = {
    loading: (path) =>
      reportLoading(
        path,
        state.surfaceView.path === path ? "Opening Surface: loading geometry…" : `Loading surface ${path}…`,
      ),
    progress: (path, done, total) =>
      reportLoading(path, `Opening Surface: reading geometry… ${Math.round((done / total) * 100)}%`),
    loaded: (overlay) => {
      renderRows();
      surfacesChanged();
      redraw();
      reportStatus(`Surface ${overlay.path} loaded.`);
      if (
        state.fullscreenViewport === "surface" &&
        state.surfaceView.path === overlay.path &&
        surfaceController.overlay !== overlay
      ) {
        void surfaceController.openPath(overlay.path).catch((error) => surfaceController.handleOpenError(error));
      }
    },
    failed: (path, error) => {
      reportStatus(error.message, true);
      if (awaited(path)) surfaceControls.setRestoreStatus(`Could not restore Surface. ${error.message}`, syncLoads);
    },
  };

  function syncLoads() {
    renderRows();
    syncOverlayLoads(report);
  }

  function renderRows() {
    const openPath =
      [...elements.overlayItems.querySelectorAll(".overlay-item")].find(
        (row) => !row.querySelector(".layer-row-popover").hidden,
      )?.dataset.path ?? null;
    elements.overlayItems.replaceChildren(
      ...state.overlays.map((descriptor) => {
        const path = descriptor.path;
        const row = document.createElement("div");
        row.className = "overlay-item layer-row";
        row.dataset.path = path;

        const name = document.createElement("span");
        name.className = "overlay-name layer-row-name";
        name.textContent = path.split("/").filter(Boolean).at(-1);
        name.title = `${path}\nvolume ${descriptor.primarySource}`;

        const toggle = document.createElement("input");
        toggle.type = "checkbox";
        toggle.checked = descriptor.visible;
        toggle.setAttribute("aria-label", `Show ${name.textContent}`);
        toggle.addEventListener("change", () => {
          setOverlayVisible(path, toggle.checked);
          surfacesChanged();
          if (!toggle.checked && state.surfaceView.path === path) surfaceController.close();
          redraw();
          queueMicrotask(syncLoads);
        });

        const swatch = document.createElement("span");
        swatch.className = "overlay-swatch";
        swatch.style.background = overlayColor(descriptor);

        const viewSurface = document.createElement("button");
        viewSurface.type = "button";
        viewSurface.className = "overlay-view";
        viewSurface.textContent = "View surface";
        viewSurface.addEventListener("click", async () => {
          try {
            const shown = () => surfaceController.active() && surfaceController.overlay?.path === path;
            if (!shown()) await surfaceController.openPath(path);
            if (!shown()) return;
            if (state.surfaceView.layout.maximized !== null && state.surfaceView.layout.maximized !== "surface") {
              await surfaceController.setLayout({ ...state.surfaceView.layout, maximized: null });
              onLayoutChange();
            }
            if (!shown()) return;
            setActiveViewport("surface", { redrawOnly: true });
            view.getSurfacePane().canvas.focus();
          } catch (error) {
            if (error.name !== "AbortError") reportStatus(error.message, true);
          }
        });

        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "overlay-remove layer-row-remove";
        remove.textContent = "Remove";
        remove.setAttribute("aria-label", `Remove ${name.textContent}`);
        remove.addEventListener("click", () => {
          if (state.surfaceView.path === path) surfaceController.close({ clearSelection: true });
          removeOverlay(path);
          renderRows();
          surfacesChanged();
          redraw();
        });

        row.append(toggle, swatch, name);
        const menu = layerRowMenu(wirePopover, row, name.textContent);
        menu.popover.append(viewSurface, remove);
        if (path === openPath) menu.open();
        return row;
      }),
    );
  }

  wireLayerRowMenu(wirePopover, elements.overlayDisplayMenu, elements.overlayDisplayPopover);
  setFormLoading(false);
  for (const input of [elements.overlayLineWidth, elements.overlayLineWidthNumber]) {
    Object.assign(input, { min: OVERLAY_LINE_WIDTH_RANGE[0], max: OVERLAY_LINE_WIDTH_RANGE[1] });
    input.value = state.overlayLineWidth;
  }
  async function addSurface(input) {
    return (await loadOverlay(input, report)) !== null;
  }

  elements.overlayForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const input = elements.overlayPath.value.trim();
    if (input === "" || elements.overlayAdd.disabled) return;
    setFormLoading(true);
    if (await addSurface(input)) elements.overlayPath.value = "";
    setFormLoading(false);
  });
  bindRangeAndNumber(elements.overlayLineWidth, elements.overlayLineWidthNumber, (value) => {
    setOverlayLineWidth(value);
    elements.overlayLineWidth.value = state.overlayLineWidth;
    elements.overlayLineWidthNumber.value = state.overlayLineWidth;
    redraw();
  });
  syncLoads();
  return { syncLoads, addSurface };
}
