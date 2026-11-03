import { SURFACE_DIVIDER_KEY_STEP } from "../config.js";
import { SURFACE_DOCK_PANES } from "../surface/layout.js";
import { viewerDock } from "../views/pose.js";

export function createSurfaceLayout({
  state,
  view,
  elements,
  resetPaneOrientation,
  setActivePlane,
  surfaceController,
  surfaceControls,
  publishSession,
}) {
  function initSurfaceLayout({ handleViewerLayoutChange, cancelPointerInteraction }) {
    const mode = elements.surfaceLayoutMode;
    const change = async () => {
      if (mode.disabled) return;
      cancelPointerInteraction();
      const layout = structuredClone(state.surfaceView.layout);
      layout.mode = mode.value;
      layout.planes = [...elements.surfacePlaneInputs]
        .filter((input) => input.checked)
        .map((input) => input.dataset.surfacePlane);
      if (layout.planes.length === 0) {
        surfaceControls.syncControls();
        return;
      }
      layout.maximized = null;
      mode.disabled = true;
      for (const input of elements.surfacePlaneInputs) input.disabled = true;
      try {
        await surfaceController.setLayout(layout);
        handleViewerLayoutChange();
      } catch (error) {
        if (error.name !== "AbortError") surfaceControls.setStatus(error.message, true);
      } finally {
        mode.disabled = false;
        for (const input of elements.surfacePlaneInputs) input.disabled = false;
      }
    };
    mode.addEventListener("change", change);
    for (const input of elements.surfacePlaneInputs) input.addEventListener("change", change);
    for (const select of document.querySelectorAll(".pane-orientation")) {
      select.addEventListener("change", () => {
        if (select.value === "") return;
        cancelPointerInteraction();
        setActivePlane(select.dataset.plane, { redrawOnly: true });
        resetPaneOrientation(select.dataset.plane, select.value);
        select.value = "";
      });
    }
    let drag = null;
    const apply = (divider, delta, initial) => {
      const { primary, layout } = viewerDock(state);
      let split = layout.split;
      const weights = [...layout.weights];
      if (divider.dataset.split === "main") {
        split = Math.max(0.2, Math.min(0.85, initial.split + delta));
      } else {
        const index = Number(divider.dataset.split);
        const first = SURFACE_DOCK_PANES.indexOf(layout.planes[index]);
        const second = SURFACE_DOCK_PANES.indexOf(layout.planes[index + 1]);
        const sum = initial.weights[first] + initial.weights[second];
        const value = Math.max(0.1, Math.min(sum - 0.1, initial.weights[first] + delta * sum));
        weights[first] = value;
        weights[second] = sum - value;
      }
      surfaceController.setDockSizes(split, weights);
      view.surfaceLayout.apply(layout, true, primary);
      handleViewerLayoutChange();
    };
    for (const divider of [view.surfaceLayout.divider, ...view.surfaceLayout.dividers]) {
      divider.addEventListener("pointerdown", (event) => {
        if (event.button !== 0 || !event.isPrimary) return;
        event.preventDefault();
        cancelPointerInteraction();
        const vertical = divider.getAttribute("aria-orientation") === "vertical";
        let size;
        if (divider.dataset.split === "main") {
          const rect = view.surfaceLayout.grid.getBoundingClientRect();
          size = vertical ? rect.width : rect.height;
        } else {
          const a = divider.previousElementSibling.getBoundingClientRect();
          const b = divider.nextElementSibling.getBoundingClientRect();
          size = vertical ? a.width + b.width : a.height + b.height;
        }
        drag = {
          pointerId: event.pointerId,
          vertical,
          size,
          start: vertical ? event.clientX : event.clientY,
          initial: structuredClone(viewerDock(state).layout),
        };
        divider.setPointerCapture(event.pointerId);
      });
      divider.addEventListener("pointermove", (event) => {
        if (drag === null || event.pointerId !== drag.pointerId) return;
        apply(divider, ((drag.vertical ? event.clientX : event.clientY) - drag.start) / drag.size, drag.initial);
      });
      const end = (event) => {
        if (drag === null || event.pointerId !== drag.pointerId) return;
        drag = null;
        if (divider.hasPointerCapture(event.pointerId)) divider.releasePointerCapture(event.pointerId);
        publishSession();
      };
      divider.addEventListener("pointerup", end);
      divider.addEventListener("pointercancel", end);
      divider.addEventListener("lostpointercapture", end);
      divider.addEventListener("keydown", (event) => {
        const vertical = divider.getAttribute("aria-orientation") === "vertical";
        const keys = vertical ? ["ArrowLeft", "ArrowRight"] : ["ArrowUp", "ArrowDown"];
        if (!keys.includes(event.key)) return;
        event.preventDefault();
        event.stopPropagation();
        apply(
          divider,
          event.key === keys[0] ? -SURFACE_DIVIDER_KEY_STEP : SURFACE_DIVIDER_KEY_STEP,
          structuredClone(viewerDock(state).layout),
        );
        publishSession();
      });
    }
  }

  return { initSurfaceLayout };
}
