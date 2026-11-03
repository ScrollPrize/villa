import {
  defaultSurfaceReliefMask,
  removeNearestSurfaceReliefControl,
  setSurfaceReliefControl,
  updateSurfaceReliefControl,
} from "../surface/relief-mask.js";
import { SurfaceReliefMaskOverlay } from "../views/relief-mask-overlay.js";
import { bindRangeAndNumber } from "./dom.js";

function normalizedSurfaceCenter(center_UV, grid_HW) {
  return [
    Math.min(1, Math.max(0, center_UV[0] / Math.max(1, grid_HW[1] - 1))),
    Math.min(1, Math.max(0, center_UV[1] / Math.max(1, grid_HW[0] - 1))),
  ];
}

export function createReliefEditor({ state, elements, pane, surface, setStatus }) {
  let boundary = "top";
  let depth = 0;
  let overlayVisible = true;

  function sync() {
    const mask = state.surfaceView.volumetric.mask;
    elements.surfaceReliefMaskEnabled.checked = mask.enabled;
    elements.surfaceReliefMaskOverlayVisible.checked = overlayVisible;
    elements.surfaceReliefMaskControls.hidden = !mask.enabled;
    elements.surfaceReliefMaskBoundary.value = boundary;
    elements.surfaceReliefMaskDepth.value = depth;
    elements.surfaceReliefMaskDepthNumber.value = depth;
    elements.surfaceReliefMaskStatus.textContent = `Top ${mask.top.length} ridges, bottom ${mask.bottom.length} ridges.`;
  }

  function syncOverlay() {
    const volumetric = state.surfaceView.volumetric;
    const shown = overlayVisible && volumetric.enabled && volumetric.mask.enabled ? surface.overlayView() : null;
    if (shown === null || shown.visibleSpan_UV === null) {
      overlay.clear();
      return;
    }
    overlay.update({
      visible: true,
      mask: volumetric.mask,
      grid_HW: shown.grid_HW,
      center_UV: state.surfaceView.center_UV,
      visibleSpan_UV: shown.visibleSpan_UV,
      camera: shown.camera,
      volumetric,
    });
  }

  function setControl() {
    const shown = surface.overlayView();
    if (shown === null) return;
    const [u, v] = normalizedSurfaceCenter(state.surfaceView.center_UV, shown.grid_HW);
    try {
      const mask = setSurfaceReliefControl(state.surfaceView.volumetric.mask, boundary, u, v, depth);
      if (surface.setReliefMask(mask)) {
        elements.surfaceReliefMaskStatus.textContent = `${boundary} control set at U ${u.toFixed(3)}, V ${v.toFixed(3)}.`;
      }
    } catch (error) {
      setStatus(error.message, true);
    }
  }

  function removeControl() {
    const shown = surface.overlayView();
    if (shown === null) return;
    const [u, v] = normalizedSurfaceCenter(state.surfaceView.center_UV, shown.grid_HW);
    try {
      const mask = removeNearestSurfaceReliefControl(state.surfaceView.volumetric.mask, boundary, u, v);
      if (surface.setReliefMask(mask)) {
        elements.surfaceReliefMaskStatus.textContent = `Nearest ${boundary} control removed.`;
      }
    } catch (error) {
      setStatus(error.message, true);
    }
  }

  const overlay = new SurfaceReliefMaskOverlay({
    container: pane.element,
    canvas: pane.canvas,
    onSelect: (selected) => {
      boundary = selected.boundary;
      depth = selected.depth;
      sync();
    },
    onEdit: (edit) => {
      try {
        const mask = updateSurfaceReliefControl(
          state.surfaceView.volumetric.mask,
          edit.boundary,
          edit.ridgeIndex,
          edit.pointIndex,
          edit.patch,
        );
        boundary = edit.boundary;
        depth = mask[edit.boundary][edit.ridgeIndex].points[edit.pointIndex].depth;
        surface.setReliefMask(mask, { syncControls: false, announce: false });
      } catch (error) {
        syncOverlay();
        setStatus(error.message, true);
      }
    },
  });

  elements.surfaceReliefMaskEnabled.addEventListener("change", () =>
    surface.setReliefMask({ ...state.surfaceView.volumetric.mask, enabled: elements.surfaceReliefMaskEnabled.checked }),
  );
  elements.surfaceReliefMaskOverlayVisible.addEventListener("change", () => {
    overlayVisible = elements.surfaceReliefMaskOverlayVisible.checked;
    syncOverlay();
  });
  elements.surfaceReliefMaskBoundary.addEventListener("change", () => {
    boundary = elements.surfaceReliefMaskBoundary.value;
    depth = boundary === "top" ? 0 : 1;
    sync();
  });
  bindRangeAndNumber(elements.surfaceReliefMaskDepth, elements.surfaceReliefMaskDepthNumber, (value) => {
    depth = Math.min(1, Math.max(0, value));
    sync();
  });
  elements.surfaceReliefMaskSet.addEventListener("click", setControl);
  elements.surfaceReliefMaskRemove.addEventListener("click", removeControl);
  elements.surfaceReliefMaskReset.addEventListener("click", () => surface.setReliefMask(defaultSurfaceReliefMask()));

  return { sync, syncOverlay };
}
