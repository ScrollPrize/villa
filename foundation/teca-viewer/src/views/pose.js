import { BASE_VIEW_WIDTH_VOXELS, ORTHOGONAL_PLANES } from "../config.js";
import { isIdentityRotation } from "../math.js";
import { rotatedPlaneBasis } from "../planes.js";

export function planeRotation(state, plane) {
  if (!ORTHOGONAL_PLANES.includes(plane)) throw new Error(`Unknown plane: ${plane}`);
  return state.rotationMode === "frame" ? state.rotation : state.paneRotations[plane];
}

export function planeDisplayBasis(state, plane) {
  return rotatedPlaneBasis(plane, planeRotation(state, plane));
}

export function viewIsRotated(state, plane = null) {
  if (plane !== null) return !isIdentityRotation(planeRotation(state, plane));
  return ORTHOGONAL_PLANES.some((name) => !isIdentityRotation(planeRotation(state, name)));
}

export function baseViewWidth(state) {
  const shape_ZYX = state.volume.fullShape_ZYX;
  return Math.min(shape_ZYX[1], shape_ZYX[2], BASE_VIEW_WIDTH_VOXELS);
}

export function viewWidth(state) {
  return baseViewWidth(state) / state.zoom;
}

export function orthogonalStepPlane(state) {
  return state.activeViewport === "xy" || state.activeViewport === "zy" || state.activeViewport === "xz"
    ? state.activeViewport
    : state.plane;
}

export function drawnPlanes(state, planes = ORTHOGONAL_PLANES) {
  const requested = [...new Set(planes)];
  for (const plane of requested) {
    if (!ORTHOGONAL_PLANES.includes(plane)) throw new Error(`Unknown plane: ${plane}`);
  }
  return drawnPanes(state, requested);
}

export function shownPanes(state, panes) {
  const requested = [...new Set(panes)];
  const dock = viewerDock(state);
  if (dock !== null && dock.layout.mode !== "solo") {
    const { primary, layout } = dock;
    const visible = layout.maximized === null ? [primary, ...layout.planes] : [layout.maximized];
    return requested.filter((pane) => visible.includes(pane));
  }
  if (state.fullscreenViewport === null) return requested;
  return requested.filter((pane) => pane === state.fullscreenViewport);
}

export function drawnPanes(state, panes) {
  const shown = shownPanes(state, panes);
  if (shownPanes(state, ["3d"]).length === 0) return shown;
  return [...new Set(panes)].filter((pane) => shown.includes(pane) || ORTHOGONAL_PLANES.includes(pane));
}

export function viewerDock(state) {
  if (state.fullscreenViewport === "surface") return { primary: "surface", layout: state.surfaceView.layout };
  return null;
}

export function shownView(state) {
  const dock = viewerDock(state);
  if (dock === null) return state.fullscreenViewport ?? "grid";
  return dock.layout.mode === "solo" ? "surface" : (dock.layout.maximized ?? "surface");
}

export function panelVoxelOffset(canvasX, canvasY, panelX, panelY, panelWidth, panelHeight, zoom, baseViewWidth) {
  const viewWidthVoxels = baseViewWidth / zoom;
  const aspect = Math.max(0.001, panelWidth / panelHeight);
  const visibleU = viewWidthVoxels * (aspect >= 1 ? aspect : 1);
  const visibleV = viewWidthVoxels * (aspect >= 1 ? 1 : 1 / aspect);
  return {
    u: ((canvasX - panelX) / panelWidth - 0.5) * visibleU,
    v: ((canvasY - panelY) / panelHeight - 0.5) * visibleV,
  };
}
