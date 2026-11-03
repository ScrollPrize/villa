import { ORTHOGONAL_PLANES } from "../config.js";

export const SURFACE_DOCK_PANES = [...ORTHOGONAL_PLANES, "3d"];

export function dockedViewport(layout, viewport) {
  return layout.mode !== "solo" && layout.planes.includes(viewport) ? viewport : "surface";
}

export function defaultSurfaceLayout() {
  return { mode: "solo", planes: ["xy"], split: 0.7, weights: [1, 1, 1, 1], maximized: null };
}

export function validateSurfaceLayout(layout) {
  if (
    layout === null ||
    typeof layout !== "object" ||
    Array.isArray(layout) ||
    Object.keys(layout).sort().join(",") !== "maximized,mode,planes,split,weights" ||
    !["solo", "below", "right"].includes(layout.mode) ||
    !Array.isArray(layout.planes) ||
    layout.planes.length < 1 ||
    layout.planes.length > 4 ||
    new Set(layout.planes).size !== layout.planes.length ||
    layout.planes.some((plane) => !SURFACE_DOCK_PANES.includes(plane)) ||
    !Number.isFinite(layout.split) ||
    layout.split < 0.2 ||
    layout.split > 0.85 ||
    !Array.isArray(layout.weights) ||
    layout.weights.length !== 4 ||
    layout.weights.some((weight) => !Number.isFinite(weight) || weight < 0.1 || weight > 10) ||
    (layout.maximized !== null && layout.maximized !== "surface" && !layout.planes.includes(layout.maximized))
  ) {
    throw new Error("Invalid Surface pane layout.");
  }
}
