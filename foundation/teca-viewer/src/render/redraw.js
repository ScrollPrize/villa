import { ORTHOGONAL_PLANES } from "../config.js";
import { drawnPanes } from "../views/pose.js";

function unionPlaneLists(current, next) {
  if (current === null || next === null) return null;
  return [...new Set([...current, ...next])];
}

export function createRedraw({ updateOverlays, updateMarks, surfaceMarkers, surfaceViewDrawn, state, view }) {
  let orthogonalRedrawPending = false;
  let orthogonalRedrawPlanes;
  function queueOrthogonalRedraw(planes = null) {
    if (planes === null) {
      orthogonalRedrawPlanes = null;
    } else if (orthogonalRedrawPlanes === undefined) {
      orthogonalRedrawPlanes = [...planes];
    } else if (orthogonalRedrawPlanes !== null) {
      orthogonalRedrawPlanes = unionPlaneLists(orthogonalRedrawPlanes, planes);
    }
    if (orthogonalRedrawPending) return;
    orthogonalRedrawPending = true;
    window.requestAnimationFrame(() => {
      orthogonalRedrawPending = false;
      const requestedPanes = orthogonalRedrawPlanes ?? view.paneNames();
      orthogonalRedrawPlanes = undefined;
      const panesToRender = drawnPanes(state, requestedPanes);
      const drawnPlanes = panesToRender.filter((pane) => ORTHOGONAL_PLANES.includes(pane));
      if (drawnPlanes.length > 0 && !panesToRender.includes("3d")) panesToRender.push(...drawnPanes(state, ["3d"]));
      if (panesToRender.length === 0) return;
      const overlayContext = updateOverlays(drawnPlanes, queueOrthogonalRedraw);
      updateMarks(drawnPlanes);
      const surfaceDrawn = panesToRender.includes("surface") && surfaceMarkers.shown();
      if (surfaceDrawn) surfaceMarkers.syncFocus();
      view.showOrthogonalGrid(panesToRender, state.cameraOrientation, overlayContext);
      if (surfaceDrawn) surfaceMarkers.syncPointer();
      surfaceViewDrawn();
    });
  }

  return { queueOrthogonalRedraw };
}
