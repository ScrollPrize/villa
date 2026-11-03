import { sessionSnapshot } from "../session.js";

export function installProbe({ state, view, surfaceController, getOverlay2dStats }) {
  window.__tecaDebug = {
    session: () => sessionSnapshot(state),
    renderStats: () => ({ ...view.renderStats }),
    slicedPlaneStats: () => view.slicedPlane.stats(),
    layerStats: () => view.layerStats(),
    overlay2dStats: () => getOverlay2dStats(),
    surfaceStats: () => surfaceController.stats(),
    rendererStats: () => view.getRendererStats(),
  };
}
