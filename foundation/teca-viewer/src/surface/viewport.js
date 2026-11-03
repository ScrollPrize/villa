import {
  SURFACE_BASE_VIEW_CELLS,
  SURFACE_NORMAL_OFFSET_LIMIT,
  SURFACE_WORK_LIMIT,
  SURFACE_ZOOM_LIMITS,
} from "../config.js";
import { surfaceCameraDemandBounds, surfaceCameraScreenToPoint } from "./camera.js";

export function surfaceVisibleSpan(grid_HW, scale_UV, zoom, aspect) {
  const base = Math.min(SURFACE_BASE_VIEW_CELLS, grid_HW[0], grid_HW[1]) / zoom;
  const gridAspect = (aspect * scale_UV[0]) / scale_UV[1];
  return [base * Math.max(1, gridAspect), base * Math.max(1, 1 / gridAspect)];
}

export function surfaceDemandWindow(source, camera, center_UV, zoom, aspect) {
  const span_UV = surfaceVisibleSpan(source.grid_HW, source.scale_UV, zoom, aspect);
  const demand = surfaceCameraDemandBounds(span_UV, camera.minimumDepth, camera.maximumDepth, camera);
  return { center_UV: center_UV.map((value, axis) => value + demand.centerOffset_UV[axis]), span_UV: demand.span_UV };
}

export function validateSurfaceWork(volumetric, panelSize) {
  const pixelCount = Math.max(1, Math.round(panelSize[0])) * Math.max(1, Math.round(panelSize[1]));
  const cameraWork = volumetric.tilt !== 0 || volumetric.perspective !== 0 ? 1 : 0;
  const lightingWork = volumetric.lighting > 0 || volumetric.feature !== "intensity" ? 1 : 0;
  const samplesPerPixel = volumetric.enabled
    ? (volumetric.layersFront + volumetric.layersBehind + 1) * (1 + cameraWork + lightingWork) +
      (volumetric.compareFlat ? 1 : 0)
    : 1;
  const work = pixelCount * samplesPerPixel;
  if (work > SURFACE_WORK_LIMIT) {
    throw new Error(
      `The Surface presentation needs ${work} work units per frame, exceeding the ${SURFACE_WORK_LIMIT}-unit interactive budget.`,
    );
  }
}

export function constrainSurfacePose(pose, grid_HW) {
  return {
    center_UV: [
      Math.min(grid_HW[1] - 1, Math.max(0, pose.center_UV[0])),
      Math.min(grid_HW[0] - 1, Math.max(0, pose.center_UV[1])),
    ],
    zoom: Math.min(SURFACE_ZOOM_LIMITS[1], Math.max(SURFACE_ZOOM_LIMITS[0], pose.zoom)),
    normalOffset: Math.min(SURFACE_NORMAL_OFFSET_LIMIT, Math.max(-SURFACE_NORMAL_OFFSET_LIMIT, pose.normalOffset)),
    normalSign: pose.normalSign,
  };
}

export function surfaceOffsetAtScreen(fraction_XY, visibleSpan_UV, camera) {
  return surfaceCameraScreenToPoint(
    [fraction_XY[0] * visibleSpan_UV[0], -fraction_XY[1] * visibleSpan_UV[1]],
    0,
    visibleSpan_UV,
    camera,
  );
}
