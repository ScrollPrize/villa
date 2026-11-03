import { SCALE_PICK_MAX_PIXELS, SCALE_PICK_MIN_PIXELS } from "../config.js";
import { clamp } from "../math.js";

function scaleBound(value, automaticValue, scaleCount) {
  return value === null || value === undefined || Number.isNaN(value)
    ? automaticValue
    : clamp(value, 0, scaleCount - 1);
}

export function scalePickPixels(viewWidthVoxels, panePixels) {
  return clamp(Math.floor(viewWidthVoxels), SCALE_PICK_MIN_PIXELS, Math.min(panePixels, SCALE_PICK_MAX_PIXELS));
}

export function resolveScaleIndex(autoScaleIndex, manualScaleIndex, manualCoarsestScaleIndex, scaleCount) {
  const finestScaleIndex = scaleBound(manualScaleIndex, 0, scaleCount);
  const coarsestScaleIndex = scaleBound(manualCoarsestScaleIndex, scaleCount - 1, scaleCount);
  if (finestScaleIndex > coarsestScaleIndex) {
    throw new Error("Finest scale index must not exceed coarsest scale index.");
  }
  return clamp(autoScaleIndex, finestScaleIndex, coarsestScaleIndex);
}

export function demandScale(scale) {
  return {
    path: scale.path,
    shape_ZYX: scale.shape_ZYX,
    chunks_ZYX: scale.chunks_ZYX,
    downsample_ZYX: scale.downsample_ZYX,
    offset_ZYX: scale.offset_ZYX,
  };
}
