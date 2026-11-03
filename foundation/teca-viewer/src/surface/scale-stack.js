import { SCALE_PICK_TOLERANCE } from "../config.js";
import { presentationScale } from "../gpu/presentation-bricks.js";

const brickScales = new WeakMap();

export function surfacePresentationScale(scale) {
  if (!brickScales.has(scale)) brickScales.set(scale, presentationScale(scale, scale.downsample_ZYX));
  return brickScales.get(scale);
}

function validateScales(scales) {
  if (!Array.isArray(scales) || scales.length === 0) {
    throw new Error("Surface volume needs at least one pyramid level.");
  }
}

function scaleBound(value, automaticValue, scaleCount, label) {
  if (value === null) return automaticValue;
  if (!Number.isSafeInteger(value) || value < 0 || value >= scaleCount) {
    throw new Error(`Surface ${label} level must be inside 0..${scaleCount - 1}.`);
  }
  return value;
}

export function surfaceDisplayScaleIndex(
  scales,
  visibleSpan_UV,
  scale_UV,
  panelWidth,
  finestScaleIndex,
  coarsestScaleIndex,
) {
  validateScales(scales);
  if (
    !Array.isArray(visibleSpan_UV) ||
    visibleSpan_UV.length !== 2 ||
    visibleSpan_UV.some((value) => !Number.isFinite(value) || value <= 0) ||
    !Array.isArray(scale_UV) ||
    scale_UV.length !== 2 ||
    scale_UV.some((value) => !Number.isFinite(value) || value <= 0) ||
    !Number.isFinite(panelWidth) ||
    panelWidth <= 0
  ) {
    throw new Error("Surface display-scale inputs are malformed.");
  }
  const finest = scaleBound(finestScaleIndex, 0, scales.length, "finest");
  const coarsest = scaleBound(coarsestScaleIndex, scales.length - 1, scales.length, "coarsest");
  if (finest > coarsest) throw new Error("Surface finest level must not be coarser than its coarsest level.");
  const voxelsPerPixel = (visibleSpan_UV[0] / scale_UV[0] / panelWidth) * SCALE_PICK_TOLERANCE;
  let selected = finest;
  for (let index = finest; index <= coarsest; index++) {
    if (Math.max(...scales[index].downsample_ZYX) <= voxelsPerPixel) selected = index;
  }
  return selected;
}

export function surfaceVolumeScaleStack(scales, finestScaleIndex, coarsestScaleIndex, displayScaleIndex = null) {
  validateScales(scales);
  const finest = scaleBound(finestScaleIndex, 0, scales.length, "finest");
  const coarsest = scaleBound(coarsestScaleIndex, scales.length - 1, scales.length, "coarsest");
  if (finest > coarsest) throw new Error("Surface finest level must not be coarser than its coarsest level.");
  const display = scaleBound(displayScaleIndex, finest, scales.length, "display");
  if (display < finest || display > coarsest) {
    throw new Error("Surface display level must be inside its finest-to-coarsest bounds.");
  }
  return (display === coarsest ? [scales[display]] : [scales[display], scales[coarsest]]).map(surfacePresentationScale);
}
