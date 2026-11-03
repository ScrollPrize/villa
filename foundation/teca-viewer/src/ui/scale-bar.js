import { rotatedPlaneBasis } from "../planes.js";

const WIDTH_SHARE = 0.22;

function niceScaleLengthUm(targetUm) {
  if (!Number.isFinite(targetUm) || targetUm <= 0) {
    throw new Error("Scale-bar target must be finite and positive.");
  }
  const exponent = Math.floor(Math.log10(targetUm));
  const power = 10 ** exponent;
  const fraction = targetUm / power;
  const multiplier = fraction >= 5 ? 5 : fraction >= 2 ? 2 : 1;
  return multiplier * power;
}

export function formatScaleLengthUm(lengthUm) {
  return `${Number(lengthUm.toPrecision(6))} um`;
}

export function scaleBarMeasurement(
  plane,
  rotation,
  physicalScale_ZYX,
  physicalUnits_ZYX,
  viewWidthVoxels,
  panelCssWidth,
  aspect,
) {
  if (plane === "3d") return null;
  const { right_XYZ } = rotatedPlaneBasis(plane, rotation);
  const right_ZYX = [right_XYZ[2], right_XYZ[1], right_XYZ[0]];
  const contributingAxes = right_ZYX
    .map((component, axis) => [component, axis])
    .filter(([component]) => Math.abs(component) > 1e-12);
  const units = new Set(contributingAxes.map(([, axis]) => physicalUnits_ZYX[axis]));
  const hasPhysicalScale =
    contributingAxes.every(([, axis]) => physicalScale_ZYX[axis] !== null && physicalUnits_ZYX[axis] !== null) &&
    units.size === 1;
  const unitsPerDisplayedVoxel = hasPhysicalScale
    ? Math.hypot(...right_ZYX.map((component, axis) => component * physicalScale_ZYX[axis]))
    : 1;
  const visibleVoxels = viewWidthVoxels * (aspect >= 1 ? aspect : 1);
  const pixelsPerDisplayedVoxel = panelCssWidth / visibleVoxels;
  const targetUnits = visibleVoxels * unitsPerDisplayedVoxel * WIDTH_SHARE;
  const lengthUnits = niceScaleLengthUm(targetUnits);
  return {
    lengthUnits,
    unit: hasPhysicalScale ? contributingAxes.map(([, axis]) => physicalUnits_ZYX[axis])[0] : null,
    unitsPerDisplayedVoxel,
    widthPixels: (lengthUnits / unitsPerDisplayedVoxel) * pixelsPerDisplayedVoxel,
  };
}
