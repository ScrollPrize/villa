import { ORTHOGONAL_PLANES } from "../config.js";
import { isIdentityRotation, rotationAngleDegrees } from "../math.js";

export function rotationReadout({ plane, rotationMode, rotation, paneRotations }) {
  if (!ORTHOGONAL_PLANES.includes(plane)) throw new Error(`Unknown plane: ${plane}`);
  if (rotationMode === "frame") {
    const orientation = isIdentityRotation(rotation)
      ? "axis-aligned"
      : `tilted ${rotationAngleDegrees(rotation).toFixed(1)}°`;
    return `All panes ${orientation}`;
  }
  if (rotationMode !== "pane") throw new Error(`Unknown rotation mode: ${rotationMode}.`);
  const activeRotation = paneRotations[plane];
  const orientation = isIdentityRotation(activeRotation)
    ? "axis-aligned"
    : `tilted ${rotationAngleDegrees(activeRotation).toFixed(1)}°`;
  const residual = ORTHOGONAL_PLANES.filter((name) => name !== plane && !isIdentityRotation(paneRotations[name]))
    .map((name) => `${name.toUpperCase()} tilted ${rotationAngleDegrees(paneRotations[name]).toFixed(1)}°`)
    .join(", ");
  return `Active ${plane.toUpperCase()} ${orientation}${residual ? `, ${residual}` : ""}`;
}
