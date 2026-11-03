import { isIdentityRotation, rotateVectorByQuat } from "./math.js";

const PLANE_BASES_XYZ = {
  xy: { right_XYZ: [1, 0, 0], up_XYZ: [0, 1, 0], normal_XYZ: [0, 0, 1] },
  zy: { right_XYZ: [0, 0, 1], up_XYZ: [0, 1, 0], normal_XYZ: [1, 0, 0] },
  xz: { right_XYZ: [1, 0, 0], up_XYZ: [0, 0, 1], normal_XYZ: [0, 1, 0] },
};

export function planeViewAxes(plane) {
  if (plane === "xy") return [2, 1];
  if (plane === "zy") return [0, 1];
  if (plane === "xz") return [2, 0];
  throw new Error(`Unknown plane: ${plane}`);
}

export function planeStepAxis(plane) {
  if (plane === "xy") return 0;
  if (plane === "zy") return 2;
  if (plane === "xz") return 1;
  throw new Error(`Unknown plane: ${plane}`);
}

export function rotatedPlaneBasis(plane, rotation) {
  const basis = PLANE_BASES_XYZ[plane];
  if (!basis) throw new Error(`Unknown plane: ${plane}`);
  if (isIdentityRotation(rotation)) return basis;
  return {
    right_XYZ: rotateVectorByQuat(basis.right_XYZ, rotation),
    up_XYZ: rotateVectorByQuat(basis.up_XYZ, rotation),
    normal_XYZ: rotateVectorByQuat(basis.normal_XYZ, rotation),
  };
}
