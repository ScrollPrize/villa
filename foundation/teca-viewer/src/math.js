// Rotation quaternions are [x, y, z, w] and act on XYZ vectors.

export const IDENTITY_ROTATION = [0, 0, 0, 1];

export function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export function dot_ZYX(a_ZYX, b_ZYX) {
  return a_ZYX[0] * b_ZYX[0] + a_ZYX[1] * b_ZYX[1] + a_ZYX[2] * b_ZYX[2];
}

function normalize_XYZ(vector_XYZ) {
  const length = Math.hypot(vector_XYZ[0], vector_XYZ[1], vector_XYZ[2]) || 1;
  return [vector_XYZ[0] / length, vector_XYZ[1] / length, vector_XYZ[2] / length];
}

function cross_XYZ(a_XYZ, b_XYZ) {
  return [
    a_XYZ[1] * b_XYZ[2] - a_XYZ[2] * b_XYZ[1],
    a_XYZ[2] * b_XYZ[0] - a_XYZ[0] * b_XYZ[2],
    a_XYZ[0] * b_XYZ[1] - a_XYZ[1] * b_XYZ[0],
  ];
}

export function isIdentityRotation(rotation) {
  return rotation[0] === 0 && rotation[1] === 0 && rotation[2] === 0;
}

export function quatFromAxisAngle(axis_XYZ, degrees) {
  const [x, y, z] = normalize_XYZ(axis_XYZ);
  const half = (degrees * Math.PI) / 360;
  const s = Math.sin(half);
  return [x * s, y * s, z * s, Math.cos(half)];
}

export function quatMultiply(a, b) {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  const q = [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
  const norm = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return q.map((value) => value / norm);
}

export function quatToEulerXYZDegrees(rotation) {
  if (!Array.isArray(rotation) || rotation.length !== 4 || rotation.some((value) => !Number.isFinite(value))) {
    throw new Error("Quaternion rotation must contain four finite components.");
  }
  const norm = Math.hypot(...rotation);
  if (norm < 1e-12) throw new Error("Quaternion rotation must be nonzero.");
  const [x, y, z, w] = rotation.map((value) => value / norm);
  const radians_XYZ = [
    Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
    Math.asin(clamp(2 * (w * y - z * x), -1, 1)),
    Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
  ];
  return radians_XYZ.map((value) => (value * 180) / Math.PI);
}

export function quatAlignVectors(from_XYZ, to_XYZ, fallbackAxis_XYZ = null) {
  for (const [name, vector_XYZ] of [
    ["from", from_XYZ],
    ["to", to_XYZ],
  ]) {
    if (
      !Array.isArray(vector_XYZ) ||
      vector_XYZ.length !== 3 ||
      vector_XYZ.some((value) => !Number.isFinite(value)) ||
      Math.hypot(...vector_XYZ) < 1e-12
    ) {
      throw new Error(`Quaternion alignment ${name} vector must contain three finite nonzero components.`);
    }
  }
  const fromUnit_XYZ = normalize_XYZ(from_XYZ);
  const toUnit_XYZ = normalize_XYZ(to_XYZ);
  const dot = clamp(
    fromUnit_XYZ[0] * toUnit_XYZ[0] + fromUnit_XYZ[1] * toUnit_XYZ[1] + fromUnit_XYZ[2] * toUnit_XYZ[2],
    -1,
    1,
  );
  if (dot > 1 - 1e-12) return [...IDENTITY_ROTATION];
  if (dot < -1 + 1e-12) {
    const fallback_XYZ =
      fallbackAxis_XYZ === null ? (Math.abs(fromUnit_XYZ[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]) : fallbackAxis_XYZ;
    if (
      !Array.isArray(fallback_XYZ) ||
      fallback_XYZ.length !== 3 ||
      fallback_XYZ.some((value) => !Number.isFinite(value))
    ) {
      throw new Error("Quaternion alignment fallback axis must contain three finite components.");
    }
    const projected_XYZ = fallback_XYZ.map(
      (value, axis) =>
        value -
        fromUnit_XYZ[axis] *
          (fallback_XYZ[0] * fromUnit_XYZ[0] + fallback_XYZ[1] * fromUnit_XYZ[1] + fallback_XYZ[2] * fromUnit_XYZ[2]),
    );
    if (Math.hypot(...projected_XYZ) < 1e-12) {
      throw new Error("Opposite-vector alignment needs a fallback axis not parallel to the vectors.");
    }
    return quatFromAxisAngle(projected_XYZ, 180);
  }
  const crossProduct_XYZ = cross_XYZ(fromUnit_XYZ, toUnit_XYZ);
  const quaternion = [crossProduct_XYZ[0], crossProduct_XYZ[1], crossProduct_XYZ[2], 1 + dot];
  const norm = Math.hypot(...quaternion);
  return quaternion.map((value) => value / norm);
}

export function rotateVectorByQuat(vector_XYZ, rotation) {
  const [qx, qy, qz, qw] = rotation;
  const [vx, vy, vz] = vector_XYZ;
  const tx = 2 * (qy * vz - qz * vy);
  const ty = 2 * (qz * vx - qx * vz);
  const tz = 2 * (qx * vy - qy * vx);
  return [vx + qw * tx + qy * tz - qz * ty, vy + qw * ty + qz * tx - qx * tz, vz + qw * tz + qx * ty - qy * tx];
}

export function rotationAngleDegrees(rotation) {
  return (2 * Math.acos(clamp(Math.abs(rotation[3]), 0, 1)) * 180) / Math.PI;
}

export function chordSteps(pointAt, from_ZYX, to_ZYX, tolerance, cells) {
  const steps = [];
  const cut = (low, low_ZYX, high, high_ZYX) => {
    const parts = Math.max(2, Math.ceil((high - low) * cells));
    let worst = null;
    for (let part = 1; part < parts; part++) {
      const t = low + ((high - low) * part) / parts;
      const point_ZYX = t > low && t < high ? pointAt(t) : null;
      if (point_ZYX === null) continue;
      const sag = Math.hypot(
        ...point_ZYX.map((value, axis) => value - low_ZYX[axis] - ((high_ZYX[axis] - low_ZYX[axis]) * part) / parts),
      );
      if (sag > tolerance && sag > (worst?.sag ?? 0)) worst = { t, point_ZYX, sag };
    }
    if (worst === null) {
      steps.push({ t: high, point_ZYX: high_ZYX });
      return;
    }
    cut(low, low_ZYX, worst.t, worst.point_ZYX);
    cut(worst.t, worst.point_ZYX, high, high_ZYX);
  };
  cut(0, from_ZYX, 1, to_ZYX);
  return steps;
}
