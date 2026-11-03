// The mask compiles to a texture, top in R and bottom in G.

export const SURFACE_RELIEF_MASK_SIZE = 64;
const SURFACE_RELIEF_RIDGE_LIMIT = 8;
const SURFACE_RELIEF_POINT_LIMIT = 8;

function boundaryRidges(depth) {
  return [0, 1].map((v) => ({
    v,
    points: [
      { u: 0, depth },
      { u: 1, depth },
    ],
  }));
}

export function defaultSurfaceReliefMask() {
  return { enabled: false, top: boundaryRidges(0), bottom: boundaryRidges(1) };
}

function evaluateRidge(points, u) {
  let high = 1;
  while (high < points.length - 1 && u > points[high].u) high += 1;
  const low = high - 1;
  const weight = (u - points[low].u) / (points[high].u - points[low].u);
  return points[low].depth + (points[high].depth - points[low].depth) * weight;
}

function evaluateRidges(ridges, u, v) {
  let high = 1;
  while (high < ridges.length - 1 && v > ridges[high].v) high += 1;
  const low = high - 1;
  const weight = (v - ridges[low].v) / (ridges[high].v - ridges[low].v);
  const lowDepth = evaluateRidge(ridges[low].points, u);
  const highDepth = evaluateRidge(ridges[high].points, u);
  return lowDepth + (highDepth - lowDepth) * weight;
}

function validateRidges(ridges) {
  if (!Array.isArray(ridges) || ridges.length < 2 || ridges.length > SURFACE_RELIEF_RIDGE_LIMIT) {
    throw new Error("Surface relief ridges are malformed.");
  }
  let previousV = -1;
  const admitted = ridges.map((ridge) => {
    if (
      ridge === null ||
      typeof ridge !== "object" ||
      Array.isArray(ridge) ||
      Object.keys(ridge).length !== 2 ||
      !Object.hasOwn(ridge, "v") ||
      !Object.hasOwn(ridge, "points") ||
      !Number.isFinite(ridge.v) ||
      ridge.v < 0 ||
      ridge.v > 1 ||
      ridge.v <= previousV ||
      !Array.isArray(ridge.points) ||
      ridge.points.length < 2 ||
      ridge.points.length > SURFACE_RELIEF_POINT_LIMIT
    ) {
      throw new Error("Surface relief ridges are malformed.");
    }
    previousV = ridge.v;
    let previousU = -1;
    const points = ridge.points.map((point) => {
      if (
        point === null ||
        typeof point !== "object" ||
        Array.isArray(point) ||
        Object.keys(point).length !== 2 ||
        !Object.hasOwn(point, "u") ||
        !Object.hasOwn(point, "depth") ||
        !Number.isFinite(point.u) ||
        point.u < 0 ||
        point.u > 1 ||
        point.u <= previousU ||
        !Number.isFinite(point.depth) ||
        point.depth < 0 ||
        point.depth > 1
      ) {
        throw new Error("Surface relief points are malformed.");
      }
      previousU = point.u;
      return { ...point };
    });
    if (points[0].u !== 0 || points.at(-1).u !== 1) throw new Error("Surface relief ridges must span U 0–1.");
    return { v: ridge.v, points };
  });
  if (admitted[0].v !== 0 || admitted.at(-1).v !== 1) throw new Error("Surface relief ridges must span V 0–1.");
  return admitted;
}

export function validateSurfaceReliefMask(mask) {
  if (
    mask === null ||
    typeof mask !== "object" ||
    Array.isArray(mask) ||
    Object.keys(mask).length !== 3 ||
    !Object.hasOwn(mask, "enabled") ||
    !Object.hasOwn(mask, "top") ||
    !Object.hasOwn(mask, "bottom") ||
    typeof mask.enabled !== "boolean"
  ) {
    throw new Error("Surface relief mask is malformed.");
  }
  const admitted = {
    enabled: mask.enabled,
    top: validateRidges(mask.top),
    bottom: validateRidges(mask.bottom),
  };
  const positionsU = new Set(
    [...admitted.top, ...admitted.bottom].flatMap((ridge) => ridge.points.map((point) => point.u)),
  );
  const positionsV = new Set([...admitted.top, ...admitted.bottom].map((ridge) => ridge.v));
  for (const v of positionsV) {
    for (const u of positionsU) {
      if (evaluateRidges(admitted.top, u, v) > evaluateRidges(admitted.bottom, u, v) + 1e-7) {
        throw new Error("Surface relief top boundary must not cross its bottom boundary.");
      }
    }
  }
  return admitted;
}

export function compileSurfaceReliefMask(mask, size = SURFACE_RELIEF_MASK_SIZE) {
  const admitted = validateSurfaceReliefMask(mask);
  if (!Number.isSafeInteger(size) || size < 2) throw new Error("Surface relief mask texture size is malformed.");
  const values = new Float32Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    const v = y / (size - 1);
    for (let x = 0; x < size; x++) {
      const u = x / (size - 1);
      const offset = (y * size + x) * 4;
      values[offset] = evaluateRidges(admitted.top, u, v);
      values[offset + 1] = evaluateRidges(admitted.bottom, u, v);
      values[offset + 2] = 0;
      values[offset + 3] = 1;
    }
  }
  return values;
}

export function setSurfaceReliefControl(mask, boundary, u, v, depth) {
  const admitted = validateSurfaceReliefMask(mask);
  if (!new Set(["top", "bottom"]).has(boundary)) throw new Error("Surface relief boundary is malformed.");
  if (![u, v, depth].every(Number.isFinite) || [u, v, depth].some((value) => value < 0 || value > 1)) {
    throw new Error("Surface relief control must be normalized.");
  }
  const ridges = admitted[boundary];
  let ridge = ridges.find((candidate) => Math.abs(candidate.v - v) < 0.001);
  if (ridge === undefined) {
    if (ridges.length >= SURFACE_RELIEF_RIDGE_LIMIT) throw new Error("Surface relief ridge limit reached.");
    ridge = {
      v,
      points: [
        { u: 0, depth: evaluateRidges(ridges, 0, v) },
        { u: 1, depth: evaluateRidges(ridges, 1, v) },
      ],
    };
    ridges.push(ridge);
    ridges.sort((a, b) => a.v - b.v);
  }
  const point = ridge.points.find((candidate) => Math.abs(candidate.u - u) < 0.001);
  if (point !== undefined) point.depth = depth;
  else {
    if (ridge.points.length >= SURFACE_RELIEF_POINT_LIMIT) throw new Error("Surface relief point limit reached.");
    ridge.points.push({ u, depth });
    ridge.points.sort((a, b) => a.u - b.u);
  }
  admitted.enabled = true;
  return validateSurfaceReliefMask(admitted);
}

export function removeNearestSurfaceReliefControl(mask, boundary, u, v) {
  const admitted = validateSurfaceReliefMask(mask);
  if (!new Set(["top", "bottom"]).has(boundary)) throw new Error("Surface relief boundary is malformed.");
  if (![u, v].every(Number.isFinite)) throw new Error("Surface relief position must be finite.");
  let nearest = null;
  for (const [ridgeIndex, ridge] of admitted[boundary].entries()) {
    for (const [pointIndex, point] of ridge.points.entries()) {
      if (point.u === 0 || point.u === 1) continue;
      const distance = Math.hypot(point.u - u, ridge.v - v);
      if (nearest === null || distance < nearest.distance) nearest = { ridgeIndex, pointIndex, distance };
    }
  }
  if (nearest === null) throw new Error(`The ${boundary} relief boundary has no editable controls.`);
  const ridges = admitted[boundary];
  ridges[nearest.ridgeIndex].points.splice(nearest.pointIndex, 1);
  if (ridges[nearest.ridgeIndex].points.length === 2 && !new Set([0, ridges.length - 1]).has(nearest.ridgeIndex)) {
    ridges.splice(nearest.ridgeIndex, 1);
  }
  return validateSurfaceReliefMask(admitted);
}

export function updateSurfaceReliefControl(mask, boundary, ridgeIndex, pointIndex, patch) {
  const admitted = validateSurfaceReliefMask(mask);
  if (!new Set(["top", "bottom"]).has(boundary)) throw new Error("Surface relief boundary is malformed.");
  const ridges = admitted[boundary];
  if (
    !Number.isSafeInteger(ridgeIndex) ||
    ridgeIndex < 0 ||
    ridgeIndex >= ridges.length ||
    !Number.isSafeInteger(pointIndex) ||
    pointIndex < 0 ||
    pointIndex >= ridges[ridgeIndex].points.length ||
    patch === null ||
    typeof patch !== "object" ||
    Array.isArray(patch) ||
    Object.keys(patch).length === 0 ||
    Object.keys(patch).some((key) => !new Set(["u", "v", "depth"]).has(key)) ||
    Object.values(patch).some((value) => !Number.isFinite(value) || value < 0 || value > 1)
  ) {
    throw new Error("Surface relief control update is malformed.");
  }
  const ridge = ridges[ridgeIndex];
  const point = ridge.points[pointIndex];
  if (Object.hasOwn(patch, "u") && pointIndex > 0 && pointIndex < ridge.points.length - 1) {
    point.u = Math.min(
      ridge.points[pointIndex + 1].u - 0.001,
      Math.max(ridge.points[pointIndex - 1].u + 0.001, patch.u),
    );
  }
  if (Object.hasOwn(patch, "v") && ridgeIndex > 0 && ridgeIndex < ridges.length - 1) {
    ridge.v = Math.min(ridges[ridgeIndex + 1].v - 0.001, Math.max(ridges[ridgeIndex - 1].v + 0.001, patch.v));
  }
  if (Object.hasOwn(patch, "depth")) point.depth = patch.depth;
  return validateSurfaceReliefMask(admitted);
}
