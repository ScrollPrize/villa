import { SHOWN_CHUNK_PRIORITY_LIMIT } from "../config.js";
import { clamp } from "../math.js";

export const ADDRESS_EXPANSION_LIMIT = 65_536;

export function chunkKey(scale, chunk_ZYX) {
  return `${scale.path}\0${chunk_ZYX.join("/")}`;
}

export function chunkKeyParts(key) {
  const [path, chunk] = key.split("\0");
  return { path, chunk_ZYX: chunk.split("/").map(Number) };
}

export function visibleChunkPriority(distance) {
  return Math.min(SHOWN_CHUNK_PRIORITY_LIMIT - 1, Math.round(distance * 10));
}

// Level texel g holds the mean of fine voxels [g*d, g*d + d), centroid g*d + (d - 1)/2, matching the sliced-plane
// shader.
export function levelSamplePosition(scale, axis, world) {
  return (world - scale.offset_ZYX[axis] - (scale.downsample_ZYX[axis] - 1) / 2) / scale.downsample_ZYX[axis];
}

export function toLevelSamplePoint_ZYX(scale, point_ZYX) {
  return point_ZYX.map((world, axis) => levelSamplePosition(scale, axis, world));
}

export function volumeBounds_ZYX(volume) {
  const finestScale = volume.scales[0];
  return {
    minimum_ZYX: [...finestScale.offset_ZYX],
    maximum_ZYX: finestScale.offset_ZYX.map(
      (offset, axis) => offset + finestScale.shape_ZYX[axis] * finestScale.downsample_ZYX[axis],
    ),
  };
}

function xyzToZyx(vector_XYZ) {
  if (!Array.isArray(vector_XYZ) || vector_XYZ.length !== 3) {
    throw new Error("Rotated plane basis vectors must contain three values.");
  }
  return [vector_XYZ[2], vector_XYZ[1], vector_XYZ[0]];
}

function rotatedPlaneChunkBounds(center_ZYX, basis, viewWidthVoxels, scale, coverage, viewAspect) {
  const [visibleU, visibleV] = visiblePlaneSpan(viewWidthVoxels, viewAspect);
  const halfU = (visibleU * coverage) / 2;
  const halfV = (visibleV * coverage) / 2;
  const right_ZYX = xyzToZyx(basis.right_XYZ);
  const up_ZYX = xyzToZyx(basis.up_XYZ);
  const normal_ZYX = xyzToZyx(basis.normal_XYZ);
  for (const [vector_ZYX, label] of [
    [right_ZYX, "right"],
    [up_ZYX, "up"],
    [normal_ZYX, "normal"],
  ]) {
    if (vector_ZYX.some((value) => !Number.isFinite(value))) {
      throw new Error(`Rotated plane ${label} vector must contain three finite values.`);
    }
  }

  const corners_ZYX = [];
  for (const u of [-halfU, halfU])
    for (const v of [-halfV, halfV])
      corners_ZYX.push(center_ZYX.map((center, axis) => center + right_ZYX[axis] * u + up_ZYX[axis] * v));
  const ranges_ZYX = [0, 1, 2].map((axis) => {
    const minimum = Math.min(...corners_ZYX.map((corner_ZYX) => corner_ZYX[axis]));
    const maximum = Math.max(...corners_ZYX.map((corner_ZYX) => corner_ZYX[axis]));
    const minimumVoxel = Math.floor(levelSamplePosition(scale, axis, minimum));
    const maximumVoxel = Math.ceil(levelSamplePosition(scale, axis, maximum));
    const maximumChunk = Math.floor((scale.shape_ZYX[axis] - 1) / scale.chunks_ZYX[axis]);
    return [
      clamp(Math.floor(minimumVoxel / scale.chunks_ZYX[axis]), 0, maximumChunk),
      clamp(Math.floor(maximumVoxel / scale.chunks_ZYX[axis]), 0, maximumChunk),
    ];
  });
  const candidateCount = ranges_ZYX.reduce((count, [minimum, maximum]) => count * (maximum - minimum + 1), 1);
  return { ranges_ZYX, candidateCount, halfU, halfV, right_ZYX, up_ZYX, normal_ZYX };
}

export function rotatedPlaneChunkAddresses(center_ZYX, basis, viewWidthVoxels, scale, coverage, viewAspect) {
  const { ranges_ZYX, halfU, halfV, right_ZYX, up_ZYX, normal_ZYX } = rotatedPlaneChunkBounds(
    center_ZYX,
    basis,
    viewWidthVoxels,
    scale,
    coverage,
    viewAspect,
  );
  const first_ZYX = ranges_ZYX.map(([first]) => first);
  const last_ZYX = ranges_ZYX.map(([, last]) => last);
  const absoluteRight_ZYX = right_ZYX.map(Math.abs);
  const absoluteUp_ZYX = up_ZYX.map(Math.abs);
  const absoluteNormal_ZYX = normal_ZYX.map(Math.abs);
  const rounding =
    64 *
    Number.EPSILON *
    Math.max(
      1,
      ...center_ZYX.map(Math.abs),
      ...scale.offset_ZYX.map((offset, axis) => Math.abs(offset) + scale.shape_ZYX[axis] * scale.downsample_ZYX[axis]),
    );
  const addresses = [];
  const visit = () => {
    let projectedU = 0,
      projectedV = 0,
      projectedNormal = 0;
    let radiusU = 0,
      radiusV = 0,
      radiusNormal = 0;
    let splitAxis = 0;
    for (let axis = 0; axis < 3; axis++) {
      // Trilinear sampling can reach one texel below a chunk's first voxel.
      const downsample = scale.downsample_ZYX[axis];
      const start = first_ZYX[axis] * scale.chunks_ZYX[axis];
      const end = Math.min((last_ZYX[axis] + 1) * scale.chunks_ZYX[axis], scale.shape_ZYX[axis]);
      const low = scale.offset_ZYX[axis] + (start - 1) * downsample + (downsample - 1) / 2;
      const high = scale.offset_ZYX[axis] + end * downsample + (downsample - 1) / 2;
      const delta = (low + high) / 2 - center_ZYX[axis];
      const half = (high - low) / 2;
      projectedU += delta * right_ZYX[axis];
      projectedV += delta * up_ZYX[axis];
      projectedNormal += delta * normal_ZYX[axis];
      radiusU += half * absoluteRight_ZYX[axis];
      radiusV += half * absoluteUp_ZYX[axis];
      radiusNormal += half * absoluteNormal_ZYX[axis];
      if (last_ZYX[axis] - first_ZYX[axis] > last_ZYX[splitAxis] - first_ZYX[splitAxis]) splitAxis = axis;
    }
    const leaf = first_ZYX[splitAxis] === last_ZYX[splitAxis];
    const tolerance = leaf ? 0 : rounding;
    if (
      Math.abs(projectedU) > halfU + radiusU + tolerance ||
      Math.abs(projectedV) > halfV + radiusV + tolerance ||
      Math.abs(projectedNormal) > radiusNormal + tolerance
    )
      return;
    if (leaf) {
      if (addresses.length === ADDRESS_EXPANSION_LIMIT)
        throw new RangeError(
          `Rotated view intersects more than ${ADDRESS_EXPANSION_LIMIT} chunks, exceeding the ${ADDRESS_EXPANSION_LIMIT}-address limit.`,
        );
      addresses.push({
        chunk_ZYX: [...first_ZYX],
        priority: Math.hypot(
          projectedU / Math.max(radiusU, 1),
          projectedV / Math.max(radiusV, 1),
          projectedNormal / Math.max(radiusNormal, 1),
        ),
      });
      return;
    }
    const first = first_ZYX[splitAxis],
      last = last_ZYX[splitAxis];
    const middle = first + Math.floor((last - first) / 2);
    last_ZYX[splitAxis] = middle;
    visit();
    first_ZYX[splitAxis] = middle + 1;
    last_ZYX[splitAxis] = last;
    visit();
    first_ZYX[splitAxis] = first;
  };
  visit();
  addresses.sort(
    (a, b) =>
      a.priority - b.priority ||
      a.chunk_ZYX[0] - b.chunk_ZYX[0] ||
      a.chunk_ZYX[1] - b.chunk_ZYX[1] ||
      a.chunk_ZYX[2] - b.chunk_ZYX[2],
  );
  return addresses;
}

function fixedChunkRange(center, chunkSize, axisSize) {
  const chunk = Math.floor(Math.round(center) / chunkSize);
  const maxChunk = Math.floor((axisSize - 1) / chunkSize);
  return [clamp(chunk, 0, maxChunk), clamp(chunk, 0, maxChunk)];
}

function spanChunkRange(min, max, chunkSize, axisSize, marginChunks) {
  const maxChunk = Math.floor((axisSize - 1) / chunkSize);
  return [
    clamp(Math.floor(min / chunkSize) - marginChunks, 0, maxChunk),
    clamp(Math.floor(max / chunkSize) + marginChunks, 0, maxChunk),
  ];
}

export function visibleChunkRanges(plane, center_ZYX, halfU, halfV, shape_ZYX, chunks_ZYX, marginChunks) {
  const ranges = {
    z: fixedChunkRange(center_ZYX[0], chunks_ZYX[0], shape_ZYX[0]),
    y: fixedChunkRange(center_ZYX[1], chunks_ZYX[1], shape_ZYX[1]),
    x: fixedChunkRange(center_ZYX[2], chunks_ZYX[2], shape_ZYX[2]),
  };
  if (plane === "xy") {
    ranges.y = spanChunkRange(center_ZYX[1] - halfV, center_ZYX[1] + halfV, chunks_ZYX[1], shape_ZYX[1], marginChunks);
    ranges.x = spanChunkRange(center_ZYX[2] - halfU, center_ZYX[2] + halfU, chunks_ZYX[2], shape_ZYX[2], marginChunks);
  } else if (plane === "zy") {
    ranges.z = spanChunkRange(center_ZYX[0] - halfU, center_ZYX[0] + halfU, chunks_ZYX[0], shape_ZYX[0], marginChunks);
    ranges.y = spanChunkRange(center_ZYX[1] - halfV, center_ZYX[1] + halfV, chunks_ZYX[1], shape_ZYX[1], marginChunks);
  } else if (plane === "xz") {
    ranges.z = spanChunkRange(center_ZYX[0] - halfV, center_ZYX[0] + halfV, chunks_ZYX[0], shape_ZYX[0], marginChunks);
    ranges.x = spanChunkRange(center_ZYX[2] - halfU, center_ZYX[2] + halfU, chunks_ZYX[2], shape_ZYX[2], marginChunks);
  } else {
    throw new Error(`Unknown plane: ${plane}`);
  }
  return ranges;
}

export function visiblePlaneSpan(viewWidthVoxels, viewAspect = 1) {
  const aspect = Math.max(0.001, viewAspect);
  return [viewWidthVoxels * (aspect >= 1 ? aspect : 1), viewWidthVoxels * (aspect >= 1 ? 1 : 1 / aspect)];
}
