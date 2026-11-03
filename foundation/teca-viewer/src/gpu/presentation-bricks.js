import { BRICK_EDGE, MIN_BRICK_EDGE } from "../config.js";

function largestUsefulDivisor(value, brickEdge, minimum) {
  if (value <= brickEdge) return value;
  for (let divisor = brickEdge; divisor >= minimum; divisor--) {
    if (value % divisor === 0) return divisor;
  }
  return value;
}

function presentationChunkShape(sourceChunks_ZYX, downsample_ZYX) {
  if (
    !Array.isArray(sourceChunks_ZYX) ||
    sourceChunks_ZYX.length !== 3 ||
    sourceChunks_ZYX.some((value) => !Number.isSafeInteger(value) || value < 1)
  ) {
    throw new Error("Presentation chunk shape must contain three positive integers.");
  }
  if (downsample_ZYX.length !== 3 || downsample_ZYX.some((value) => !Number.isFinite(value) || value <= 0))
    throw new Error("Presentation downsampling must contain three positive numbers.");
  const maximumDownsample = Math.max(...downsample_ZYX);
  const edges_ZYX = downsample_ZYX.map((value) => Math.max(1, Math.floor((BRICK_EDGE * value) / maximumDownsample)));
  const sourceVoxels = sourceChunks_ZYX.reduce((product, value) => product * value, 1);
  if (sourceVoxels <= BRICK_EDGE ** 3 && edges_ZYX.every((edge) => edge === BRICK_EDGE)) return [...sourceChunks_ZYX];
  const chunks_ZYX = sourceChunks_ZYX.map((value, axis) =>
    largestUsefulDivisor(value, edges_ZYX[axis], edges_ZYX[axis] < BRICK_EDGE ? 1 : MIN_BRICK_EDGE),
  );
  const presentationVoxels = chunks_ZYX.reduce((product, value) => product * value, 1);
  return presentationVoxels <= BRICK_EDGE ** 3 ? chunks_ZYX : [...sourceChunks_ZYX];
}

export function presentationChunkRequest(scale, chunk_ZYX) {
  const sourceScale = scale.sourceScale ?? scale;
  if (sourceScale === scale) {
    return { scale, chunk_ZYX, localStart_ZYX: null };
  }
  const start_ZYX = chunk_ZYX.map((chunk, axis) => chunk * scale.chunks_ZYX[axis]);
  const sourceChunk_ZYX = start_ZYX.map((start, axis) => Math.floor(start / sourceScale.chunks_ZYX[axis]));
  const localStart_ZYX = start_ZYX.map((start, axis) => start - sourceChunk_ZYX[axis] * sourceScale.chunks_ZYX[axis]);
  return { scale: sourceScale, chunk_ZYX: sourceChunk_ZYX, localStart_ZYX };
}

export function copyBrick(source_DHW, sourceChunks_ZYX, chunks_ZYX, localStart_ZYX) {
  const sourceVoxels = sourceChunks_ZYX.reduce((product, value) => product * value, 1);
  if (!ArrayBuffer.isView(source_DHW) || source_DHW.length !== sourceVoxels) {
    throw new Error(`GPU presentation bricking needs one ${sourceVoxels}-voxel source chunk.`);
  }
  if (
    chunks_ZYX.some(
      (length, axis) =>
        !Number.isSafeInteger(length) ||
        length < 1 ||
        !Number.isSafeInteger(localStart_ZYX[axis]) ||
        localStart_ZYX[axis] < 0 ||
        localStart_ZYX[axis] + length > sourceChunks_ZYX[axis],
    )
  ) {
    throw new Error("GPU presentation brick falls outside its source chunk.");
  }
  const end = chunks_ZYX.reduce((product, value) => product * value, 1);
  const data_DHW = new source_DHW.constructor(end);
  const [, brickY, brickX] = chunks_ZYX;
  const [, sourceY, sourceX] = sourceChunks_ZYX;
  const [startZ, startY, startX] = localStart_ZYX;
  let voxel = 0;
  while (voxel < end) {
    const z = Math.floor(voxel / (brickY * brickX));
    const remainder = voxel - z * brickY * brickX;
    const y = Math.floor(remainder / brickX);
    const x = remainder - y * brickX;
    const contiguous =
      brickX === sourceX ? (brickY === sourceY ? end - voxel : brickY * brickX - remainder) : brickX - x;
    const count = Math.min(contiguous, end - voxel);
    const sourceOffset = (startZ + z) * sourceY * sourceX + (startY + y) * sourceX + startX + x;
    data_DHW.set(source_DHW.subarray(sourceOffset, sourceOffset + count), voxel);
    voxel += count;
  }
  return data_DHW;
}

export function presentationScale(sourceScale, downsample_ZYX = [1, 1, 1]) {
  const chunks_ZYX = presentationChunkShape(sourceScale.chunks_ZYX, downsample_ZYX);
  if (chunks_ZYX.every((value, axis) => value === sourceScale.chunks_ZYX[axis])) return sourceScale;
  return Object.freeze({
    ...sourceScale,
    chunks_ZYX: Object.freeze(chunks_ZYX),
    sourceScale,
  });
}

export function layerPresentationScales(primaryScales, sourceScales, present) {
  if (sourceScales.length < primaryScales.length)
    throw new Error(`The layer has ${sourceScales.length} levels; the CT has ${primaryScales.length}.`);
  return primaryScales.map((primaryScale, index) => {
    const scale = present(sourceScales[index]);
    const primary = present(primaryScale);
    for (const key of ["shape_ZYX", "downsample_ZYX", "offset_ZYX", "chunks_ZYX"]) {
      if (scale[key].join() !== primary[key].join())
        throw new Error(
          `Layer level ${scale.path} has ${key} ${scale[key].join(",")}; CT level ${primary.path} has ${primary[key].join(",")}.`,
        );
    }
    return scale;
  });
}
