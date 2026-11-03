import { PLANE_PREFETCH_CHUNKS_IN_FLIGHT, PREFETCH_CHUNK_PRIORITY } from "../config.js";
import { planeViewAxes } from "../planes.js";
import { ADDRESS_EXPANSION_LIMIT, toLevelSamplePoint_ZYX, visibleChunkRanges, visiblePlaneSpan } from "./addresses.js";

async function mapWithConcurrency(items, concurrency, mapper) {
  let nextIndex = 0;
  let failed = false;
  let failure;
  const workers = Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (nextIndex < items.length && !failed) {
      const item = items[nextIndex];
      nextIndex += 1;
      try {
        await mapper(item);
      } catch (error) {
        if (!failed) {
          failed = true;
          failure = error;
        }
      }
    }
  });
  await Promise.all(workers);
  if (failed) throw failure;
}

function compareChunk_ZYX(first_ZYX, second_ZYX) {
  for (let axis = 0; axis < 3; axis++) {
    if (first_ZYX[axis] !== second_ZYX[axis]) return first_ZYX[axis] - second_ZYX[axis];
  }
  return 0;
}

function planePrefetchAddresses(plane, center_ZYX, viewWidthVoxels, scale, marginChunks, viewAspect) {
  const chunks_ZYX = scale.chunks_ZYX;
  const [uAxis, vAxis] = planeViewAxes(plane);
  const [visibleU, visibleV] = visiblePlaneSpan(viewWidthVoxels, viewAspect);
  const centerInScale_ZYX = toLevelSamplePoint_ZYX(scale, center_ZYX);
  const ranges = visibleChunkRanges(
    plane,
    centerInScale_ZYX,
    visibleU / 2 / scale.downsample_ZYX[uAxis],
    visibleV / 2 / scale.downsample_ZYX[vAxis],
    scale.shape_ZYX,
    chunks_ZYX,
    marginChunks,
  );
  const addressCount =
    (ranges.z[1] - ranges.z[0] + 1) * (ranges.y[1] - ranges.y[0] + 1) * (ranges.x[1] - ranges.x[0] + 1);
  if (!Number.isSafeInteger(addressCount) || addressCount > ADDRESS_EXPANSION_LIMIT) {
    throw new Error(
      `Prefetch expands to ${addressCount} chunks, exceeding the ${ADDRESS_EXPANSION_LIMIT}-address limit.`,
    );
  }
  const targetChunk_ZYX = centerInScale_ZYX.map((value, axis) => value / chunks_ZYX[axis]);
  const addresses = [];
  for (let chunkZ = ranges.z[0]; chunkZ <= ranges.z[1]; chunkZ++) {
    for (let chunkY = ranges.y[0]; chunkY <= ranges.y[1]; chunkY++) {
      for (let chunkX = ranges.x[0]; chunkX <= ranges.x[1]; chunkX++) {
        const chunk_ZYX = [chunkZ, chunkY, chunkX];
        addresses.push({
          chunk_ZYX,
          distance: chunk_ZYX.reduce((total, value, axis) => total + Math.abs(value - targetChunk_ZYX[axis]), 0),
        });
      }
    }
  }
  return addresses.sort(
    (first, second) => first.distance - second.distance || compareChunk_ZYX(first.chunk_ZYX, second.chunk_ZYX),
  );
}

export function prefetchPlaneChunks(
  volume,
  channel,
  plane,
  center_ZYX,
  viewWidthVoxels,
  marginChunks,
  scale,
  signal,
  priorityBase,
  viewAspect,
  shouldContinue,
) {
  if (signal.aborted || !shouldContinue()) return Promise.resolve();
  const requests = planePrefetchAddresses(plane, center_ZYX, viewWidthVoxels, scale, marginChunks, viewAspect).map(
    ({ chunk_ZYX, distance }) => ({
      chunk_ZYX,
      priority: Math.min(PREFETCH_CHUNK_PRIORITY - 1, priorityBase + Math.round(distance)),
    }),
  );
  volume.retainChunkRequests(requests, scale, channel);
  return mapWithConcurrency(requests, PLANE_PREFETCH_CHUNKS_IN_FLIGHT, ({ chunk_ZYX, priority }) => {
    if (signal.aborted || !shouldContinue()) return;
    return volume.getChunk(...chunk_ZYX, channel, scale, signal, priority);
  }).catch((error) => {
    if (error.name === "AbortError" || signal.aborted) return;
    console.warn(`Prefetch failed for ${plane}:`, error);
  });
}
