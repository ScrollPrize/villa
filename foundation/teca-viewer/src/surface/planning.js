import { ADDRESS_EXPANSION_LIMIT, levelSamplePosition } from "../gpu/addresses.js";
import { clamp } from "../math.js";

export function visibleSurfaceStrips({ center_UV, visibleSpan_UV, grid_HW, stripRows }) {
  const minimumY = clamp(Math.floor(center_UV[1] - visibleSpan_UV[1] / 2) - 1, 0, grid_HW[0] - 1);
  const maximumY = clamp(Math.ceil(center_UV[1] + visibleSpan_UV[1] / 2) + 1, 0, grid_HW[0] - 1);
  const first = Math.floor(minimumY / stripRows);
  const last = Math.floor(maximumY / stripRows);
  const center = center_UV[1] / stripRows;
  return Array.from({ length: last - first + 1 }, (_, offset) => first + offset).sort(
    (a, b) => Math.abs(a - center) - Math.abs(b - center),
  );
}

function accumulateRequiredChunks(required, grid_ZYX, ranges_6, limit) {
  let rangeCount = 1;
  for (let axis = 0; axis < 3; axis++) rangeCount *= ranges_6[axis * 2 + 1] - ranges_6[axis * 2] + 1;
  if (rangeCount > limit) return false;
  for (let z = ranges_6[0]; z <= ranges_6[1]; z++) {
    for (let y = ranges_6[2]; y <= ranges_6[3]; y++) {
      for (let x = ranges_6[4]; x <= ranges_6[5]; x++) {
        required.add((z * grid_ZYX[1] + y) * grid_ZYX[2] + x);
        if (required.size > limit) return false;
      }
    }
  }
  return true;
}

function chunkDistance(chunk_ZYX, center_ZYX) {
  return Math.hypot(chunk_ZYX[0] - center_ZYX[0], chunk_ZYX[1] - center_ZYX[1], chunk_ZYX[2] - center_ZYX[2]);
}

function heapPush(heap, candidate, compare) {
  heap.push(candidate);
  let index = heap.length - 1;
  while (index > 0) {
    const parent = Math.floor((index - 1) / 2);
    if (compare(heap[parent], candidate) <= 0) break;
    heap[index] = heap[parent];
    index = parent;
  }
  heap[index] = candidate;
}

function sinkHeap(heap, start, compare) {
  let index = start;
  const candidate = heap[index];
  while (true) {
    const left = index * 2 + 1;
    if (left >= heap.length) break;
    const right = left + 1;
    const child = right < heap.length && compare(heap[right], heap[left]) < 0 ? right : left;
    if (compare(candidate, heap[child]) <= 0) break;
    heap[index] = heap[child];
    index = child;
  }
  heap[index] = candidate;
}

function heapPop(heap, compare) {
  const first = heap[0];
  const last = heap.pop();
  if (heap.length > 0) {
    heap[0] = last;
    sinkHeap(heap, 0, compare);
  }
  return first;
}

function compareChunks(first_ZYX, second_ZYX) {
  for (let axis = 0; axis < 3; axis++) {
    if (first_ZYX[axis] !== second_ZYX[axis]) return first_ZYX[axis] - second_ZYX[axis];
  }
  return 0;
}

function compareRangeCandidates(first, second) {
  return first.distance - second.distance || compareChunks(first.chunk_ZYX, second.chunk_ZYX);
}

function* nearestRangeChunks(ranges_6, limit) {
  const center_ZYX = [
    (ranges_6[0] + ranges_6[1]) / 2,
    (ranges_6[2] + ranges_6[3]) / 2,
    (ranges_6[4] + ranges_6[5]) / 2,
  ];
  const countInRange =
    (ranges_6[1] - ranges_6[0] + 1) * (ranges_6[3] - ranges_6[2] + 1) * (ranges_6[5] - ranges_6[4] + 1);
  if (countInRange <= limit) {
    const candidates = [];
    for (let z = ranges_6[0]; z <= ranges_6[1]; z++)
      for (let y = ranges_6[2]; y <= ranges_6[3]; y++)
        for (let x = ranges_6[4]; x <= ranges_6[5]; x++) {
          const chunk_ZYX = [z, y, x];
          candidates.push({ chunk_ZYX, distance: chunkDistance(chunk_ZYX, center_ZYX) });
        }
    yield* candidates.sort(compareRangeCandidates);
    return;
  }
  const start_ZYX = center_ZYX.map((center, axis) =>
    clamp(Math.round(center), ranges_6[axis * 2], ranges_6[axis * 2 + 1]),
  );
  const heap = [{ chunk_ZYX: start_ZYX, distance: chunkDistance(start_ZYX, center_ZYX) }];
  const queued = new Set([start_ZYX.join("/")]);
  let count = 0;
  while (heap.length > 0 && count < limit) {
    const candidate = heapPop(heap, compareRangeCandidates);
    count += 1;
    yield candidate;
    for (let axis = 0; axis < 3; axis++) {
      for (const direction of [-1, 1]) {
        const chunk_ZYX = [...candidate.chunk_ZYX];
        chunk_ZYX[axis] += direction;
        if (chunk_ZYX[axis] < ranges_6[axis * 2] || chunk_ZYX[axis] > ranges_6[axis * 2 + 1]) continue;
        const key = chunk_ZYX.join("/");
        if (queued.has(key)) continue;
        queued.add(key);
        heapPush(heap, { chunk_ZYX, distance: chunkDistance(chunk_ZYX, center_ZYX) }, compareRangeCandidates);
      }
    }
  }
}

function compareAddresses(first, second) {
  return first.priority - second.priority || compareChunks(first.chunk_ZYX, second.chunk_ZYX);
}

function addressSelection(limit) {
  return { limit, heap: [], indices: new Map() };
}

function sinkAddress(selection, start) {
  let index = start;
  const candidate = selection.heap[index];
  while (true) {
    const left = index * 2 + 1;
    if (left >= selection.heap.length) break;
    const right = left + 1;
    const child =
      right < selection.heap.length && compareAddresses(selection.heap[right], selection.heap[left]) > 0 ? right : left;
    if (compareAddresses(candidate, selection.heap[child]) >= 0) break;
    selection.heap[index] = selection.heap[child];
    selection.indices.set(selection.heap[index].key, index);
    index = child;
  }
  selection.heap[index] = candidate;
  selection.indices.set(candidate.key, index);
}

function offerAddress(selection, candidate) {
  const currentIndex = selection.indices.get(candidate.key);
  if (currentIndex !== undefined) {
    if (compareAddresses(candidate, selection.heap[currentIndex]) >= 0) return;
    selection.heap[currentIndex] = candidate;
    if (selection.heap.length === selection.limit) sinkAddress(selection, currentIndex);
    return;
  }
  if (selection.heap.length < selection.limit) {
    const index = selection.heap.length;
    selection.heap.push(candidate);
    selection.indices.set(candidate.key, index);
    if (selection.heap.length === selection.limit)
      for (let parent = Math.floor(selection.limit / 2) - 1; parent >= 0; parent--) sinkAddress(selection, parent);
    return;
  }
  if (compareAddresses(candidate, selection.heap[0]) >= 0) return;
  selection.indices.delete(selection.heap[0].key);
  selection.heap[0] = candidate;
  selection.indices.set(candidate.key, 0);
  sinkAddress(selection, 0);
}

function selectSurfaceChunks(grid_ZYX, visitRanges, maximumAddresses) {
  const selection = addressSelection(maximumAddresses);
  const required = new Set();
  const coverageLimit = Math.min(ADDRESS_EXPANSION_LIMIT, maximumAddresses + 256);
  let coverageComplete = true;
  visitRanges((ranges) => {
    for (const range of ranges) {
      if (coverageComplete)
        coverageComplete = accumulateRequiredChunks(required, grid_ZYX, range.ranges_6, coverageLimit);
      const basePriority = range.surfacePriority * 1024;
      if (selection.heap.length === maximumAddresses && basePriority > selection.heap[0].priority) continue;
      const [z0, z1, y0, y1, x0, x1] = range.ranges_6;
      if (selection.heap.length + (z1 - z0 + 1) * (y1 - y0 + 1) * (x1 - x0 + 1) <= maximumAddresses) {
        for (let z = z0; z <= z1; z++)
          for (let y = y0; y <= y1; y++)
            for (let x = x0; x <= x1; x++)
              offerAddress(selection, {
                key: (z * grid_ZYX[1] + y) * grid_ZYX[2] + x,
                chunk_ZYX: [z, y, x],
                priority: basePriority + Math.hypot(z - (z0 + z1) / 2, y - (y0 + y1) / 2, x - (x0 + x1) / 2),
              });
        continue;
      }
      for (const candidate of nearestRangeChunks(range.ranges_6, maximumAddresses)) {
        const address = {
          key: (candidate.chunk_ZYX[0] * grid_ZYX[1] + candidate.chunk_ZYX[1]) * grid_ZYX[2] + candidate.chunk_ZYX[2],
          chunk_ZYX: candidate.chunk_ZYX,
          priority: basePriority + candidate.distance,
        };
        if (selection.heap.length === maximumAddresses && compareAddresses(address, selection.heap[0]) >= 0) {
          break;
        }
        offerAddress(selection, address);
      }
    }
  });
  return {
    addresses: selection.heap.map(({ chunk_ZYX, priority }) => ({ chunk_ZYX, priority })).sort(compareAddresses),
    complete: coverageComplete && required.size <= maximumAddresses,
    requiredCount: coverageComplete ? required.size : null,
    requiredCountLowerBound: coverageComplete ? required.size : Math.max(required.size, coverageLimit + 1),
  };
}

function fillChunkRanges(ranges_6, bounds_ZYX6, registration, radius, scale) {
  for (let axis = 0; axis < 3; axis++) {
    const nativeMinimum = bounds_ZYX6[axis];
    const nativeMaximum = bounds_ZYX6[3 + axis];
    if (!Number.isFinite(nativeMinimum) || !Number.isFinite(nativeMaximum)) return false;
    const minimum = nativeMinimum * registration.scale_ZYX[axis] + registration.offset_ZYX[axis] - radius;
    const maximum = nativeMaximum * registration.scale_ZYX[axis] + registration.offset_ZYX[axis] + radius;
    const low = Math.floor(levelSamplePosition(scale, axis, minimum));
    const high = Math.floor(levelSamplePosition(scale, axis, maximum)) + 1;
    const maximumChunk = Math.floor((scale.shape_ZYX[axis] - 1) / scale.chunks_ZYX[axis]);
    ranges_6[axis * 2] = clamp(Math.floor(low / scale.chunks_ZYX[axis]), 0, maximumChunk);
    ranges_6[axis * 2 + 1] = clamp(Math.floor(high / scale.chunks_ZYX[axis]), 0, maximumChunk);
  }
  return true;
}

function visibleChunkRanges(surfaceBounds, registration, radius, scale, consume) {
  const rangesByKey = new Map();
  const ranges_6 = new Int32Array(6);
  let previous = null;
  let streamed = false;
  for (const { bounds_ZYX6, surfacePriority } of surfaceBounds) {
    if (!fillChunkRanges(ranges_6, bounds_ZYX6, registration, radius, scale)) continue;
    let current = previous;
    if (
      current === null ||
      ranges_6[0] !== current.ranges_6[0] ||
      ranges_6[1] !== current.ranges_6[1] ||
      ranges_6[2] !== current.ranges_6[2] ||
      ranges_6[3] !== current.ranges_6[3] ||
      ranges_6[4] !== current.ranges_6[4] ||
      ranges_6[5] !== current.ranges_6[5]
    ) {
      const key = `${ranges_6[0]}/${ranges_6[1]}/${ranges_6[2]}/${ranges_6[3]}/${ranges_6[4]}/${ranges_6[5]}`;
      current = rangesByKey.get(key);
      if (current === undefined) {
        if (rangesByKey.size === ADDRESS_EXPANSION_LIMIT) {
          consume(rangesByKey.values());
          rangesByKey.clear();
          streamed = true;
        }
        current = { ranges_6: new Int32Array(ranges_6), surfacePriority: Infinity };
        rangesByKey.set(key, current);
      }
      previous = current;
    }
    current.surfacePriority = Math.min(current.surfacePriority, surfacePriority);
  }
  consume(rangesByKey.values());
  return streamed ? null : rangesByKey;
}

function surfaceVolumePlanner({ surfaceBounds, registration, normalRadius }) {
  const rangesByScale = new Map();
  return (scale, maximumAddresses) => {
    if (!Number.isSafeInteger(maximumAddresses) || maximumAddresses < 1 || maximumAddresses > ADDRESS_EXPANSION_LIMIT) {
      throw new Error(`Surface address capacity must be inside 1..${ADDRESS_EXPANSION_LIMIT}.`);
    }
    const grid_ZYX = scale.shape_ZYX.map((size, axis) => Math.floor((size - 1) / scale.chunks_ZYX[axis]) + 1);
    return selectSurfaceChunks(
      grid_ZYX,
      (consume) => {
        const cached = rangesByScale.get(scale);
        if (cached !== undefined) {
          consume(cached.values());
          return;
        }
        const ranges = visibleChunkRanges(surfaceBounds, registration, normalRadius, scale, consume);
        if (ranges !== null) rangesByScale.set(scale, ranges);
      },
      maximumAddresses,
    );
  };
}

export function packSurfaceDemandBounds(bounds) {
  let data_B7 = new Float64Array(1024 * 7);
  let offset = 0;
  for (const { bounds_ZYX6, surfacePriority } of bounds) {
    if (offset === data_B7.length) {
      const grown_B7 = new Float64Array(data_B7.length * 2);
      grown_B7.set(data_B7);
      data_B7 = grown_B7;
    }
    data_B7.set(bounds_ZYX6, offset);
    data_B7[offset + 6] = surfacePriority;
    offset += 7;
  }
  return data_B7.slice(0, offset);
}

export function planSurfaceDemand({ options: { surfaceBounds_B7, ...options }, scales, capacities }) {
  const surfaceBounds = {
    *[Symbol.iterator]() {
      for (let offset = 0; offset < surfaceBounds_B7.length; offset += 7)
        yield {
          bounds_ZYX6: surfaceBounds_B7.subarray(offset, offset + 6),
          surfacePriority: surfaceBounds_B7[offset + 6],
        };
    },
  };
  const plan = surfaceVolumePlanner({ ...options, surfaceBounds });
  return scales.map((scale, index) => plan(scale, capacities[index]));
}
