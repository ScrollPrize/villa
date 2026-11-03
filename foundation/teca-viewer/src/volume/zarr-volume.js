import {
  CACHE_COARSE_LEVEL_OFFSET,
  CACHE_COARSE_RESIDENCY_FRACTION,
  CACHE_TRAILING_NAVIGATION_GENERATIONS,
  CHUNK_JOBS_IN_FLIGHT,
  CHUNK_STAGING_BYTES,
  SHOWN_CHUNK_PRIORITY_LIMIT,
} from "../config.js";
import { displaySamples } from "./display-window.js";
import { declaredDatasetPaths, readVolumeDocument } from "./ngff.js";
import { QueuedZarrStore } from "./queued-zarr-store.js";
import { abortable, MirrorRequestQueue, PriorityRequestQueue } from "./request-queue.js";
import { SourceDecoder } from "./source-decoder.js";
import { DirectZarrStore } from "./zarr-store.js";
import { openDirectZarrV2 } from "./zarr-v2.js";
import { openDirectZarrV3 } from "./zarr-v3.js";

function storageChunkIndices(chunkZ, chunkY, chunkX, channel, scale) {
  const chunk = { t: 0, z: chunkZ, y: chunkY, x: chunkX };
  return scale.axes.map((axis, axisIndex) =>
    axis === "c" ? Math.floor(channel / scale.storageChunks[axisIndex]) : chunk[axis],
  );
}

function validateCacheByteLimit(byteLimit, name) {
  if (!Number.isSafeInteger(byteLimit) || byteLimit <= 0) {
    throw new Error(`${name} byte limit must be a positive safe integer.`);
  }
}

function product(values) {
  return values.reduce((result, value) => result * value, 1);
}

function extractSpatialChunk(storageValues, scale, selectedChannel) {
  if (scale.axes.join(",") === "z,y,x") return storageValues;
  const strides = new Array(scale.storageChunks.length);
  let stride = 1;
  for (let axis = scale.storageChunks.length - 1; axis >= 0; axis--) {
    strides[axis] = stride;
    stride *= scale.storageChunks[axis];
  }
  const channelAxis = scale.axes.indexOf("c");
  const channelOffset =
    channelAxis === -1 ? 0 : (selectedChannel % scale.storageChunks[channelAxis]) * strides[channelAxis];
  const spatialVoxels = product(scale.chunks_ZYX);
  if (scale.axes.slice(-3).join(",") === "z,y,x") {
    return storageValues.subarray(channelOffset, channelOffset + spatialVoxels);
  }
  const output = new storageValues.constructor(spatialVoxels);
  const zStride = strides[scale.axes.indexOf("z")] ?? 0;
  const yStride = strides[scale.axes.indexOf("y")] ?? 0;
  const xStride = strides[scale.axes.indexOf("x")] ?? 0;
  let outputIndex = 0;
  for (let z = 0; z < scale.chunks_ZYX[0]; z++) {
    const planeOffset = channelOffset + z * zStride;
    for (let y = 0; y < scale.chunks_ZYX[1]; y++) {
      const rowOffset = planeOffset + y * yStride;
      for (let x = 0; x < scale.chunks_ZYX[2]; x++) {
        output[outputIndex++] = storageValues[rowOffset + x * xStride];
      }
    }
  }
  return output;
}

function withTrailingSlash(url) {
  const normalized = new URL(url);
  if (!normalized.pathname.endsWith("/")) normalized.pathname += "/";
  return normalized.href;
}

export class ZarrVolume {
  constructor(rootUrl, scales, metadata, chunkCacheByteLimit, opened, byteStore, downloads) {
    this.arrays = opened.arrays;
    this.encodedByteLimits = opened.encodedByteLimits;
    this.decoder = opened.decoder ?? null;
    this.resetDecoders = opened.resetDecoders ?? null;
    this.metadataReservedBytes = opened.metadataByteLength ?? 0;
    if (this.metadataReservedBytes >= chunkCacheByteLimit) {
      throw new Error("Zarr metadata leaves no capacity in the source byte budget.");
    }
    this.sourceByteLimit = chunkCacheByteLimit;
    this.rootUrl = rootUrl;
    this.scales = scales.sort((a, b) => Math.max(...a.downsample_ZYX) - Math.max(...b.downsample_ZYX));
    this.metadata = metadata;
    this.chunkCache = new Map();
    this.fillChunks = opened.fillChunks ?? new WeakSet();
    this.chunkCacheBytes = 0;
    this.chunkCacheByteLimit = chunkCacheByteLimit - this.metadataReservedBytes;
    this.navigationGeneration = 0;
    this.byteStore = byteStore;
    this.downloads = downloads;
    this.chunkQueue = new PriorityRequestQueue(CHUNK_JOBS_IN_FLIGHT, CHUNK_JOBS_IN_FLIGHT, CHUNK_STAGING_BYTES);
    this.dispatchGroup = {};
  }

  static async open(rootUrl, signal, chunkCacheByteLimit, downloadStorage, downloads) {
    validateCacheByteLimit(chunkCacheByteLimit, "Chunk cache");
    const normalizedRootUrl = withTrailingSlash(rootUrl);
    const source = new DirectZarrStore(normalizedRootUrl, downloadStorage, downloads);
    const networkQueue = new MirrorRequestQueue(normalizedRootUrl);
    const store = new QueuedZarrStore(source, networkQueue);
    let volume = null;
    try {
      const attrsBytes = await store.get("/.zattrs", { signal });
      if (attrsBytes === undefined) {
        const directV3 = await openDirectZarrV3(store, signal);
        if (directV3 === null) throw new Error(`No Zarr metadata exists at ${normalizedRootUrl}.`);
        const built = readVolumeDocument(directV3.attrs, directV3.zarrays);
        volume = new ZarrVolume(
          normalizedRootUrl,
          built.scales,
          built.metadata,
          chunkCacheByteLimit,
          directV3,
          store,
          downloads,
        );
      } else {
        const attrs = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(attrsBytes));
        const datasetPaths = declaredDatasetPaths(attrs);
        const zarrays = Object.fromEntries(
          await Promise.all(
            datasetPaths.map(async (path) => {
              const bytes = await store.get(`/${path}/.zarray`, { signal });
              if (bytes === undefined) throw new Error(`Array metadata for ${path} is missing.`);
              return [path, JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes))];
            }),
          ),
        );
        const built = readVolumeDocument(attrs, zarrays);
        volume = new ZarrVolume(
          normalizedRootUrl,
          built.scales,
          built.metadata,
          chunkCacheByteLimit,
          await openDirectZarrV2(built.scales, store),
          store,
          downloads,
        );
      }
      await volume.initializeDisplay(signal);
      return volume;
    } catch (error) {
      if (volume === null) {
        networkQueue.dispose();
        source.dispose();
      } else volume.dispose();
      throw error;
    }
  }

  async initializeDisplay(signal) {
    if (this.metadata.displayEncoding !== null) return;
    const samples = await displaySamples(this, signal);
    this.decoder ??= new SourceDecoder();
    const encoding = await this.decoder.run({ kind: "window", samples });
    signal?.throwIfAborted();
    this.metadata.displayEncoding = encoding;
  }

  getQueueStats() {
    const network = this.byteStore.queue.getStats();
    return { ...network, pending: network.pending + this.chunkQueue.pending.length };
  }

  setChunkCacheByteLimit(byteLimit) {
    validateCacheByteLimit(byteLimit, "Chunk cache");
    if (this.metadataReservedBytes >= byteLimit) {
      throw new Error("Zarr metadata leaves no capacity in the source byte budget.");
    }
    this.sourceByteLimit = byteLimit;
    this.chunkCacheByteLimit = byteLimit - this.metadataReservedBytes;
    this.evictCachedChunks(null);
  }

  clearChunkCache() {
    this.chunkQueue.cancelPending(0);
    let abandonedBytes = 0;
    for (const request of this.chunkCache.values()) {
      request.discarded = true;
      if (request.started) abandonedBytes += this.abandonRequest(request);
    }
    this.chunkCache.clear();
    this.resetDecoders?.();
    this.chunkCacheBytes = 0;
    this.downloads.abandonmentEvent(abandonedBytes);
  }

  beginNavigation(generation) {
    if (!Number.isInteger(generation) || generation < this.navigationGeneration) {
      throw new Error("Chunk-cache navigation generation must be monotone.");
    }
    this.navigationGeneration = generation;
    this.dispatchGroup = {};
  }

  dispose() {
    this.decoder?.dispose();
    this.clearChunkCache();
    this.chunkQueue.dispose();
    this.byteStore.queue.dispose();
    this.byteStore.store.dispose();
  }

  cancelPendingChunks(minPriority, maxPriority = Infinity, preserveCurrent = false) {
    const isCurrent = (request) => preserveCurrent && request.lastRequestedNavigation === this.navigationGeneration;
    this.chunkQueue.cancelPending(minPriority, maxPriority, (request) => !isCurrent(request));
    this.byteStore.queue.cancelPending(minPriority, maxPriority, (request) => !isCurrent(request.chunkRequest));
  }

  get fullShape_ZYX() {
    return this.scales[0].shape_ZYX;
  }

  pickScaleIndexForView(viewWidthVoxels, textureSize, scaleTolerance) {
    const voxelsPerPixel = Math.max(1, viewWidthVoxels / textureSize) * Math.max(1, scaleTolerance);
    let chosen = 0;
    for (let index = 0; index < this.scales.length; index++) {
      const downsample = Math.max(...this.scales[index].downsample_ZYX);
      if (downsample <= voxelsPerPixel) {
        chosen = index;
      }
    }
    return chosen;
  }

  async getChunk(chunkZ, chunkY, chunkX, channel, scale, signal, priority) {
    const requestScaleIndex = this.scales.indexOf(scale);
    if (requestScaleIndex < 0) throw new Error(`Chunk scale ${scale.path} does not belong to this volume.`);
    const key = this.chunkCacheKey(chunkZ, chunkY, chunkX, channel, scale);
    let request = this.chunkCache.get(key);
    if (request?.aborted) {
      this.deleteCachedChunk(key, request);
      request = null;
    }
    if (request) {
      request.promote(priority, this.dispatchGroup);
      request.lastRequestedNavigation = this.navigationGeneration;
      this.chunkCache.delete(key);
      this.chunkCache.set(key, request);
    } else {
      const fetchController = new AbortController();
      const expectedByteLength = product(scale.storageChunks) * scale.numericType.itemSize;
      const encodedByteLimit = Math.max(expectedByteLength, this.encodedByteLimits.get(scale.path));
      request = this.chunkQueue.enqueue(
        async () => {
          await Promise.resolve();
          fetchController.signal.throwIfAborted();
          this.byteStore.requests.set(fetchController.signal, request);
          const storageValues = await this.arrays
            .get(scale.path)
            .getChunk(storageChunkIndices(chunkZ, chunkY, chunkX, channel, scale), scale, fetchController.signal);
          fetchController.signal.throwIfAborted();
          if (request.discarded) return storageValues;
          request.byteLength = storageValues.byteLength;
          request.values = storageValues;
          this.chunkCacheBytes += request.byteLength;
          this.evictCachedChunks(key);
          return storageValues;
        },
        priority,
        null,
        this.dispatchGroup,
        expectedByteLength + encodedByteLimit,
      );
      request.reads = new Set();
      const promote = request.promote;
      request.promote = (priority, group) => {
        promote(priority, group);
        for (const read of request.reads) read.promote(priority, group);
      };
      request.fetchController = fetchController;
      request.discarded = false;
      request.lastRequestedNavigation = this.navigationGeneration;
      request.expectedByteLength = encodedByteLimit;
      request.scaleIndex = requestScaleIndex;
      request.lastRenderedGeneration = null;
      request.byteLength = 0;
      request.spatialChunk = null;
      request.promise = request.promise
        .catch((error) => {
          fetchController.abort();
          if (this.chunkCache.get(key) === request) {
            this.deleteCachedChunk(key, request);
          }
          if (error.name === "AbortError") throw error;
          throw new Error(`Failed to fetch chunk ${key}: ${error.message}`);
        })
        .finally(() => this.byteStore.requests.delete(fetchController.signal));
      this.chunkCache.set(key, request);
    }

    const storageValues = await (signal ? abortable(request.promise, signal) : request.promise);
    if (priority < SHOWN_CHUNK_PRIORITY_LIMIT && !signal?.aborted && this.chunkCache.get(key) === request) {
      request.lastRenderedGeneration = this.navigationGeneration;
      this.chunkCache.delete(key);
      this.chunkCache.set(key, request);
    }
    if (scale.axes.join(",") === "z,y,x") return storageValues;
    if (request.spatialChunk?.channel === channel) return request.spatialChunk.values_DHW;
    const values_DHW = extractSpatialChunk(storageValues, scale, channel);
    if (this.fillChunks.has(storageValues)) this.fillChunks.add(values_DHW);
    const retainedBytes = values_DHW.buffer === storageValues.buffer ? 0 : values_DHW.byteLength;
    if (
      this.chunkCache.get(key) === request &&
      !request.discarded &&
      request.byteLength - (request.spatialChunk?.retainedBytes ?? 0) + retainedBytes <= this.chunkCacheByteLimit
    ) {
      const previousBytes = request.spatialChunk?.retainedBytes ?? 0;
      request.spatialChunk = { channel, values_DHW, retainedBytes };
      request.byteLength += retainedBytes - previousBytes;
      this.chunkCacheBytes += retainedBytes - previousBytes;
      this.evictCachedChunks(key);
    }
    return values_DHW;
  }

  isFill(chunk_DHW) {
    return this.fillChunks.has(chunk_DHW);
  }

  heldChunk(chunkZ, chunkY, chunkX, channel, scale) {
    const request = this.chunkCache.get(this.chunkCacheKey(chunkZ, chunkY, chunkX, channel, scale));
    if (request?.values === undefined) return null;
    if (scale.axes.join(",") === "z,y,x") return request.values;
    if (request.spatialChunk?.channel === channel) return request.spatialChunk.values_DHW;
    return extractSpatialChunk(request.values, scale, channel);
  }

  retainChunkRequests(requests, scale, channel) {
    for (const { chunk_ZYX, priority } of requests) {
      const request = this.chunkCache.get(this.chunkCacheKey(...chunk_ZYX, channel, scale));
      if (request === undefined || request.aborted) continue;
      request.promote(priority, this.dispatchGroup);
      request.lastRequestedNavigation = this.navigationGeneration;
    }
  }

  chunkCacheKey(chunkZ, chunkY, chunkX, channel, scale) {
    const storageIndices = storageChunkIndices(chunkZ, chunkY, chunkX, channel, scale);
    return `${scale.path}/${storageIndices.join("/")}`;
  }

  abandonRequest(request) {
    if (request.byteLength > 0 || request.abandoned) {
      request.fetchController.abort();
      return 0;
    }
    request.abandoned = true;
    this.downloads.abandon(request.expectedByteLength);
    request.fetchController.abort();
    return request.expectedByteLength;
  }

  evictCachedChunks(retainKey) {
    if (this.chunkCacheBytes <= this.chunkCacheByteLimit) return;
    const recentCutoff = this.navigationGeneration - CACHE_TRAILING_NAVIGATION_GENERATIONS;
    let finestRenderedScaleIndex = null;
    for (const request of this.chunkCache.values()) {
      if (request.lastRenderedGeneration !== null && request.lastRenderedGeneration >= recentCutoff)
        finestRenderedScaleIndex = Math.min(finestRenderedScaleIndex ?? Infinity, request.scaleIndex);
    }
    const isCoarse = (request) =>
      finestRenderedScaleIndex !== null && request.scaleIndex >= finestRenderedScaleIndex + CACHE_COARSE_LEVEL_OFFSET;
    const isRecent = (request) =>
      request.lastRenderedGeneration !== null && request.lastRenderedGeneration >= recentCutoff;
    let coarseBytes = 0;
    for (const request of this.chunkCache.values()) if (isCoarse(request)) coarseBytes += request.byteLength;
    const evictPass = (eligible) => {
      for (const [key, request] of this.chunkCache) {
        if (this.chunkCacheBytes <= this.chunkCacheByteLimit) break;
        if (key === retainKey || request.byteLength === 0 || !eligible(request)) continue;
        this.deleteCachedChunk(key, request);
        if (isCoarse(request)) coarseBytes -= request.byteLength;
      }
    };
    evictPass((request) => !isCoarse(request) && !isRecent(request));
    evictPass((request) => !isCoarse(request) && isRecent(request));
    const coarseExceedsShare = () => coarseBytes > this.chunkCacheByteLimit * CACHE_COARSE_RESIDENCY_FRACTION;
    evictPass((request) => isCoarse(request) && !isRecent(request) && coarseExceedsShare());
    evictPass((request) => isCoarse(request) && isRecent(request) && coarseExceedsShare());
  }

  deleteCachedChunk(key, request) {
    this.chunkCache.delete(key);
    this.chunkCacheBytes -= request.byteLength;
  }
}
