import { WORKER_CACHE_BYTE_LIMIT } from "../config.js";
import { copyBrick } from "./presentation-bricks.js";

const chunks = new Map();
let cacheBytes = 0;

self.onmessage = ({ data }) => {
  const { id, chunkCacheKey, sourceChunks_ZYX, chunks_ZYX, localStart_ZYX } = data;
  try {
    let chunk = data.chunk;
    if (chunk !== undefined) {
      const previous = chunks.get(chunkCacheKey);
      if (previous !== undefined) cacheBytes -= previous.buffer.byteLength;
      chunks.delete(chunkCacheKey);
      chunks.set(chunkCacheKey, chunk);
      cacheBytes += chunk.buffer.byteLength;
      while (cacheBytes > WORKER_CACHE_BYTE_LIMIT) {
        const oldest = chunks.keys().next().value;
        cacheBytes -= chunks.get(oldest).buffer.byteLength;
        chunks.delete(oldest);
      }
    } else {
      chunk = chunks.get(chunkCacheKey);
      if (chunk === undefined) throw new Error(`Missing preparation chunk ${chunkCacheKey}.`);
      chunks.delete(chunkCacheKey);
      chunks.set(chunkCacheKey, chunk);
    }
    const brick_DHW = copyBrick(chunk, sourceChunks_ZYX, chunks_ZYX, localStart_ZYX);
    self.postMessage({ id, data: brick_DHW }, [brick_DHW.buffer]);
  } catch (error) {
    self.postMessage({ id, error: error.message });
  }
};
