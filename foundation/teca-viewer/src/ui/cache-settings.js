import { CHUNK_CACHE_BYTE_LIMIT, CHUNK_CACHE_MIB_RANGE } from "../config.js";

const STORAGE_KEY = "teca.viewer.chunk-cache-bytes";
const MIB = 1024 * 1024;

function validateChunkBudget(bytes) {
  const mib = bytes / MIB;
  if (!Number.isSafeInteger(mib) || mib < CHUNK_CACHE_MIB_RANGE[0] || mib > CHUNK_CACHE_MIB_RANGE[1]) {
    throw new Error(`Chunk memory must be whole MiB inside ${CHUNK_CACHE_MIB_RANGE.join("..")}, not ${mib}.`);
  }
  return bytes;
}

export function readChunkBudget(storage) {
  const stored = storage.getItem(STORAGE_KEY);
  return stored === null ? CHUNK_CACHE_BYTE_LIMIT : validateChunkBudget(Number(stored));
}

export function initChunkBudgetControls({ input, apply, status, bytes, applyBudget, storage }) {
  input.value = String(bytes / MIB);
  apply.addEventListener("click", () => {
    try {
      const next = validateChunkBudget(Number(input.value) * MIB);
      applyBudget(next);
      storage.setItem(STORAGE_KEY, String(next));
      status.textContent = `Chunk memory: ${input.value} MiB per reader.`;
    } catch (error) {
      status.textContent = error.message;
    }
  });
}

export function initStoredGeometryControls(clear, status, openSettings, geometryByteCache) {
  clear.disabled = geometryByteCache.storage === undefined || geometryByteCache.locks === undefined;
  if (clear.disabled) status.textContent = "Stored geometry is unavailable in this browser; surfaces load over HTTP.";
  openSettings.addEventListener("click", () => {
    if (geometryByteCache.error !== null)
      status.textContent = `Stored geometry paused: ${geometryByteCache.error}. Surfaces load over HTTP; Clear retries.`;
  });
  clear.addEventListener("click", async () => {
    clear.disabled = true;
    try {
      await geometryByteCache.clear();
      status.textContent = "Stored geometry cleared.";
    } catch (error) {
      status.textContent = `Could not clear stored geometry: ${error.message}`;
    } finally {
      clear.disabled = false;
    }
  });
}
