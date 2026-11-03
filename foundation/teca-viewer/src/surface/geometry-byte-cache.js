import { GEOMETRY_CACHE_BYTES } from "../config.js";

const CACHE_NAME = "teca.geometry-bytes.v1";
const LOCK_NAME = "teca.geometry-bytes";

export class GeometryByteCache {
  constructor({ storage, locks, origin }) {
    this.storage = storage;
    this.locks = locks;
    this.origin = origin;
    this.byteLimit = GEOMETRY_CACHE_BYTES;
    this.error = null;
    this.hits = 0;
    this.hitBytes = 0;
  }

  key(url, version, offset, length) {
    const key = new URL("/__teca_geometry_bytes__", this.origin);
    key.search = new URLSearchParams({ url, etag: version.etag, size: version.totalBytes, offset, length });
    return key.href;
  }

  async access(run, fallback, signal) {
    signal?.throwIfAborted();
    if (this.storage === undefined || this.locks === undefined || this.error !== null) return fallback;
    try {
      return await this.locks.request(
        LOCK_NAME,
        signal === null || signal === undefined ? {} : { signal },
        async () => {
          signal?.throwIfAborted();
          const result = await run(await this.storage.open(CACHE_NAME));
          signal?.throwIfAborted();
          return result;
        },
      );
    } catch (error) {
      if (signal?.aborted) throw signal.reason;
      this.error = error.message;
      console.warn(`Stored geometry is unavailable; using HTTP reads: ${error.message}`);
      return fallback;
    }
  }

  async get(url, version, offset, length, signal = null) {
    if (version.etag === null || !version.cacheable) return null;
    return this.access(
      async (cache) => {
        const key = this.key(url, version, offset, length);
        const response = await cache.match(key);
        if (response === undefined) return null;
        const bytes = new Uint8Array(await response.arrayBuffer());
        if (bytes.byteLength !== length) {
          await cache.delete(key);
          return null;
        }
        this.hits++;
        this.hitBytes += length;
        return bytes;
      },
      null,
      signal,
    );
  }

  async put(url, version, offset, bytes, signal = null) {
    if (version.etag === null || !version.cacheable || bytes.byteLength > this.byteLimit) return;
    await this.access(
      async (cache) => {
        const key = this.key(url, version, offset, bytes.byteLength);
        if ((await cache.match(key)) !== undefined) return;
        const keys = await cache.keys();
        let used = keys.reduce((sum, entry) => sum + Number(new URL(entry.url).searchParams.get("length")), 0);
        for (const entry of keys) {
          if (used + bytes.byteLength <= this.byteLimit) break;
          used -= Number(new URL(entry.url).searchParams.get("length"));
          await cache.delete(entry);
        }
        await cache.put(key, new Response(bytes, { headers: { "Content-Length": String(bytes.byteLength) } }));
      },
      undefined,
      signal,
    );
  }

  async clear() {
    if (this.storage === undefined || this.locks === undefined) return;
    await this.locks.request(LOCK_NAME, () => this.storage.delete(CACHE_NAME));
    this.error = null;
    this.hits = 0;
    this.hitBytes = 0;
  }
}
