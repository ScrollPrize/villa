import { CachedDownload } from "./cached-download.js";
import { DownloadCache, strongEtag } from "./download-cache.js";
import { readResponseBytes } from "./response-bytes.js";

const cache = new DownloadCache(self.indexedDB);
const readers = new Map();
const reads = new Map();

self.onmessage = async ({ data }) => {
  if (data.abort !== undefined) {
    reads.get(data.abort)?.abort();
    return;
  }
  const { id } = data;
  try {
    if (data.read === undefined) {
      let value;
      if (data.status) value = await cache.status();
      else if (data.configure !== undefined) await cache.configure(data.configure.enabled, data.configure.byteLimit);
      else if (data.clear !== undefined) await cache.clear(data.clear.group);
      else if (data.forget !== undefined) readers.delete(data.forget);
      self.postMessage({ id, value, keeps: await cache.keeps() });
      return;
    }
    const { reader, group, key, url, requestUrl, range, byteLimit } = data.read;
    const controller = new AbortController();
    reads.set(id, controller);
    const signal = controller.signal;
    let download = readers.get(reader);
    if (download === undefined) {
      download = new CachedDownload(group, cache);
      readers.set(reader, download);
    }
    const request =
      range === null ? { signal, cache: "default" } : { signal, headers: { Range: range.header }, cache: "no-store" };
    const { response, save } = await download.fetch(url, request, requestUrl);
    if (response.status !== (range === null ? 200 : 206)) {
      await response.body?.cancel();
      self.postMessage({ id, value: { status: response.status, bytes: null }, keeps: await cache.keeps() });
      return;
    }
    if (range !== null) {
      // S3 CORS rules may not expose Content-Range; the reader's version memo then pins the strong ETag.
      const contentRangeHeader = response.headers.get("Content-Range");
      if (contentRangeHeader === null) {
        if (strongEtag(response.headers) === null) {
          await response.body?.cancel();
          throw new Error(`Zarr range request for ${key} exposed neither Content-Range nor a strong ETag.`);
        }
      } else {
        const contentRange = /^bytes (\d+)-(\d+)\/(\d+)$/.exec(contentRangeHeader);
        if (contentRange === null) {
          await response.body?.cancel();
          throw new Error(`Zarr range request for ${key} returned an invalid Content-Range.`);
        }
        const [start, end, total] = contentRange.slice(1).map(Number);
        if (
          !Number.isSafeInteger(start) ||
          !Number.isSafeInteger(end) ||
          !Number.isSafeInteger(total) ||
          total < 1 ||
          start < 0 ||
          start > end ||
          end >= total ||
          end - start + 1 !== range.length ||
          (range.offset === null
            ? end !== total - 1 || start !== total - range.length
            : start !== range.offset || end !== range.offset + range.length - 1)
        ) {
          await response.body?.cancel();
          throw new Error(`Zarr range request for ${key} returned Content-Range outside the requested bytes.`);
        }
      }
    }
    const bytes = await readResponseBytes(response, byteLimit, signal);
    if (range !== null && bytes.byteLength !== range.length) {
      throw new Error(`Zarr range request for ${key} returned ${bytes.byteLength} bytes, expected ${range.length}.`);
    }
    await save(bytes);
    const network = /^https?:/.test(response.url);
    self.postMessage({ id, value: { status: response.status, bytes, network }, keeps: await cache.keeps() }, [
      bytes.buffer,
    ]);
  } catch (error) {
    self.postMessage({ id, error: error.message, name: error.name });
  } finally {
    reads.delete(id);
  }
};
