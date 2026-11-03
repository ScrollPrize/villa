// A 404 is undefined, which zarrita takes as a fill chunk.

import { ZARR_METADATA_BYTE_LIMIT, ZARR_OBJECT_BYTE_LIMIT, ZARR_SHARD_INDEX_BYTE_LIMIT } from "../config.js";
import { fetchWithRetry } from "./request-queue.js";
import { readResponseBytes } from "./response-bytes.js";
import { mirror } from "./s3-mirrors.js";

export const METADATA_KEY = /\/(?:zarr.json|\.zattrs|\.zarray|\.zgroup)$/;

export class DirectZarrStore {
  constructor(rootUrl, downloadStorage, downloads) {
    this.rootUrl = new URL(rootUrl);
    if (!this.rootUrl.pathname.endsWith("/")) this.rootUrl.pathname += "/";
    this.downloadStorage = downloadStorage;
    this.reader = downloadStorage.openReader();
    this.metadata = new Map();
    this.dataLimits = new Map();
    this.downloads = downloads;
  }

  setDataLimit(prefix, byteLimit) {
    if (
      typeof prefix !== "string" ||
      !prefix.startsWith("/") ||
      !prefix.endsWith("/") ||
      !Number.isSafeInteger(byteLimit) ||
      byteLimit < 1 ||
      byteLimit > ZARR_OBJECT_BYTE_LIMIT
    ) {
      throw new Error("Zarr data limit needs an absolute key prefix and a bounded positive byte count.");
    }
    this.dataLimits.set(prefix, byteLimit);
  }

  keepsMirrorReads() {
    return this.downloadStorage.keeps();
  }

  async get(key, options = {}) {
    const cached = this.metadata.get(key);
    if (cached !== undefined) return cached;
    if (!METADATA_KEY.test(key)) return this.readData(key, null, options.byteLimit ?? this.dataLimit(key), options);
    const url = this.resolve(key);
    const response = await fetchWithRetry(options.url ?? url, options.signal, { cache: "no-cache" });
    if (response.status === 404) {
      await response.body?.cancel();
      return undefined;
    }
    if (response.status !== 200) {
      await response.body?.cancel();
      throw new Error(`Failed to fetch Zarr key ${key}: ${response.status}.`);
    }
    const bytes = await readResponseBytes(
      response,
      options.byteLimit ?? ZARR_METADATA_BYTE_LIMIT,
      options.signal ?? null,
    );
    if (key.endsWith("/zarr.json") || key === "/zarr.json") this.metadata.set(key, bytes);
    return bytes;
  }

  async getRange(key, range, options = {}) {
    if ("suffixLength" in range) {
      if (
        !Number.isSafeInteger(range.suffixLength) ||
        range.suffixLength < 1 ||
        range.suffixLength > ZARR_SHARD_INDEX_BYTE_LIMIT
      ) {
        throw new Error("Zarr shard index exceeds the direct decoder byte limit.");
      }
      const suffix = { header: `bytes=-${range.suffixLength}`, offset: null, length: range.suffixLength };
      return this.readData(key, suffix, suffix.length, options);
    }
    const end = range.offset + range.length - 1;
    if (
      !Number.isSafeInteger(range.offset) ||
      range.offset < 0 ||
      !Number.isSafeInteger(range.length) ||
      range.length < 1 ||
      range.length > this.dataLimit(key) ||
      !Number.isSafeInteger(end)
    ) {
      throw new Error("Zarr byte range must be positive and fit the direct object byte limit.");
    }
    const part = { header: `bytes=${range.offset}-${end}`, offset: range.offset, length: range.length };
    return this.readData(key, part, part.length, options);
  }

  async readData(key, range, byteLimit, options) {
    const url = this.resolve(key);
    const { status, bytes } = await this.downloadStorage.read(
      { reader: this.reader, group: this.rootUrl.href, key, url, requestUrl: options.url ?? url, range, byteLimit },
      options.signal ?? null,
    );
    if (status === 404) {
      this.downloads.download(0);
      return undefined;
    }
    if (bytes === null) {
      throw new Error(
        range === null
          ? `Failed to fetch Zarr key ${key}: ${status}.`
          : `Zarr range request for ${key} returned ${status}, expected 206.`,
      );
    }
    this.downloads.download(bytes.byteLength);
    return bytes;
  }

  dispose() {
    void this.downloadStorage.forget(this.reader);
  }

  resolve(key) {
    if (
      typeof key !== "string" ||
      !key.startsWith("/") ||
      key.split("/").some((part, index) => index > 0 && (part === "" || part === "." || part === ".."))
    ) {
      throw new Error(`Invalid Zarr store key: ${key}.`);
    }
    const resolved = new URL(this.rootUrl);
    resolved.pathname += key
      .slice(1)
      .split("/")
      .map((part) => encodeURIComponent(part))
      .join("/");
    return METADATA_KEY.test(key) ? resolved.href : mirror(resolved.href).url;
  }

  dataLimit(key) {
    let result = null;
    for (const [prefix, byteLimit] of this.dataLimits) {
      if (key.startsWith(prefix) && (result === null || prefix.length > result.prefix.length)) {
        result = { prefix, byteLimit };
      }
    }
    return result?.byteLimit ?? ZARR_OBJECT_BYTE_LIMIT;
  }
}
