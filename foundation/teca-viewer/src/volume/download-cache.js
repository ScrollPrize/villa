import {
  DOWNLOAD_CACHE_BYTE_LIMIT,
  DOWNLOAD_CACHE_GIB_RANGE,
  HEURISTIC_FRESHNESS_SHARE,
  STORAGE_OPEN_LIMIT_MS,
} from "../config.js";

const GIB = 1024 ** 3;
const DATABASE = "teca.downloads.v1";
const DEFAULT_SETTINGS = {
  id: "settings",
  enabled: true,
  byteLimit: DOWNLOAD_CACHE_BYTE_LIMIT,
  used: 0,
  epoch: 0,
  clock: 0,
};
const NO_STORE = /(?:^|,)\s*no-store(?:\s|,|$)/i;
const NO_CACHE = /(?:^|,)\s*no-cache(?:\s|,|=|$)/i;
const MAX_AGE = /(?:^|,)\s*max-age\s*=\s*(\d+)/i;

function result(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function removeEntry(entries, groups, settings, entry) {
  entries.delete(entry.key);
  settings.used -= entry.bytes;
  const source = await result(groups.get(entry.group));
  source.bytes -= entry.bytes;
  source.count--;
  if (source.count === 0) groups.delete(source.id);
  else groups.put(source);
}

async function evict(entries, groups, settings, limit) {
  while (settings.used > limit) {
    const cursor = await result(entries.index("used").openCursor());
    if (cursor === null) throw new Error("Download storage byte accounting is inconsistent.");
    await removeEntry(entries, groups, settings, cursor.value);
  }
}

export class DownloadCache {
  constructor(factory) {
    this.factory = factory;
    this.database = null;
    this.error = null;
    this.hitBytes = 0;
    this.enabled = false;
    this.settingsRead = null;
  }

  async open() {
    if (this.database === null) {
      const request = this.factory.open(DATABASE, 1);
      request.onupgradeneeded = () => {
        const database = request.result;
        const entries = database.createObjectStore("entries", { keyPath: "key" });
        entries.createIndex("used", "used");
        entries.createIndex("group", "group");
        database.createObjectStore("groups", { keyPath: "id" });
        database.createObjectStore("control", { keyPath: "id" }).put({ ...DEFAULT_SETTINGS });
      };
      const limit = new Promise((_, reject) =>
        setTimeout(() => {
          request.onsuccess = () => request.result.close();
          reject(new Error(`the database did not open within ${STORAGE_OPEN_LIMIT_MS} ms`));
        }, STORAGE_OPEN_LIMIT_MS),
      );
      this.database = Promise.race([result(request), limit]).catch((error) => {
        this.database = null;
        throw error;
      });
    }
    return this.database;
  }

  async transaction(mode, run) {
    const database = await this.open();
    const transaction = database.transaction(["entries", "groups", "control"], mode);
    let finished = false;
    const done = new Promise((resolve, reject) => {
      transaction.oncomplete = () => {
        finished = true;
        resolve();
      };
      transaction.onabort = () => {
        finished = true;
        reject(transaction.error ?? new Error("Download storage transaction aborted."));
      };
    });
    const entries = transaction.objectStore("entries");
    const groups = transaction.objectStore("groups");
    const control = transaction.objectStore("control");
    try {
      const settings = await result(control.get("settings"));
      const value = await run(entries, groups, control, settings);
      await done;
      this.enabled = settings.enabled;
      return value;
    } catch (error) {
      if (!finished) transaction.abort();
      await done.catch(() => {});
      throw error;
    }
  }

  async optional(run, fallback) {
    if (this.factory === undefined || this.error !== null) return fallback;
    try {
      return await run();
    } catch (error) {
      this.error = error.message;
      console.warn(`Download storage unavailable; continuing over HTTP: ${error.message}`);
      return fallback;
    }
  }

  async keeps() {
    this.settingsRead ??= this.optional(() => this.transaction("readonly", async () => {}), undefined);
    await this.settingsRead;
    return this.factory !== undefined && this.error === null && this.enabled;
  }

  async lookup(group, url, range) {
    return this.optional(
      () =>
        this.transaction("readonly", async (entries, _groups, _control, settings) => {
          if (!settings.enabled) return null;
          const key = JSON.stringify([group, url, range]);
          const entry = await result(entries.get(key));
          return { key, epoch: settings.epoch, entry };
        }),
      null,
    );
  }

  async put(lease, group, response, bytes) {
    if (lease === null) return;
    const etag = storableEtag(response.headers);
    if (etag === null) return;
    await this.optional(
      () =>
        this.transaction("readwrite", async (entries, groups, control, settings) => {
          if (!settings.enabled || settings.epoch !== lease.epoch || bytes.byteLength > settings.byteLimit) return;
          const previous = await result(entries.get(lease.key));
          if (previous !== undefined) await removeEntry(entries, groups, settings, previous);
          await evict(entries, groups, settings, settings.byteLimit - bytes.byteLength);
          const source = (await result(groups.get(group))) ?? { id: group, bytes: 0, count: 0 };
          source.bytes += bytes.byteLength;
          source.count++;
          groups.put(source);
          entries.put({
            key: lease.key,
            group,
            etag,
            status: response.status,
            headers: [...response.headers],
            body: new Blob([bytes]),
            bytes: bytes.byteLength,
            used: ++settings.clock,
            freshUntil: freshUntil(response.headers, Date.now()),
          });
          settings.used += bytes.byteLength;
          control.put(settings);
        }),
      undefined,
    );
  }

  async refresh(lease, until) {
    await this.optional(
      () =>
        this.transaction("readwrite", async (entries, _groups, _control, settings) => {
          if (!settings.enabled || settings.epoch !== lease.epoch) return;
          const entry = await result(entries.get(lease.key));
          if (entry?.etag === lease.entry.etag) entries.put({ ...entry, freshUntil: until });
        }),
      undefined,
    );
  }

  async configure(enabled, byteLimit) {
    if (
      typeof enabled !== "boolean" ||
      !Number.isSafeInteger(byteLimit) ||
      byteLimit < 1 ||
      byteLimit > DOWNLOAD_CACHE_GIB_RANGE[1] * GIB
    )
      throw new Error("Choose a disk budget between 1 byte and 1024 GiB.");
    await this.transaction("readwrite", async (entries, groups, control, settings) => {
      settings.enabled = enabled;
      settings.byteLimit = byteLimit;
      settings.epoch++;
      await evict(entries, groups, settings, byteLimit);
      control.put(settings);
    });
    this.error = null;
  }

  async clear(group = null) {
    await this.transaction("readwrite", async (entries, groups, control, settings) => {
      settings.epoch++;
      if (group === null) {
        entries.clear();
        groups.clear();
        settings.used = 0;
      } else {
        await new Promise((resolve, reject) => {
          const request = entries.index("group").openCursor(group);
          request.onerror = () => reject(request.error);
          request.onsuccess = () => {
            const cursor = request.result;
            if (cursor === null) return resolve();
            settings.used -= cursor.value.bytes;
            cursor.delete();
            cursor.continue();
          };
        });
        groups.delete(group);
      }
      control.put(settings);
    });
    this.error = null;
  }

  async status() {
    return this.transaction("readonly", async (_entries, groups, _control, settings) => ({
      ...settings,
      groups: await result(groups.getAll()),
      hitBytes: this.hitBytes,
      error: this.error,
    }));
  }
}

export function strongEtag(headers) {
  const etag = headers.get("ETag");
  return etag !== null && /^"[^"\r\n]*"$/.test(etag) ? etag : null;
}

export function storableEtag(headers) {
  const etag = strongEtag(headers);
  return etag !== null && !NO_STORE.test(headers.get("Cache-Control") ?? "") ? etag : null;
}

export function freshUntil(headers, at) {
  const control = headers.get("Cache-Control") ?? "";
  if (NO_CACHE.test(control)) return at;
  const maxAge = MAX_AGE.exec(control);
  if (maxAge !== null) return at + 1000 * Number(maxAge[1]);
  const expires = headers.get("Expires");
  if (expires !== null) {
    const time = Date.parse(expires);
    return time > at ? time : at;
  }
  const modified = Date.parse(headers.get("Last-Modified") ?? "");
  return modified <= at ? at + HEURISTIC_FRESHNESS_SHARE * (at - modified) : at;
}
