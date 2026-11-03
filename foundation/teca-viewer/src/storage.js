import { STORAGE_OPEN_LIMIT_MS } from "./config.js";

export async function transaction({ name, version, upgrade }, stores, mode, use) {
  const opening = indexedDB.open(name, version);
  opening.onupgradeneeded = () => upgrade(opening.result);
  const database = await new Promise((resolve, reject) => {
    const limit = setTimeout(() => {
      opening.onsuccess = () => opening.result.close();
      reject(new Error(`${name} did not open within ${STORAGE_OPEN_LIMIT_MS} ms.`));
    }, STORAGE_OPEN_LIMIT_MS);
    opening.onsuccess = () => {
      clearTimeout(limit);
      resolve(opening.result);
    };
    opening.onerror = () => {
      clearTimeout(limit);
      reject(opening.error);
    };
  });
  try {
    const running = database.transaction(stores, mode);
    const requests = use(...stores.map((store) => running.objectStore(store))) ?? [];
    await new Promise((resolve, reject) => {
      running.oncomplete = resolve;
      running.onabort = () => reject(running.error ?? new DOMException(`${name}: transaction aborted.`, "AbortError"));
    });
    return requests.map((request) => request.result);
  } finally {
    database.close();
  }
}

export function ctName(rootUrl) {
  return rootUrl
    .replace(/\/+$/, "")
    .split("/")
    .at(-1)
    .replace(/\.zarr$/, "");
}

export function saveFile(blob, name) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = name;
  link.click();
  URL.revokeObjectURL(link.href);
}

export function lock(name) {
  const { promise: released, resolve: release } = Promise.withResolvers();
  const free = Promise.withResolvers();
  const ready = Promise.withResolvers();
  const held = { free: free.promise, ready: ready.promise, granted: false, release };
  const granted = () => {
    held.granted = true;
    ready.resolve();
    return released;
  };
  void navigator.locks.request(name, { ifAvailable: true }, (lock) => {
    free.resolve(lock !== null);
    if (lock !== null) return granted();
    void navigator.locks.request(name, granted);
    return null;
  });
  return held;
}
