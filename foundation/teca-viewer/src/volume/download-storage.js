import { responseDownloads } from "./download-stats.js";

export class DownloadStorage {
  constructor(available) {
    this.available = available;
    this.worker = new Worker(new URL("./download-storage.worker.js", import.meta.url), { type: "module" });
    this.replies = new Map();
    this.nextId = 1;
    this.nextReader = 1;
    this.kept = null;
    this.failure = null;
    this.worker.onmessage = ({ data }) => {
      const reply = this.replies.get(data.id);
      if (reply === undefined) return;
      this.replies.delete(data.id);
      if (data.error !== undefined) {
        reply.reject(data.name === "AbortError" ? new DOMException(data.error, "AbortError") : new Error(data.error));
        return;
      }
      this.kept = data.keeps;
      reply.resolve(data.value);
    };
    this.worker.onerror = (event) => {
      this.failure = new Error(`Download storage worker: ${event.message}`);
      for (const reply of this.replies.values()) reply.reject(this.failure);
      this.replies.clear();
      this.worker.terminate();
    };
  }

  ask(message, signal = null) {
    if (this.failure !== null) return Promise.reject(this.failure);
    signal?.throwIfAborted();
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const abort = () => {
        this.replies.delete(id);
        this.worker.postMessage({ abort: id });
        reject(new DOMException("Fetch aborted.", "AbortError"));
      };
      signal?.addEventListener("abort", abort, { once: true });
      const settle = (finish) => (value) => {
        signal?.removeEventListener("abort", abort);
        finish(value);
      };
      this.replies.set(id, { resolve: settle(resolve), reject: settle(reject) });
      this.worker.postMessage({ ...message, id });
    });
  }

  openReader() {
    return this.nextReader++;
  }

  forget(reader) {
    return this.ask({ forget: reader });
  }

  async read({ reader, group, key, url, requestUrl, range, byteLimit }, signal) {
    const reply = await this.ask({ read: { reader, group, key, url, requestUrl, range, byteLimit } }, signal);
    if (reply.network) responseDownloads.download(reply.bytes.byteLength);
    return reply;
  }

  async keeps() {
    if (this.kept === null) await this.ask({ keeps: true });
    return this.kept;
  }

  status() {
    return this.ask({ status: true });
  }

  configure(enabled, byteLimit) {
    return this.ask({ configure: { enabled, byteLimit } });
  }

  clear(group = null) {
    return this.ask({ clear: { group } });
  }
}
