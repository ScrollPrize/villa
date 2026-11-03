export class SourceDecoder {
  constructor() {
    this.worker = new Worker(new URL("./source-decode.worker.js", import.meta.url), { type: "module" });
    this.pending = new Map();
    this.nextId = 1;
    this.failure = null;
    this.worker.onmessage = ({ data }) => {
      const pending = this.pending.get(data.id);
      if (pending === undefined) return;
      this.pending.delete(data.id);
      if (data.error !== undefined) pending.reject(new Error(data.error));
      else pending.resolve(data.result);
    };
    this.worker.onerror = (event) => {
      this.failure = new Error(event.message);
      for (const pending of this.pending.values()) pending.reject(this.failure);
      this.pending.clear();
      this.worker.terminate();
    };
  }

  run(request) {
    if (this.failure !== null) return Promise.reject(this.failure);
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      try {
        this.worker.postMessage({ ...request, id });
      } catch (error) {
        this.pending.delete(id);
        reject(error);
      }
    });
  }

  dispose() {
    this.failure = new DOMException("Source decoder closed.", "AbortError");
    this.worker.terminate();
    for (const pending of this.pending.values()) pending.reject(this.failure);
    this.pending.clear();
  }
}
