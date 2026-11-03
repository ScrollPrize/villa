import { ZARR_METADATA_TIME_LIMIT_MS } from "../config.js";
import { mirrors } from "./s3-mirrors.js";
import { METADATA_KEY } from "./zarr-store.js";

export class QueuedZarrStore {
  constructor(store, queue) {
    this.store = store;
    this.queue = queue;
    this.metadata = store.metadata;
    this.requests = new WeakMap();
  }

  get(key, options = {}) {
    if (METADATA_KEY.test(key) && !this.metadata.has(key)) {
      const read = (url, signal) => this.store.get(key, { ...options, url, signal });
      return this.queue.read(this.store.resolve(key), read, 0, {
        signal: options.signal,
        limit: ZARR_METADATA_TIME_LIMIT_MS,
      });
    }
    return this.read(key, options, (read) => this.store.get(key, read));
  }

  getRange(key, range, options = {}) {
    return this.read(key, options, (read) => this.store.getRange(key, range, read));
  }

  async read(key, options, run) {
    const signal = options.signal;
    const owner = signal === undefined || signal === null ? undefined : this.requests.get(signal);
    if (owner === undefined) return run(options);
    const home = this.store.resolve(key);
    const spread = mirrors(home).length > 1 && (await this.store.keepsMirrorReads());
    const reads = [];
    try {
      return await this.queue.read(
        home,
        (url, readSignal) => {
          this.requests.set(readSignal, owner);
          return run({ ...options, url, signal: readSignal });
        },
        owner.priority,
        {
          signal,
          group: owner.group,
          spread,
          queued: (request) => {
            request.chunkRequest = owner;
            owner.reads.add(request);
            reads.push(request);
          },
        },
      );
    } finally {
      for (const request of reads) owner.reads.delete(request);
    }
  }

  resolve(key) {
    return this.store.resolve(key);
  }

  setDataLimit(prefix, byteLimit) {
    this.store.setDataLimit(prefix, byteLimit);
  }
}
