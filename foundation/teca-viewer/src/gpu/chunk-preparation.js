import { createWorkerPool } from "../worker-pool.js";

export class ChunkPreparation {
  constructor() {
    this.pool = createWorkerPool(new URL("./chunk-preparation.worker.js", import.meta.url), "chunk preparation", {
      workerCount: 1,
      terminateOnAbort: false,
    });
    this.sources = new WeakMap();
    this.nextSource = 0;
  }

  copy(source_DHW, sourceChunks_ZYX, chunks_ZYX, localStart_ZYX, signal = null) {
    let key = this.sources.get(source_DHW);
    if (key === undefined) {
      key = String(this.nextSource++);
      this.sources.set(source_DHW, key);
    }
    return this.pool.run(
      { chunk: source_DHW, chunkCacheKey: key, sourceChunks_ZYX, chunks_ZYX, localStart_ZYX },
      [],
      signal,
    );
  }

  dispose() {
    this.pool.dispose();
  }
}
