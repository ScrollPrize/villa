// A job with `chunkCacheKey` posts its `chunk` only when the pool's copy of that worker's cache lacks the key. Both
// sides evict the same keys under WORKER_CACHE_BYTE_LIMIT, and a failed worker loses its cache on both.

import { WORKER_CACHE_BYTE_LIMIT, WORKER_QUEUE_LIMIT } from "./config.js";

function trimWorkerCache(cache) {
  let bytes = [...cache.values()].reduce((total, byteLength) => total + byteLength, 0);
  while (bytes > WORKER_CACHE_BYTE_LIMIT) {
    const oldestKey = cache.keys().next().value;
    bytes -= cache.get(oldestKey);
    cache.delete(oldestKey);
  }
}

export function createWorkerPool(workerUrl, label, { workerCount, terminateOnAbort }) {
  const slots = Array.from({ length: workerCount }, () => ({
    worker: null,
    active: null,
    cache: new Map(),
  }));
  const queue = [];
  let nextId = 1;
  let disposed = false;

  function spawnWorker(slot) {
    try {
      slot.worker = new Worker(workerUrl, { type: "module" });
    } catch (error) {
      console.warn(`${label} worker could not start.`, error);
      return false;
    }
    const worker = slot.worker;
    worker.onmessage = (event) => {
      if (slot.worker !== worker) return;
      if (slot.active === null || event.data.id !== slot.active.id) {
        finishSlot(slot, new Error(`${label} worker returned an unexpected job id.`), null, true);
        return;
      }
      finishSlot(slot, event.data.error ? new Error(event.data.error) : null, event.data);
    };
    worker.onerror = (event) => {
      if (slot.worker !== worker) return;
      const error = event.error ?? new Error(event.message || `${label} worker failed.`);
      console.warn(`${label} worker failed; discarding it.`, error);
      if (slot.active === null) releaseWorker(slot);
      else finishSlot(slot, error, null, true);
    };
    worker.onmessageerror = () => {
      if (slot.worker !== worker) return;
      const error = new Error(`${label} worker returned an undecodable message.`);
      if (slot.active === null) releaseWorker(slot);
      else finishSlot(slot, error, null, true);
    };
    return true;
  }

  function finishSlot(slot, error, data, replace = false) {
    const job = slot.active;
    if (job === null) return;
    slot.active = null;
    job.signal?.removeEventListener("abort", job.abort);
    if (!job.cancelled) {
      if (error) job.reject(error);
      else job.resolve(data.data);
    }
    if (replace) releaseWorker(slot);
    dispatch();
  }

  function releaseWorker(slot) {
    slot.worker?.terminate();
    slot.worker = null;
    slot.cache.clear();
  }

  function dispatch() {
    if (disposed) return;
    for (const slot of slots) {
      if (slot.active !== null || queue.length === 0) continue;
      if (slot.worker === null && !spawnWorker(slot)) {
        disposed = true;
        rejectAll(new Error(`${label} worker could not start.`));
        for (const existing of slots) releaseWorker(existing);
        return;
      }
      const job = queue.shift();
      if (job.signal?.aborted) {
        job.reject(new DOMException("Worker job aborted.", "AbortError"));
        continue;
      }
      slot.active = job;
      const message = { id: job.id, ...job.payload };
      if (job.payload.chunkCacheKey && slot.cache.has(job.payload.chunkCacheKey)) {
        const byteLength = slot.cache.get(job.payload.chunkCacheKey);
        slot.cache.delete(job.payload.chunkCacheKey);
        slot.cache.set(job.payload.chunkCacheKey, byteLength);
        delete message.chunk;
      } else if (job.payload.chunkCacheKey) {
        slot.cache.set(job.payload.chunkCacheKey, job.payload.chunk.buffer.byteLength);
        trimWorkerCache(slot.cache);
      }
      try {
        slot.worker.postMessage(message, job.transfer);
      } catch (error) {
        finishSlot(slot, error, null, true);
      }
    }
  }

  function rejectAll(error) {
    for (const job of queue.splice(0)) {
      job.signal?.removeEventListener("abort", job.abort);
      job.reject(error);
    }
    for (const slot of slots) {
      if (slot.active !== null) {
        const job = slot.active;
        slot.active = null;
        job.signal?.removeEventListener("abort", job.abort);
        job.reject(error);
      }
    }
  }

  return {
    run(payload, transfer = [], signal = null) {
      if (disposed) return Promise.reject(new Error(`${label} worker pool is disposed.`));
      if (queue.length + slots.filter((slot) => slot.active !== null).length >= WORKER_QUEUE_LIMIT) {
        return Promise.reject(new Error(`${label} worker queue exceeds the ${WORKER_QUEUE_LIMIT}-job limit.`));
      }
      if (signal?.aborted) return Promise.reject(new DOMException("Worker job aborted.", "AbortError"));
      return new Promise((resolve, reject) => {
        const job = {
          id: nextId++,
          payload,
          transfer,
          signal,
          resolve,
          reject,
          abort: null,
          cancelled: false,
        };
        job.abort = () => {
          const queuedIndex = queue.indexOf(job);
          if (queuedIndex !== -1) {
            queue.splice(queuedIndex, 1);
            reject(new DOMException("Worker job aborted.", "AbortError"));
            return;
          }
          const slot = slots.find((candidate) => candidate.active === job);
          if (slot !== undefined) {
            job.cancelled = true;
            reject(new DOMException("Worker job aborted.", "AbortError"));
            if (terminateOnAbort) {
              slot.active = null;
              releaseWorker(slot);
              dispatch();
            }
          }
        };
        signal?.addEventListener("abort", job.abort, { once: true });
        queue.push(job);
        dispatch();
      });
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      rejectAll(new DOMException("Worker pool disposed.", "AbortError"));
      for (const slot of slots) releaseWorker(slot);
    },
  };
}
