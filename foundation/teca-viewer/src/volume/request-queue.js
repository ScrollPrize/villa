import {
  FETCH_ATTEMPTS,
  FETCH_RETRY_DELAY_MS,
  HEDGE_AFTER_MS,
  HOST_REQUEST_CONCURRENCY,
  LATENCY_WEIGHT,
} from "../config.js";
import { mirrors } from "./s3-mirrors.js";

// S3's InternalError, a proxy's bad gateway, S3's SlowDown and a gateway timeout: answers that say to ask again.
const RETRIED_STATUSES = new Set([500, 502, 503, 504]);

export class PriorityRequestQueue {
  constructor(concurrency, groupConcurrency = concurrency, byteLimit = Infinity) {
    this.concurrency = concurrency;
    this.groupConcurrency = groupConcurrency;
    this.active = 0;
    this.activeBytes = 0;
    this.byteLimit = byteLimit;
    this.activeByGroup = new Map();
    this.sequence = 0;
    this.pending = [];
    this.completed = 0;
    this.averageLatencyMs = null;
  }

  enqueue(run, priority, signal = null, group = null, bytes = 0) {
    if (!Number.isSafeInteger(bytes) || bytes < 0 || bytes > this.byteLimit)
      throw new Error(`Request needs ${bytes} bytes, exceeding its ${this.byteLimit}-byte staging limit.`);
    const request = {
      run,
      priority,
      sequence: this.sequence++,
      signal,
      group,
      bytes,
      resolve: null,
      reject: null,
      aborted: false,
      started: false,
      abort: null,
      promise: null,
      promote: (nextPriority, nextGroup = request.group) => {
        if (request.aborted) return;
        if (!request.started) request.group = nextGroup;
        if (nextPriority < request.priority) request.priority = nextPriority;
        if (!request.started) {
          this.sortPending();
          this.drain();
        }
      },
    };
    request.promise = new Promise((resolve, reject) => {
      request.resolve = resolve;
      request.reject = reject;
    });
    request.abort = () => this.abortRequest(request, true);
    if (signal?.aborted) {
      request.abort();
      return request;
    }
    signal?.addEventListener("abort", request.abort, { once: true });
    this.pending.push(request);
    this.sortPending();
    this.drain();
    return request;
  }

  drain() {
    while (this.active < this.concurrency && this.pending.length > 0) {
      const requestIndex = this.pending.findIndex(
        (candidate) =>
          this.activeBytes + candidate.bytes <= this.byteLimit &&
          (candidate.group === null || (this.activeByGroup.get(candidate.group) ?? 0) < this.groupConcurrency),
      );
      if (requestIndex === -1) return;
      const [request] = this.pending.splice(requestIndex, 1);
      if (request.aborted || request.signal?.aborted) {
        request.reject(new DOMException("Fetch aborted.", "AbortError"));
        continue;
      }
      request.signal?.removeEventListener("abort", request.abort);
      request.started = true;
      this.active += 1;
      this.activeBytes += request.bytes;
      if (request.group !== null) {
        this.activeByGroup.set(request.group, (this.activeByGroup.get(request.group) ?? 0) + 1);
      }
      const startedAt = performance.now();
      request
        .run()
        .then(request.resolve, request.reject)
        .finally(() => {
          const latencyMs = performance.now() - startedAt;
          this.averageLatencyMs =
            this.averageLatencyMs === null
              ? latencyMs
              : this.averageLatencyMs + (latencyMs - this.averageLatencyMs) * LATENCY_WEIGHT;
          this.completed += 1;
          this.active -= 1;
          this.activeBytes -= request.bytes;
          if (request.group !== null) {
            const activeInGroup = this.activeByGroup.get(request.group) - 1;
            if (activeInGroup === 0) this.activeByGroup.delete(request.group);
            else this.activeByGroup.set(request.group, activeInGroup);
          }
          this.drain();
        });
    }
  }

  sortPending() {
    this.pending.sort((a, b) => a.priority - b.priority || a.sequence - b.sequence);
  }

  getStats() {
    return {
      active: this.active,
      pending: this.pending.length,
      concurrency: this.concurrency,
      completed: this.completed,
      averageLatencyMs: this.averageLatencyMs,
      activeBytes: this.activeBytes,
      byteLimit: Number.isFinite(this.byteLimit) ? this.byteLimit : null,
    };
  }

  cancelPending(minPriority, maxPriority = Infinity, matches = () => true) {
    const retained = [];
    const canceled = [];
    for (const request of this.pending) {
      if (request.priority >= minPriority && request.priority <= maxPriority && matches(request)) {
        canceled.push(request);
      } else {
        retained.push(request);
      }
    }
    this.pending = retained;
    for (const request of canceled) this.abortRequest(request, false);
  }

  abortRequest(request, removePending) {
    if (request.aborted) return;
    request.aborted = true;
    request.signal?.removeEventListener("abort", request.abort);
    if (removePending) {
      this.pending = this.pending.filter((item) => item !== request);
    }
    request.reject(new DOMException("Fetch aborted.", "AbortError"));
  }

  dispose() {
    this.cancelPending(0);
  }
}

const originQueues = new Map();

export function originRequestStats() {
  const stats = { active: 0, pending: 0, concurrency: 0, completed: 0 };
  for (const queue of originQueues.values()) {
    stats.active += queue.active;
    stats.pending += queue.pending.length;
    if (queue.active > 0 || queue.pending.length > 0) stats.concurrency += queue.concurrency;
    stats.completed += queue.completed;
  }
  return stats;
}

class OriginRequestQueue {
  constructor(origin) {
    this.origin = origin;
    this.owner = Symbol(origin);
    this.disposed = false;
    let queue = originQueues.get(origin);
    if (queue === undefined) {
      queue = new PriorityRequestQueue(HOST_REQUEST_CONCURRENCY, HOST_REQUEST_CONCURRENCY - 1);
      originQueues.set(origin, queue);
    }
    this.queue = queue;
  }

  enqueue(run, priority, signal = null, group = null) {
    if (this.disposed) throw new Error(`Request queue for ${this.origin} is disposed.`);
    const request = this.queue.enqueue(run, priority, signal, group);
    request.owner = this.owner;
    return request;
  }

  getStats() {
    return this.queue.getStats();
  }

  cancelPending(minPriority, maxPriority = Infinity, matches = () => true) {
    this.queue.cancelPending(minPriority, maxPriority, (request) => request.owner === this.owner && matches(request));
  }

  release() {
    this.disposed = true;
  }

  dispose() {
    if (this.disposed) return;
    this.cancelPending(0);
    this.release();
  }
}

export class MirrorRequestQueue {
  constructor(rootUrl) {
    this.queues = new Map(mirrors(rootUrl).map(({ origin }) => [origin, new OriginRequestQueue(origin)]));
    this.released = false;
  }

  enqueue(url, run, priority, signal = null, group = null) {
    return this.queues.get(new URL(url).origin).enqueue(run, priority, signal, group);
  }

  read(url, run, priority, { signal = null, group = null, spread = true, limit = null, queued = () => {} } = {}) {
    const mirrored = spread && mirrors(url).length > 1;
    return new Promise((resolve, reject) => {
      const reads = [];
      let timer = null;
      let deadline = null;
      let failures = 0;
      const finish = (settle, value) => {
        clearTimeout(timer);
        clearTimeout(deadline);
        for (const { controller } of reads) controller.abort();
        settle(value);
      };
      const read = (place) => {
        const controller = new AbortController();
        const readSignal = signal === null ? controller.signal : AbortSignal.any([signal, controller.signal]);
        const request = this.enqueue(
          place,
          () => {
            for (const other of reads) if (!other.request.started) other.controller.abort();
            if (mirrored && timer === null)
              timer = setTimeout(() => {
                if (!this.released) read(this.spare(place));
              }, HEDGE_AFTER_MS);
            if (limit !== null && deadline === null)
              deadline = setTimeout(
                () => finish(reject, new Error(`No answer for ${url} in ${limit / 1000} s.`)),
                limit,
              );
            return run(place, readSignal);
          },
          priority,
          readSignal,
          group,
        );
        reads.push({ request, controller });
        queued(request);
        request.promise.then(
          (value) => finish(resolve, value),
          (error) => {
            if (++failures === reads.length) finish(reject, error);
          },
        );
      };
      read(url);
      if (mirrored && !reads[0].request.started) read(this.spare(url));
    });
  }

  spare(url) {
    const { origin } = new URL(url);
    let best = null;
    for (const place of mirrors(url)) {
      if (place.origin === origin) continue;
      const { active, pending } = this.queues.get(place.origin).getStats();
      if (best === null || active + pending < best.load) best = { url: place.url, load: active + pending };
    }
    return best.url;
  }

  getStats() {
    const stats = { active: 0, pending: 0, concurrency: 0, completed: 0, averageLatencyMs: null };
    let measured = 0;
    for (const queue of this.queues.values()) {
      const host = queue.getStats();
      stats.active += host.active;
      stats.pending += host.pending;
      stats.concurrency += host.concurrency;
      stats.completed += host.completed;
      if (host.averageLatencyMs !== null) {
        stats.averageLatencyMs = (stats.averageLatencyMs ?? 0) + host.averageLatencyMs;
        measured += 1;
      }
    }
    if (measured > 0) stats.averageLatencyMs /= measured;
    return stats;
  }

  cancelPending(minPriority, maxPriority = Infinity, matches = () => true) {
    for (const queue of this.queues.values()) queue.cancelPending(minPriority, maxPriority, matches);
  }

  release() {
    this.released = true;
    for (const queue of this.queues.values()) queue.release();
  }

  dispose() {
    this.cancelPending(0);
    this.release();
  }
}

export async function fetchWithRetry(url, signal = null, options = {}) {
  for (let attempt = 1; ; attempt++) {
    if (signal?.aborted) throw new DOMException("Fetch aborted.", "AbortError");
    try {
      const response = await fetch(url, { ...options, signal });
      if (!RETRIED_STATUSES.has(response.status) || attempt === FETCH_ATTEMPTS) return response;
      await response.body?.cancel();
    } catch (error) {
      if (error.name === "AbortError" || attempt === FETCH_ATTEMPTS) throw error;
    }
    const delay = FETCH_RETRY_DELAY_MS * 2 ** (attempt - 1);
    await abortable(new Promise((resolve) => setTimeout(resolve, delay)), signal);
  }
}

export function abortable(promise, signal) {
  if (!signal) return promise;
  if (signal.aborted) {
    return Promise.reject(new DOMException("Fetch aborted.", "AbortError"));
  }
  return new Promise((resolve, reject) => {
    const abort = () => reject(new DOMException("Fetch aborted.", "AbortError"));
    signal.addEventListener("abort", abort, { once: true });
    promise.then(
      (value) => {
        signal.removeEventListener("abort", abort);
        resolve(value);
      },
      (error) => {
        signal.removeEventListener("abort", abort);
        reject(error);
      },
    );
  });
}
