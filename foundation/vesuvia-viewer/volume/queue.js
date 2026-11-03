import { RETRY_MS } from '../config.js'
import { mirror } from './mirrors.js'

export class FetchQueue {
  constructor(workers, queues, { concurrency, onArrive, onFail, onError }) {
    this.queues = queues
    queues.push(this)
    this.closed = false
    this.scheduled = false
    this.workers = workers
    this.concurrency = concurrency
    this.onArrive = onArrive
    this.onFail = onFail
    this.onError = onError
    this.reported = false
    this.next = 0
    this.queued = []
    this.queuedByKey = new Map()
    this.inFlight = new Map()
    this.failed = new Map()
    for (const worker of workers) {
      worker.onmessage = ({ data }) => this.settle(data)
      worker.onerror = event => {
        if (!this.reported) this.onError(`worker: ${event.message}`)
        this.reported = true
        this.close()
      }
    }
  }

  request(key, chunk, priority, message, keep = false) {
    if (this.closed || this.inFlight.has(key)) return true
    const failedAt = this.failed.get(key)
    if (failedAt !== undefined) {
      if (Date.now() - failedAt < RETRY_MS) return false
      this.failed.delete(key)
    }
    const queued = this.queuedByKey.get(key)
    if (queued === undefined) {
      const entry = { key, chunk, priority, message: { ...message, ...mirror(message.url, message.range) }, keep }
      this.queued.push(entry)
      this.queuedByKey.set(key, entry)
    } else {
      queued.priority = Math.min(queued.priority, priority)
      queued.keep ||= keep
    }
    if (!this.scheduled) {
      this.scheduled = true
      queueMicrotask(() => {
        this.scheduled = false
        this.queued.sort((a, b) => a.priority - b.priority)
        this.dispatch()
      })
    }
    return true
  }

  clearQueue() {
    this.queued = this.queued.filter(entry => entry.keep)
    this.queuedByKey = new Map(this.queued.map(entry => [entry.key, entry]))
  }

  unkeep(key) {
    const queued = this.queuedByKey.get(key)
    if (queued !== undefined) queued.keep = false
  }

  dispatch() {
    for (;;) {
      const counts = new Map()
      for (const queue of this.queues) {
        for (const { message } of queue.inFlight.values()) counts.set(message.origin, (counts.get(message.origin) ?? 0) + 1)
      }
      let next = null
      let index = -1
      for (const queue of this.queues) {
        const free = queue.queued.findIndex(entry => (counts.get(entry.message.origin) ?? 0) < queue.concurrency)
        if (free >= 0 && (next === null || queue.queued[free].priority < next.queued[index].priority)) {
          next = queue
          index = free
        }
      }
      if (next === null) break
      const [{ key, chunk, message }] = next.queued.splice(index, 1)
      next.queuedByKey.delete(key)
      next.post(key, chunk, message, 1)
    }
  }

  post(key, chunk, message, attempts) {
    this.inFlight.set(key, { chunk, message, attempts })
    this.workers[this.next].postMessage({ id: key, ...message })
    this.next = (this.next + 1) % this.workers.length
  }

  settle({ id: key, status, data, message }) {
    const { chunk, message: request, attempts } = this.inFlight.get(key)
    this.inFlight.delete(key)
    if (status === 'ok') {
      this.onArrive(key, chunk, data)
    } else if (status === 'missing') {
      this.onArrive(key, chunk, null)
    } else if (attempts < 2) {
      this.post(key, chunk, request, attempts + 1)
    } else {
      this.failed.set(key, Date.now())
      this.onFail(key, message)
      if (!this.reported) this.onError(message)
      this.reported = true
    }
    this.dispatch()
  }

  stats() {
    return { inFlight: this.inFlight.size, failed: this.failed.size }
  }

  close() {
    this.closed = true
    this.queued = []
    this.queuedByKey.clear()
    this.inFlight.clear()
    for (const worker of this.workers) { worker.onmessage = null; worker.terminate() }
    const index = this.queues.indexOf(this)
    if (index >= 0) this.queues.splice(index, 1)
    this.dispatch()
  }
}
