import { REGION_PRIORITY } from '../config.js'
import { ChunkCache } from './cache.js'
import { FetchQueue } from './queue.js'

export function openVolume(meta, { cacheBytes, concurrency, workers, onError, queues }) {
  const { url, dtype, fill, levels } = meta
  const cache = new ChunkCache(cacheBytes)
  const pool = Array.from({ length: workers }, () => new Worker(new URL('./worker.js', import.meta.url), { type: 'module' }))
  const counts = levels.map(({ chunk_zyx: [cz, cy, cx] }) => cz * cy * cx)
  const key = (level, cz, cy, cx) => `${level}/${cz}/${cy}/${cx}`

  let arrived = []
  const waiting = new Map()

  const arrive = (k, chunk, data) => {
    cache.set(k, data)
    arrived.push({ ...chunk, data })
    for (const { resolve } of waiting.get(k) ?? []) resolve(data)
    waiting.delete(k)
  }
  const queue = new FetchQueue(pool, queues, {
    concurrency,
    onError,
    onArrive: arrive,
    onFail(k, message) {
      for (const { reject } of waiting.get(k) ?? []) reject(new Error(message))
      waiting.delete(k)
    },
  })

  function request(level, cz, cy, cx, priority, keep) {
    const { chunkUrl, chunkRange, heldChunk, codec, littleEndian } = levels[level]
    if (heldChunk !== undefined) {
      arrive(key(level, cz, cy, cx), { level, cz, cy, cx }, heldChunk(cz, cy, cx))
      return true
    }
    const message = { url: chunkUrl(cz, cy, cx), range: chunkRange?.(cz, cy, cx) ?? null, codec, dtype, littleEndian, count: counts[level] }
    return queue.request(key(level, cz, cy, cx), { level, cz, cy, cx }, priority, message, keep)
  }

  return {
    url, meta, levels, dtype, fill, key,

    request(level, cz, cy, cx, priority) {
      if (!cache.has(key(level, cz, cy, cx))) request(level, cz, cy, cx, priority, false)
    },
    clearQueue: () => queue.clearQueue(),
    cached: (level, cz, cy, cx) => cache.get(key(level, cz, cy, cx)),
    take() {
      const chunks = arrived
      arrived = []
      return chunks
    },

    read(level, cz, cy, cx, signal = null) {
      const k = key(level, cz, cy, cx)
      const data = cache.get(k)
      if (data !== undefined) return Promise.resolve(data)
      if (levels[level].heldChunk !== undefined) return Promise.resolve(levels[level].heldChunk(cz, cy, cx))
      if (signal?.aborted) return Promise.reject(signal.reason)
      if (queue.closed) return Promise.reject(new Error(`${url} is closed`))
      if (!request(level, cz, cy, cx, REGION_PRIORITY, true)) return Promise.reject(new Error(`chunk ${k} failed; retry later`))

      return new Promise((resolve, reject) => {
        const waiter = { resolve, reject }
        waiting.set(k, [...(waiting.get(k) ?? []), waiter])
        signal?.addEventListener('abort', () => {
          const others = (waiting.get(k) ?? []).filter(other => other !== waiter)
          if (others.length > 0) {
            waiting.set(k, others)
          } else {
            waiting.delete(k)
            queue.unkeep(k)
          }
          reject(signal.reason)
        }, { once: true })
      })
    },

    stats: () => queue.stats(),
    close() {
      queue.close()
      for (const waiters of waiting.values()) for (const { reject } of waiters) reject(new Error(`${url} closed`))
      waiting.clear()
    },
  }
}
