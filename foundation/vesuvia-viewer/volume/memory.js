import { DTYPE_ARRAY } from '../config.js'

// levels are registered to CT level-0 voxels like any layer's. The owner writes every level it declares; nothing builds a pyramid for it.
export function createMemoryVolume({ name, dtype, fill, labels, levels }) {
  const Typed = DTYPE_ARRAY[dtype]
  const url = `memory:${name}`
  const chunks = new Map()
  let arrived = new Map()
  const key = (level, cz, cy, cx) => `${level}/${cz}/${cy}/${cx}`
  const cached = (level, cz, cy, cx) => chunks.get(key(level, cz, cy, cx)) ?? null

  return {
    url, meta: { url, dtype, fill, labels, levels }, levels, dtype, fill, key, cached,
    request() {},
    clearQueue() {},
    take() {
      const taken = [...arrived.values()]
      arrived = new Map()
      return taken
    },
    read: (level, cz, cy, cx, signal = null) => signal?.aborted ? Promise.reject(signal.reason) : Promise.resolve(cached(level, cz, cy, cx)),
    stats: () => ({ inFlight: 0, failed: 0 }),
    close() {},

    write(levelIndex, start_zyx, shape_zyx, data_DHW, mask_DHW = null) {
      const { shape_zyx: level_zyx, chunk_zyx } = levels[levelIndex]
      const inside = [0, 1, 2].every(a => Number.isInteger(start_zyx[a]) && Number.isInteger(shape_zyx[a])
        && start_zyx[a] >= 0 && shape_zyx[a] >= 1 && start_zyx[a] + shape_zyx[a] <= level_zyx[a])
      if (!inside) throw new Error(`write of ${shape_zyx.join('×')} at ${start_zyx} is not whole voxels inside level ${levelIndex} (${level_zyx.join('×')})`)
      const count = shape_zyx[0] * shape_zyx[1] * shape_zyx[2]
      if (data_DHW.length !== count || (mask_DHW !== null && mask_DHW.length !== count)) {
        throw new Error(`write of ${shape_zyx.join('×')} voxels got ${data_DHW.length}${mask_DHW === null ? '' : ` with a mask of ${mask_DHW.length}`}`)
      }

      const first_zyx = start_zyx.map((s, a) => Math.floor(s / chunk_zyx[a]))
      const last_zyx = start_zyx.map((s, a) => Math.floor((s + shape_zyx[a] - 1) / chunk_zyx[a]))
      for (let cz = first_zyx[0]; cz <= last_zyx[0]; cz++) {
        for (let cy = first_zyx[1]; cy <= last_zyx[1]; cy++) {
          for (let cx = first_zyx[2]; cx <= last_zyx[2]; cx++) {
            const k = key(levelIndex, cz, cy, cx)
            const data = chunks.get(k) ?? new Typed(chunk_zyx[0] * chunk_zyx[1] * chunk_zyx[2]).fill(fill)
            // Plain numbers and not arrays mapped: a brush stamp is thousands of writes of a few hundred voxels, and the arrays cost a fifth of one.
            const chunkStart_zyx = [cz * chunk_zyx[0], cy * chunk_zyx[1], cx * chunk_zyx[2]]
            const z0 = Math.max(start_zyx[0], chunkStart_zyx[0]), z1 = Math.min(start_zyx[0] + shape_zyx[0], chunkStart_zyx[0] + chunk_zyx[0])
            const y0 = Math.max(start_zyx[1], chunkStart_zyx[1]), y1 = Math.min(start_zyx[1] + shape_zyx[1], chunkStart_zyx[1] + chunk_zyx[1])
            const x0 = Math.max(start_zyx[2], chunkStart_zyx[2]), x1 = Math.min(start_zyx[2] + shape_zyx[2], chunkStart_zyx[2] + chunk_zyx[2])
            let [lo, hi] = mask_DHW === null ? [z0, z1] : [z1, z0]
            for (let z = z0; z < z1; z++) {
              for (let y = y0; y < y1; y++) {
                const from = ((z - start_zyx[0]) * shape_zyx[1] + (y - start_zyx[1])) * shape_zyx[2] + (x0 - start_zyx[2])
                const to = ((z - chunkStart_zyx[0]) * chunk_zyx[1] + (y - chunkStart_zyx[1])) * chunk_zyx[2] + (x0 - chunkStart_zyx[2])
                if (mask_DHW === null) {
                  data.set(data_DHW.subarray(from, from + x1 - x0), to)
                  continue
                }
                for (let i = 0; i < x1 - x0; i++) {
                  if (mask_DHW[from + i] === 0) continue
                  data[to + i] = data_DHW[from + i]
                  if (z < lo) lo = z
                  hi = z + 1
                }
              }
            }
            if (lo >= hi) continue
            chunks.set(k, data)
            const waiting = arrived.get(k)
            if (waiting === undefined) arrived.set(k, { level: levelIndex, cz, cy, cx, data, changed: [lo - chunkStart_zyx[0], hi - chunkStart_zyx[0]] })
            else {
              waiting.changed[0] = Math.min(waiting.changed[0], lo - chunkStart_zyx[0])
              waiting.changed[1] = Math.max(waiting.changed[1], hi - chunkStart_zyx[0])
            }
          }
        }
      }
    },
  }
}
