import { DTYPE_ARRAY, REGION_MAX_BYTES } from '../config.js'

function copyChunk(out_DHW, start_zyx, shape_zyx, data, index_zyx, chunk_zyx) {
  const chunkStart_zyx = index_zyx.map((i, a) => i * chunk_zyx[a])
  const [z0, y0, x0] = start_zyx.map((s, a) => Math.max(s, chunkStart_zyx[a]))
  const [z1, y1, x1] = start_zyx.map((s, a) => Math.min(s + shape_zyx[a], chunkStart_zyx[a] + chunk_zyx[a]))

  for (let z = z0; z < z1; z++) {
    for (let y = y0; y < y1; y++) {
      const from = ((z - chunkStart_zyx[0]) * chunk_zyx[1] + (y - chunkStart_zyx[1])) * chunk_zyx[2] + (x0 - chunkStart_zyx[2])
      const to = ((z - start_zyx[0]) * shape_zyx[1] + (y - start_zyx[1])) * shape_zyx[2] + (x0 - start_zyx[2])
      out_DHW.set(data.subarray(from, from + x1 - x0), to)
    }
  }
}

export async function readRegion(volume, levelIndex, start_zyx, shape_zyx, signal = null) {
  const level = volume.levels[levelIndex]
  const inside = [0, 1, 2].every(a => Number.isInteger(start_zyx[a]) && Number.isInteger(shape_zyx[a])
    && start_zyx[a] >= 0 && shape_zyx[a] >= 1 && start_zyx[a] + shape_zyx[a] <= level.shape_zyx[a])
  if (!inside) throw new Error(`region ${shape_zyx.join('×')} at ${start_zyx} is not whole voxels inside level ${levelIndex} (${level.shape_zyx.join('×')})`)
  const Typed = DTYPE_ARRAY[volume.dtype]
  const count = shape_zyx[0] * shape_zyx[1] * shape_zyx[2]
  if (count * Typed.BYTES_PER_ELEMENT > REGION_MAX_BYTES) {
    throw new Error(`region of ${shape_zyx.join('×')} voxels exceeds ${REGION_MAX_BYTES / 2 ** 20} MiB; pick a coarser level or a smaller size`)
  }

  const data_DHW = new Typed(count).fill(volume.fill)
  const { chunk_zyx } = level
  const first_zyx = start_zyx.map((s, a) => Math.floor(s / chunk_zyx[a]))
  const last_zyx = start_zyx.map((s, a) => Math.floor((s + shape_zyx[a] - 1) / chunk_zyx[a]))

  const reads = []
  for (let cz = first_zyx[0]; cz <= last_zyx[0]; cz++) {
    for (let cy = first_zyx[1]; cy <= last_zyx[1]; cy++) {
      for (let cx = first_zyx[2]; cx <= last_zyx[2]; cx++) {
        reads.push(volume.read(levelIndex, cz, cy, cx, signal).then(data => {
          if (data !== null) copyChunk(data_DHW, start_zyx, shape_zyx, data, [cz, cy, cx], chunk_zyx)
        }))
      }
    }
  }
  await Promise.all(reads)
  return data_DHW
}
