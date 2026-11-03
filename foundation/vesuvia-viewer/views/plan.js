import { ZOOM_TOLERANCE, FIT_FRACTION } from '../config.js'

const BAND_WEIGHT = 1000
const LEVEL_WEIGHT = 100
const DISTANCE_CAP = 99
const PIN_PRIORITY = 2 * BAND_WEIGHT

// Band 0 is visible, bands above it are prefetch, and band -1 is where the view is heading: its coarsest chunks that nothing else wants come back in `ahead`, fetched and held in no slot.
export function planChunks({ levels, views, minDownsample, slotCapacity, windowChunks, pinCoarsestChunks }) {
  const coarsest = levels.length - 1
  const grid_zyx = levels[coarsest].shape_zyx.map((s, a) => Math.ceil(s / levels[coarsest].chunk_zyx[a]))
  const pin = grid_zyx.every(n => n <= windowChunks) && grid_zyx[0] * grid_zyx[1] * grid_zyx[2] <= pinCoarsestChunks
  // A chunk's key in `wanted` is its index in a grid that holds any level's: a number, which costs a third of a string built per chunk per frame.
  const span_zyx = [0, 1, 2].map(a => Math.max(...levels.map(level => Math.ceil(level.shape_zyx[a] / level.chunk_zyx[a]))))

  const displayLevels = views.map(view => {
    const limit = Math.max(view.zoom * ZOOM_TOLERANCE, minDownsample)
    for (let level = coarsest; level >= 0; level--) {
      if (Math.max(...levels[level].downsample_zyx) <= limit) return level
    }
    return 0
  })

  const capacity = Math.floor(FIT_FRACTION * slotCapacity)
  let prefetch = views.some(view => view.boxes.some(box => box.band > 0))
  let intermediate = levels.length > 2
  let wanted, heading
  for (;;) {
    wanted = new Map()
    heading = new Map()
    const displayCounts = views.map(() => new Set())
    views.forEach((view, v) => {
      for (const box of view.boxes) {
        if (box.band > 0 && !prefetch) continue
        for (let level = displayLevels[v]; level <= coarsest; level++) {
          if (!intermediate && level !== displayLevels[v] && level !== coarsest) continue
          if (box.band < 0 && level !== coarsest) continue
          const { shape_zyx, chunk_zyx, downsample_zyx, offset_zyx } = levels[level]
          const lo_zyx = [0, 0, 0]
          const hi_zyx = [0, 0, 0]
          const centre_zyx = [0, 0, 0]
          let empty = false
          for (let a = 0; a < 3; a++) {
            const d = downsample_zyx[a]
            const half = (d - 1) / 2
            const lo = Math.max(0, Math.floor((box.min_zyx[a] - offset_zyx[a] - half) / d))
            const hi = Math.min(shape_zyx[a] - 1, Math.floor((box.max_zyx[a] - offset_zyx[a] - half) / d) + 1)
            if (lo > hi) empty = true
            lo_zyx[a] = Math.floor(lo / chunk_zyx[a])
            hi_zyx[a] = Math.floor(hi / chunk_zyx[a])
            centre_zyx[a] = ((box.min_zyx[a] + box.max_zyx[a]) / 2 - offset_zyx[a] - half) / d / chunk_zyx[a]
          }
          if (empty) continue
          for (let cz = lo_zyx[0]; cz <= hi_zyx[0]; cz++) {
            for (let cy = lo_zyx[1]; cy <= hi_zyx[1]; cy++) {
              for (let cx = lo_zyx[2]; cx <= hi_zyx[2]; cx++) {
                const distance = Math.hypot(cz + 0.5 - centre_zyx[0], cy + 0.5 - centre_zyx[1], cx + 0.5 - centre_zyx[2])
                const priority = Math.max(0, box.band) * BAND_WEIGHT + (coarsest - level) * LEVEL_WEIGHT + Math.min(DISTANCE_CAP, distance)
                const key = ((level * span_zyx[0] + cz) * span_zyx[1] + cy) * span_zyx[2] + cx
                if (box.band < 0) {
                  if (!heading.has(key)) heading.set(key, { level, cz, cy, cx, priority })
                  continue
                }
                if (level === displayLevels[v]) displayCounts[v].add(key)
                const known = wanted.get(key)
                if (known === undefined) wanted.set(key, { level, cz, cy, cx, priority, views: [v] })
                else {
                  known.priority = Math.min(known.priority, priority)
                  if (!known.views.includes(v)) known.views.push(v)
                }
              }
            }
          }
        }
      }
    })
    if (pin) {
      for (let cz = 0; cz < grid_zyx[0]; cz++) for (let cy = 0; cy < grid_zyx[1]; cy++) for (let cx = 0; cx < grid_zyx[2]; cx++) {
        const key = ((coarsest * span_zyx[0] + cz) * span_zyx[1] + cy) * span_zyx[2] + cx
        if (!wanted.has(key)) wanted.set(key, { level: coarsest, cz, cy, cx, priority: PIN_PRIORITY, views: [] })
      }
    }
    if (wanted.size <= capacity) break
    if (prefetch) {
      prefetch = false
      continue
    }
    if (intermediate) {
      intermediate = false
      continue
    }
    let victim = -1
    views.forEach((view, v) => {
      if (displayLevels[v] < coarsest && (victim < 0 || displayCounts[v].size > displayCounts[victim].size)) victim = v
    })
    if (victim < 0) break
    displayLevels[victim] += 1
  }

  const sorted = [...wanted.values()].sort((a, b) => a.priority - b.priority)
  if (sorted.length > capacity) sorted.length = capacity

  const bounds = levels.map(() => null)
  for (const c of sorted) {
    const b = bounds[c.level]
    if (b === null) {
      bounds[c.level] = { min_zyx: [c.cz, c.cy, c.cx], max_zyx: [c.cz, c.cy, c.cx] }
      continue
    }
    b.min_zyx = [Math.min(b.min_zyx[0], c.cz), Math.min(b.min_zyx[1], c.cy), Math.min(b.min_zyx[2], c.cx)]
    b.max_zyx = [Math.max(b.max_zyx[0], c.cz), Math.max(b.max_zyx[1], c.cy), Math.max(b.max_zyx[2], c.cx)]
  }
  const windows = bounds.map(b => b === null ? null
    : [0, 1, 2].map(a => b.min_zyx[a] + Math.floor((b.max_zyx[a] - b.min_zyx[a] + 1 - windowChunks) / 2)))

  const kept = sorted.filter(c => {
    const origin_zyx = windows[c.level]
    return [c.cz, c.cy, c.cx].every((i, a) => i - origin_zyx[a] >= 0 && i - origin_zyx[a] < windowChunks)
  })

  const ahead = [...heading].filter(([key]) => !wanted.has(key)).map(([, c]) => c)
  return { displayLevels, wanted: kept, ahead, windows }
}
