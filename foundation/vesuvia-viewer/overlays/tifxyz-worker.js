import { openTifxyz } from './tifxyz.js'
import { buildOutlineIndex } from './outline.js'

self.onmessage = async ({ data: { url, maxPoints } }) => {
  try {
    const surface = await openTifxyz(url, { maxPoints })
    const index = buildOutlineIndex(surface)
    self.postMessage({ surface, index }, [
      surface.points_zyx_HW3.buffer, surface.normals_zyx_HW3.buffer, surface.valid_HW.buffer,
      index.bounds_B6.buffer, index.valid_B.buffer, index.holes_B.buffer, index.bounds_G6.buffer, index.valid_G.buffer,
    ])
  } catch (error) {
    self.postMessage({ error: error.message })
  }
}
