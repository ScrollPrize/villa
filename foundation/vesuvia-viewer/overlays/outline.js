import { OUTLINE_BLOCK as BLOCK, OUTLINE_GROUP as GROUP, SURFACE_DEMAND_TILES } from '../config.js'

export const NORMAL_EPSILON = 1e-6

const missesCut = (bounds_6, b, normal_zyx, focus_zyx, min_zyx, max_zyx) => {
  let distance = 0
  let radius = 0
  for (let a = 0; a < 3; a++) {
    const lo = bounds_6[b * 6 + a], hi = bounds_6[b * 6 + 3 + a]
    if (lo > max_zyx[a] || hi < min_zyx[a]) return true
    distance += normal_zyx[a] * ((lo + hi) / 2 - focus_zyx[a])
    radius += Math.abs(normal_zyx[a]) * (hi - lo) / 2
  }
  return Math.abs(distance) > radius
}

const boxDistanceSquared = (bounds_6, b, wz, wy, wx, padding) => {
  const dz = Math.max(bounds_6[b * 6] - padding - wz, 0, wz - bounds_6[b * 6 + 3] - padding)
  const dy = Math.max(bounds_6[b * 6 + 1] - padding - wy, 0, wy - bounds_6[b * 6 + 4] - padding)
  const dx = Math.max(bounds_6[b * 6 + 2] - padding - wx, 0, wx - bounds_6[b * 6 + 5] - padding)
  return dz * dz + dy * dy + dx * dx
}

export function buildOutlineIndex(surface) {
  const { H, W, points_zyx_HW3, valid_HW } = surface
  const blockRows = Math.max(1, Math.ceil((H - 1) / BLOCK))
  const blockCols = Math.max(1, Math.ceil((W - 1) / BLOCK))
  const bounds_B6 = new Float32Array(blockRows * blockCols * 6)
  const valid_B = new Uint8Array(blockRows * blockCols)
  const holes_B = new Uint8Array(blockRows * blockCols)
  for (let bv = 0; bv < blockRows; bv++) {
    for (let bu = 0; bu < blockCols; bu++) {
      const b = bv * blockCols + bu
      let z0 = Infinity, y0 = Infinity, x0 = Infinity, z1 = -Infinity, y1 = -Infinity, x1 = -Infinity
      const vEnd = Math.min(H, bv * BLOCK + BLOCK + 1)
      const uEnd = Math.min(W, bu * BLOCK + BLOCK + 1)
      for (let v = bv * BLOCK; v < vEnd; v++) {
        for (let u = bu * BLOCK; u < uEnd; u++) {
          const i = v * W + u
          if (!valid_HW[i]) { holes_B[b] = 1; continue }
          const z = points_zyx_HW3[i * 3], y = points_zyx_HW3[i * 3 + 1], x = points_zyx_HW3[i * 3 + 2]
          if (z < z0) z0 = z
          if (y < y0) y0 = y
          if (x < x0) x0 = x
          if (z > z1) z1 = z
          if (y > y1) y1 = y
          if (x > x1) x1 = x
        }
      }
      valid_B[b] = z0 !== Infinity ? 1 : 0
      bounds_B6.set([z0, y0, x0, z1, y1, x1], b * 6)
    }
  }

  const groupRows = Math.ceil(blockRows / GROUP)
  const groupCols = Math.ceil(blockCols / GROUP)
  const bounds_G6 = new Float32Array(groupRows * groupCols * 6)
  const valid_G = new Uint8Array(groupRows * groupCols)
  for (let gv = 0; gv < groupRows; gv++) {
    for (let gu = 0; gu < groupCols; gu++) {
      const g = gv * groupCols + gu
      let z0 = Infinity, y0 = Infinity, x0 = Infinity, z1 = -Infinity, y1 = -Infinity, x1 = -Infinity
      for (let bv = gv * GROUP; bv < Math.min(blockRows, gv * GROUP + GROUP); bv++) {
        for (let bu = gu * GROUP; bu < Math.min(blockCols, gu * GROUP + GROUP); bu++) {
          const b = bv * blockCols + bu
          if (!valid_B[b]) continue
          if (bounds_B6[b * 6] < z0) z0 = bounds_B6[b * 6]
          if (bounds_B6[b * 6 + 1] < y0) y0 = bounds_B6[b * 6 + 1]
          if (bounds_B6[b * 6 + 2] < x0) x0 = bounds_B6[b * 6 + 2]
          if (bounds_B6[b * 6 + 3] > z1) z1 = bounds_B6[b * 6 + 3]
          if (bounds_B6[b * 6 + 4] > y1) y1 = bounds_B6[b * 6 + 4]
          if (bounds_B6[b * 6 + 5] > x1) x1 = bounds_B6[b * 6 + 5]
        }
      }
      valid_G[g] = z0 !== Infinity ? 1 : 0
      bounds_G6.set([z0, y0, x0, z1, y1, x1], g * 6)
    }
  }
  return { blockRows, blockCols, bounds_B6, valid_B, holes_B, groupRows, groupCols, bounds_G6, valid_G }
}

export function surfacePoint(surface, u, v, offset) {
  const { H, W, points_zyx_HW3, normals_zyx_HW3, valid_HW } = surface
  if (!(u >= 0 && v >= 0 && u <= W - 1 && v <= H - 1)) return null
  const u0 = Math.floor(u), v0 = Math.floor(v)
  const u1 = Math.min(u0 + 1, W - 1), v1 = Math.min(v0 + 1, H - 1)
  const corners = [v0 * W + u0, v0 * W + u1, v1 * W + u0, v1 * W + u1]
  if (corners.some(i => !valid_HW[i])) return null
  const fu = u - u0, fv = v - v0
  const weights = [(1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv]
  const point_zyx = [0, 0, 0], normal_zyx = [0, 0, 0]
  for (let j = 0; j < 4; j++) for (let a = 0; a < 3; a++) {
    point_zyx[a] += weights[j] * points_zyx_HW3[corners[j] * 3 + a]
    normal_zyx[a] += weights[j] * normals_zyx_HW3[corners[j] * 3 + a]
  }
  const length = Math.max(Math.hypot(...normal_zyx), NORMAL_EPSILON)
  return point_zyx.map((p, a) => p + offset * normal_zyx[a] / length)
}

export function surfaceBounds(index, min_uv, max_uv, reach, padding_zyx) {
  const { blockRows, blockCols, bounds_B6, valid_B } = index
  const u0 = Math.min(blockCols - 1, Math.max(0, Math.floor(min_uv[0] / BLOCK)))
  const v0 = Math.min(blockRows - 1, Math.max(0, Math.floor(min_uv[1] / BLOCK)))
  const u1 = Math.min(blockCols - 1, Math.floor(max_uv[0] / BLOCK))
  const v1 = Math.min(blockRows - 1, Math.floor(max_uv[1] / BLOCK))
  const stepU = Math.max(1, Math.ceil((u1 - u0 + 1) / SURFACE_DEMAND_TILES))
  const stepV = Math.max(1, Math.ceil((v1 - v0 + 1) / SURFACE_DEMAND_TILES))
  const boxes = []
  for (let v = v0; v <= v1; v += stepV) for (let u = u0; u <= u1; u += stepU) {
    const min_zyx = [Infinity, Infinity, Infinity], max_zyx = [-Infinity, -Infinity, -Infinity]
    for (let bv = v; bv <= Math.min(v1, v + stepV - 1); bv++) {
      for (let bu = u; bu <= Math.min(u1, u + stepU - 1); bu++) {
        const b = bv * blockCols + bu
        if (!valid_B[b]) continue
        for (let a = 0; a < 3; a++) {
          min_zyx[a] = Math.min(min_zyx[a], bounds_B6[b * 6 + a])
          max_zyx[a] = Math.max(max_zyx[a], bounds_B6[b * 6 + 3 + a])
        }
      }
    }
    if (min_zyx[0] === Infinity) continue
    boxes.push({ min_zyx: min_zyx.map((p, a) => p - reach - padding_zyx[a]),
      max_zyx: max_zyx.map((p, a) => p + reach + padding_zyx[a]), band: 0 })
  }
  return boxes
}

export function outlineSegments(index, surface, normal_zyx, focus_zyx, min_zyx, max_zyx, stride = 1) {
  const { H, W, points_zyx_HW3, valid_HW } = surface
  const { blockRows, blockCols, bounds_B6, valid_B, holes_B, groupRows, groupCols, bounds_G6, valid_G } = index
  const out = []
  const corners = [0, 0, 0]
  const crossing = new Float32Array(6)
  const distances_N = new Float64Array((BLOCK + 1) ** 2)
  const signed = [0, 0, 0]
  for (let gv = 0; gv < groupRows; gv++) {
    for (let gu = 0; gu < groupCols; gu++) {
      const g = gv * groupCols + gu
      if (!valid_G[g] || missesCut(bounds_G6, g, normal_zyx, focus_zyx, min_zyx, max_zyx)) continue
      for (let bv = gv * GROUP; bv < Math.min(blockRows, gv * GROUP + GROUP); bv++) {
        for (let bu = gu * GROUP; bu < Math.min(blockCols, gu * GROUP + GROUP); bu++) {
          const b = bv * blockCols + bu
          if (!valid_B[b] || missesCut(bounds_B6, b, normal_zyx, focus_zyx, min_zyx, max_zyx)) continue
          const vStart = bv * BLOCK, uStart = bu * BLOCK
          const vEnd = Math.min(H - 1, vStart + BLOCK)
          const uEnd = Math.min(W - 1, uStart + BLOCK)
          const step = holes_B[b] ? 1 : stride
          const rows = Math.ceil((vEnd - vStart) / step)
          const cols = Math.ceil((uEnd - uStart) / step)
          for (let row = 0; row <= rows; row++) {
            const v = Math.min(vStart + row * step, vEnd)
            for (let col = 0; col <= cols; col++) {
              const u = Math.min(uStart + col * step, uEnd), i = v * W + u
              if (!valid_HW[i]) continue
              let distance = 0
              for (let a = 0; a < 3; a++) distance += normal_zyx[a] * (points_zyx_HW3[i * 3 + a] - focus_zyx[a])
              distances_N[row * (cols + 1) + col] = distance
            }
          }
          for (let row = 0; row < rows; row++) {
            const v = vStart + row * step, nextV = Math.min(v + step, vEnd)
            for (let col = 0; col < cols; col++) {
              const u = uStart + col * step, nextU = Math.min(u + step, uEnd)
              const i00 = v * W + u, i01 = v * W + nextU, i10 = nextV * W + u, i11 = nextV * W + nextU
              if (!(valid_HW[i00] && valid_HW[i01] && valid_HW[i10] && valid_HW[i11])) continue
              const d00 = row * (cols + 1) + col, d01 = d00 + 1, d10 = d00 + cols + 1, d11 = d10 + 1
              const positive = distances_N[d00] >= 0
              if ((distances_N[d01] >= 0) === positive && (distances_N[d10] >= 0) === positive && (distances_N[d11] >= 0) === positive) continue
              for (let triangle = 0; triangle < 2; triangle++) {
                corners[0] = i00
                corners[1] = triangle === 0 ? i01 : i11
                corners[2] = triangle === 0 ? i11 : i10
                signed[0] = distances_N[d00]
                signed[1] = distances_N[triangle === 0 ? d01 : d11]
                signed[2] = distances_N[triangle === 0 ? d11 : d10]
                let found = 0
                for (let e = 0; e < 3; e++) {
                  const ia = corners[e], ib = corners[(e + 1) % 3]
                  const sa = signed[e], sb = signed[(e + 1) % 3]
                  if ((sa >= 0) === (sb >= 0)) continue
                  const t = sa / (sa - sb)
                  for (let k = 0; k < 3; k++) {
                    const pa = points_zyx_HW3[ia * 3 + k]
                    crossing[found * 3 + k] = pa + t * (points_zyx_HW3[ib * 3 + k] - pa)
                  }
                  found++
                }
                if (found === 2) out.push(crossing[0], crossing[1], crossing[2], crossing[3], crossing[4], crossing[5])
              }
            }
          }
        }
      }
    }
  }
  return Float32Array.from(out)
}

export function outlineNearestPoint(index, surface, world_zyx, maxDistance, offset = 0) {
  const { H, W, points_zyx_HW3, normals_zyx_HW3, valid_HW } = surface
  const { blockRows, blockCols, bounds_B6, valid_B, groupRows, groupCols, bounds_G6, valid_G } = index
  const [wz, wy, wx] = world_zyx
  const padding = Math.abs(offset)
  let bestU = -1, bestV = -1, bestSquared = maxDistance * maxDistance
  for (let gv = 0; gv < groupRows; gv++) {
    for (let gu = 0; gu < groupCols; gu++) {
      const g = gv * groupCols + gu
      if (!valid_G[g] || boxDistanceSquared(bounds_G6, g, wz, wy, wx, padding) > bestSquared) continue
      for (let bv = gv * GROUP; bv < Math.min(blockRows, gv * GROUP + GROUP); bv++) {
        for (let bu = gu * GROUP; bu < Math.min(blockCols, gu * GROUP + GROUP); bu++) {
          const b = bv * blockCols + bu
          if (!valid_B[b] || boxDistanceSquared(bounds_B6, b, wz, wy, wx, padding) > bestSquared) continue
          const vEnd = Math.min(H, bv * BLOCK + BLOCK + 1)
          const uEnd = Math.min(W, bu * BLOCK + BLOCK + 1)
          for (let v = bv * BLOCK; v < vEnd; v++) {
            for (let u = bu * BLOCK; u < uEnd; u++) {
              const i = v * W + u
              if (!valid_HW[i]) continue
              let dz = points_zyx_HW3[i * 3] - wz, dy = points_zyx_HW3[i * 3 + 1] - wy, dx = points_zyx_HW3[i * 3 + 2] - wx
              if (offset !== 0) {
                const nz = normals_zyx_HW3[i * 3], ny = normals_zyx_HW3[i * 3 + 1], nx = normals_zyx_HW3[i * 3 + 2]
                const scale = offset / Math.max(Math.hypot(nz, ny, nx), NORMAL_EPSILON)
                dz += nz * scale
                dy += ny * scale
                dx += nx * scale
              }
              const squared = dz * dz + dy * dy + dx * dx
              if (squared <= bestSquared) { bestU = u; bestV = v; bestSquared = squared }
            }
          }
        }
      }
    }
  }
  return { u: bestU, v: bestV, distance: bestU < 0 ? Infinity : Math.sqrt(bestSquared) }
}
