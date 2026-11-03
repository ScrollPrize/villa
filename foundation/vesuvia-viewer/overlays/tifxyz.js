// A pixel equal to -1 in all three of x.tif, y.tif and z.tif is a hole.
import { fetchFile } from '../volume/mirrors.js'

const TAG_IMAGE_WIDTH = 256
const TAG_IMAGE_LENGTH = 257
const TAG_BITS_PER_SAMPLE = 258
const TAG_COMPRESSION = 259        // 1 = none
const TAG_STRIP_OFFSETS = 273
const TAG_SAMPLES_PER_PIXEL = 277
const TAG_ROWS_PER_STRIP = 278
const TAG_STRIP_BYTE_COUNTS = 279
const TAG_PLANAR_CONFIG = 284      // 1 = chunky (absent = 1)
const TAG_TILE_WIDTH = 322
const TAG_SAMPLE_FORMAT = 339      // 3 = IEEE float (absent = 1, unsigned int)
const WANTED_TAGS = new Set([TAG_IMAGE_WIDTH, TAG_IMAGE_LENGTH, TAG_BITS_PER_SAMPLE, TAG_COMPRESSION, TAG_STRIP_OFFSETS,
  TAG_SAMPLES_PER_PIXEL, TAG_ROWS_PER_STRIP, TAG_STRIP_BYTE_COUNTS, TAG_PLANAR_CONFIG, TAG_TILE_WIDTH, TAG_SAMPLE_FORMAT])

const TYPE_BYTES = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4, 16: 8, 17: 8, 18: 8 }
const NORMAL_EPSILON = 1e-8

function readValues(view, type, count, fieldOffset, inlineBytes, little) {
  const bytes = TYPE_BYTES[type]
  let offset = fieldOffset
  if (bytes * count > inlineBytes) {
    offset = inlineBytes === 8 ? Number(view.getBigUint64(fieldOffset, little)) : view.getUint32(fieldOffset, little)
  }
  const values = new Array(count)
  for (let i = 0; i < count; i++, offset += bytes) {
    if (type === 1 || type === 7) values[i] = view.getUint8(offset)
    else if (type === 3) values[i] = view.getUint16(offset, little)
    else if (type === 4 || type === 13) values[i] = view.getUint32(offset, little)
    else if (type === 16 || type === 18) values[i] = Number(view.getBigUint64(offset, little))
    else throw new Error(`TIFF field type ${type} is not an unsigned integer`)
  }
  return values
}

export function parseTiffFloat32(arrayBuffer, label) {
  const view = new DataView(arrayBuffer)
  const order = view.getUint16(0, false)
  if (order !== 0x4949 && order !== 0x4d4d) throw new Error(`${label}: not a TIFF (byte order ${order.toString(16)})`)
  const little = order === 0x4949
  const magic = view.getUint16(2, little)
  let ifdOffset, entryBytes, inlineBytes
  if (magic === 42) {
    ifdOffset = view.getUint32(4, little)
    entryBytes = 12
    inlineBytes = 4
  } else if (magic === 43) {
    const offsetBytes = view.getUint16(4, little)
    if (offsetBytes !== 8) throw new Error(`${label}: BigTIFF offset size ${offsetBytes}`)
    ifdOffset = Number(view.getBigUint64(8, little))
    entryBytes = 20
    inlineBytes = 8
  } else {
    throw new Error(`${label}: TIFF magic ${magic}`)
  }

  const entryCount = magic === 42 ? view.getUint16(ifdOffset, little) : Number(view.getBigUint64(ifdOffset, little))
  const entriesStart = ifdOffset + (magic === 42 ? 2 : 8)
  const tags = new Map()
  for (let i = 0; i < entryCount; i++) {
    const entry = entriesStart + i * entryBytes
    const tag = view.getUint16(entry, little)
    const type = view.getUint16(entry + 2, little)
    const count = magic === 42 ? view.getUint32(entry + 4, little) : Number(view.getBigUint64(entry + 4, little))
    if (WANTED_TAGS.has(tag)) tags.set(tag, readValues(view, type, count, entry + (magic === 42 ? 8 : 12), inlineBytes, little))
  }
  const one = (tag) => {
    if (!tags.has(tag)) throw new Error(`${label}: missing TIFF tag ${tag}`)
    return tags.get(tag)[0]
  }

  if (tags.has(TAG_TILE_WIDTH)) throw new Error(`${label}: tiled TIFF not supported (TileWidth ${one(TAG_TILE_WIDTH)})`)
  if (tags.has(TAG_COMPRESSION) && one(TAG_COMPRESSION) !== 1) throw new Error(`${label}: compressed TIFF not supported (Compression ${one(TAG_COMPRESSION)})`)
  if (one(TAG_BITS_PER_SAMPLE) !== 32) throw new Error(`${label}: BitsPerSample ${one(TAG_BITS_PER_SAMPLE)}, expected 32`)
  if (tags.has(TAG_SAMPLES_PER_PIXEL) && one(TAG_SAMPLES_PER_PIXEL) !== 1) throw new Error(`${label}: SamplesPerPixel ${one(TAG_SAMPLES_PER_PIXEL)}, expected 1`)
  const sampleFormat = tags.has(TAG_SAMPLE_FORMAT) ? one(TAG_SAMPLE_FORMAT) : 1
  if (sampleFormat !== 3) throw new Error(`${label}: SampleFormat ${sampleFormat}, expected 3 (float)`)
  if (tags.has(TAG_PLANAR_CONFIG) && one(TAG_PLANAR_CONFIG) !== 1) throw new Error(`${label}: PlanarConfiguration ${one(TAG_PLANAR_CONFIG)}, expected 1`)

  const W = one(TAG_IMAGE_WIDTH)
  const H = one(TAG_IMAGE_LENGTH)
  const rowsPerStrip = tags.has(TAG_ROWS_PER_STRIP) ? Math.min(one(TAG_ROWS_PER_STRIP), H) : H
  const offsets = tags.get(TAG_STRIP_OFFSETS)
  const byteCounts = tags.get(TAG_STRIP_BYTE_COUNTS)
  if (offsets === undefined || byteCounts === undefined) throw new Error(`${label}: missing StripOffsets or StripByteCounts`)
  const stripCount = Math.ceil(H / rowsPerStrip)
  if (offsets.length !== stripCount || byteCounts.length !== stripCount) {
    throw new Error(`${label}: ${offsets.length} strip offsets and ${byteCounts.length} byte counts for ${stripCount} strips`)
  }

  const data_HW = new Float32Array(H * W)
  const dst = new Uint8Array(data_HW.buffer)
  for (let s = 0; s < stripCount; s++) {
    const rows = Math.min(rowsPerStrip, H - s * rowsPerStrip)
    const expected = rows * W * 4
    if (byteCounts[s] !== expected) throw new Error(`${label}: strip ${s} holds ${byteCounts[s]} bytes, expected ${expected}`)
    if (offsets[s] + expected > arrayBuffer.byteLength) throw new Error(`${label}: strip ${s} at ${offsets[s]} runs past the file (${arrayBuffer.byteLength} bytes)`)
    dst.set(new Uint8Array(arrayBuffer, offsets[s], expected), s * rowsPerStrip * W * 4)
  }
  if (!little) {
    for (let i = 0; i < dst.length; i += 4) {
      const b0 = dst[i], b1 = dst[i + 1]
      dst[i] = dst[i + 3]; dst[i + 1] = dst[i + 2]; dst[i + 2] = b1; dst[i + 3] = b0
    }
  }
  return { H, W, data_HW }
}

export function surfaceNormals(surface) {
  const { H, W, points_zyx_HW3, valid_HW } = surface
  const normals_zyx_HW3 = new Float32Array(H * W * 3)
  for (let v = 0; v < H; v++) {
    for (let u = 0; u < W; u++) {
      const i = v * W + u
      if (!valid_HW[i]) continue
      let du0 = 0, du1 = 0, du2 = 0, dv0 = 0, dv1 = 0, dv2 = 0
      if (u + 1 < W && valid_HW[i + 1]) {
        du0 += points_zyx_HW3[(i + 1) * 3] - points_zyx_HW3[i * 3]
        du1 += points_zyx_HW3[(i + 1) * 3 + 1] - points_zyx_HW3[i * 3 + 1]
        du2 += points_zyx_HW3[(i + 1) * 3 + 2] - points_zyx_HW3[i * 3 + 2]
      }
      if (u > 0 && valid_HW[i - 1]) {
        du0 += points_zyx_HW3[i * 3] - points_zyx_HW3[(i - 1) * 3]
        du1 += points_zyx_HW3[i * 3 + 1] - points_zyx_HW3[(i - 1) * 3 + 1]
        du2 += points_zyx_HW3[i * 3 + 2] - points_zyx_HW3[(i - 1) * 3 + 2]
      }
      if (v + 1 < H && valid_HW[i + W]) {
        dv0 += points_zyx_HW3[(i + W) * 3] - points_zyx_HW3[i * 3]
        dv1 += points_zyx_HW3[(i + W) * 3 + 1] - points_zyx_HW3[i * 3 + 1]
        dv2 += points_zyx_HW3[(i + W) * 3 + 2] - points_zyx_HW3[i * 3 + 2]
      }
      if (v > 0 && valid_HW[i - W]) {
        dv0 += points_zyx_HW3[i * 3] - points_zyx_HW3[(i - W) * 3]
        dv1 += points_zyx_HW3[i * 3 + 1] - points_zyx_HW3[(i - W) * 3 + 1]
        dv2 += points_zyx_HW3[i * 3 + 2] - points_zyx_HW3[(i - W) * 3 + 2]
      }
      const n0 = du1 * dv2 - du2 * dv1
      const n1 = du2 * dv0 - du0 * dv2
      const n2 = du0 * dv1 - du1 * dv0
      const length = Math.max(Math.hypot(n0, n1, n2), NORMAL_EPSILON)
      normals_zyx_HW3[i * 3] = n0 / length
      normals_zyx_HW3[i * 3 + 1] = n1 / length
      normals_zyx_HW3[i * 3 + 2] = n2 / length
    }
  }
  return normals_zyx_HW3
}

// fullResolutionUV = gridIndex / scale_uv holds on the returned grid.
export async function openTifxyz(url, { maxPoints }) {
  const base = url.replace(/\/+$/, '')
  const fetchBytes = async (name) => {
    const file = new URL(base, globalThis.location?.href)
    file.pathname = `${file.pathname.replace(/\/+$/, '')}/${name}`
    return fetchFile(file.href)
  }
  const [metaBytes, xBytes, yBytes, zBytes] = await Promise.all(['meta.json', 'x.tif', 'y.tif', 'z.tif'].map(fetchBytes))
  const meta = JSON.parse(new TextDecoder().decode(metaBytes))
  const scale = meta['scale']
  if (!Array.isArray(scale) || scale.length !== 2 || !scale.every((s) => Number.isFinite(s) && s > 0)) {
    throw new Error(`${base}/meta.json: scale must be two positive numbers, got ${JSON.stringify(scale)}`)
  }
  const x = parseTiffFloat32(xBytes, `${base}/x.tif`)
  const y = parseTiffFloat32(yBytes, `${base}/y.tif`)
  const z = parseTiffFloat32(zBytes, `${base}/z.tif`)
  if (x.H !== y.H || x.H !== z.H || x.W !== y.W || x.W !== z.W) {
    throw new Error(`${base}: grid sizes differ: x ${x.H}x${x.W}, y ${y.H}x${y.W}, z ${z.H}x${z.W}`)
  }
  const fullH = x.H, fullW = x.W

  let stride = 1
  while (Math.ceil(fullH / stride) * Math.ceil(fullW / stride) > maxPoints) stride++
  const H = Math.ceil(fullH / stride)
  const W = Math.ceil(fullW / stride)
  const points_zyx_HW3 = new Float32Array(H * W * 3)
  const valid_HW = new Uint8Array(H * W)
  let validCount = 0
  for (let v = 0; v < H; v++) {
    for (let u = 0; u < W; u++) {
      const src = v * stride * fullW + u * stride
      const px = x.data_HW[src], py = y.data_HW[src], pz = z.data_HW[src]
      const i = v * W + u
      points_zyx_HW3[i * 3] = pz
      points_zyx_HW3[i * 3 + 1] = py
      points_zyx_HW3[i * 3 + 2] = px
      if (px === -1 && py === -1 && pz === -1) continue
      valid_HW[i] = 1
      validCount++
    }
  }
  if (validCount === 0) throw new Error(`${base}: no valid points`)

  let spacingSum = 0, spacingCount = 0
  for (let v = 0; v < H; v++) {
    for (let u = 0; u < W; u++) {
      const i = v * W + u
      if (!valid_HW[i]) continue
      for (const next of [u + 1 < W ? i + 1 : -1, v + 1 < H ? i + W : -1]) {
        if (next < 0 || !valid_HW[next]) continue
        spacingSum += Math.hypot(
          points_zyx_HW3[next * 3] - points_zyx_HW3[i * 3],
          points_zyx_HW3[next * 3 + 1] - points_zyx_HW3[i * 3 + 1],
          points_zyx_HW3[next * 3 + 2] - points_zyx_HW3[i * 3 + 2],
        )
        spacingCount++
      }
    }
  }
  if (spacingSum === 0) throw new Error(`${base}: no two distinct adjacent valid points`)

  const surface = { H, W, points_zyx_HW3, valid_HW }
  return {
    ...surface,
    normals_zyx_HW3: surfaceNormals(surface),
    scale_uv: [scale[0] / stride, scale[1] / stride],
    cellSpacing: spacingSum / spacingCount,
  }
}
