import { test } from 'node:test'
import assert from 'node:assert/strict'
import { parseTiffFloat32, surfaceNormals } from '../overlays/tifxyz.js'

function encodeTiff({ H, W, values_HW, bigTiff, little, rowsPerStrip, compression }) {
  const stripCount = Math.ceil(H / rowsPerStrip)
  const entries = [
    [256, 4, [W]], [257, 4, [H]], [258, 3, [32]], [259, 3, [compression]], [262, 3, [1]],
    [273, bigTiff ? 16 : 4, new Array(stripCount)], [277, 3, [1]], [278, 4, [rowsPerStrip]],
    [279, bigTiff ? 16 : 4, new Array(stripCount)], [284, 3, [1]], [339, 3, [3]],
  ].filter(([tag]) => compression !== undefined || (tag !== 259 && tag !== 277))
  const stripOffsets = entries.find(([tag]) => tag === 273)[2]
  const stripByteCounts = entries.find(([tag]) => tag === 279)[2]
  const typeBytes = { 3: 2, 4: 4, 16: 8 }
  const headerBytes = bigTiff ? 16 : 8
  const entryBytes = bigTiff ? 20 : 12
  const inlineBytes = bigTiff ? 8 : 4
  const ifdBytes = (bigTiff ? 8 : 2) + entries.length * entryBytes + (bigTiff ? 8 : 4)
  let extra = headerBytes + ifdBytes
  const extraOffsets = entries.map(([, type, vals]) => {
    if (typeBytes[type] * vals.length <= inlineBytes) return null
    const at = extra
    extra += typeBytes[type] * vals.length
    return at
  })
  const dataStart = extra + (extra % 2)
  for (let s = 0; s < stripCount; s++) {
    const rows = Math.min(rowsPerStrip, H - s * rowsPerStrip)
    stripOffsets[s] = dataStart + s * rowsPerStrip * W * 4
    stripByteCounts[s] = rows * W * 4
  }
  const buffer = new ArrayBuffer(dataStart + H * W * 4)
  const view = new DataView(buffer)
  const put = (offset, type, value) => {
    if (type === 3) view.setUint16(offset, value, little)
    else if (type === 4) view.setUint32(offset, value, little)
    else view.setBigUint64(offset, BigInt(value), little)
  }
  view.setUint16(0, little ? 0x4949 : 0x4d4d, false)
  view.setUint16(2, bigTiff ? 43 : 42, little)
  if (bigTiff) {
    view.setUint16(4, 8, little)
    view.setBigUint64(8, BigInt(headerBytes), little)
    view.setBigUint64(headerBytes, BigInt(entries.length), little)
  } else {
    view.setUint32(4, headerBytes, little)
    view.setUint16(headerBytes, entries.length, little)
  }
  entries.forEach(([tag, type, vals], i) => {
    const entry = headerBytes + (bigTiff ? 8 : 2) + i * entryBytes
    view.setUint16(entry, tag, little)
    view.setUint16(entry + 2, type, little)
    put(entry + 4, bigTiff ? 16 : 4, vals.length)
    const field = entry + (bigTiff ? 12 : 8)
    if (extraOffsets[i] === null) vals.forEach((v, k) => put(field + k * typeBytes[type], type, v))
    else {
      put(field, bigTiff ? 16 : 4, extraOffsets[i])
      vals.forEach((v, k) => put(extraOffsets[i] + k * typeBytes[type], type, v))
    }
  })
  values_HW.forEach((v, i) => view.setFloat32(dataStart + i * 4, v, little))
  return buffer
}

test('classic little-endian TIFF with three strips and default Compression and SamplesPerPixel round-trips', () => {
  const H = 5, W = 3
  const values_HW = Float32Array.from({ length: H * W }, (_, i) => i * 0.25 - 1)
  const buffer = encodeTiff({ H, W, values_HW, bigTiff: false, little: true, rowsPerStrip: 2 })
  const image = parseTiffFloat32(buffer, 'x.tif')
  assert.equal(image.H, H)
  assert.equal(image.W, W)
  assert.deepEqual(Array.from(image.data_HW), Array.from(values_HW))
})

test('big-endian BigTIFF with one inline strip round-trips', () => {
  const H = 2, W = 2
  const values_HW = Float32Array.from([-1, 1.5, 4096.75, -0.125])
  const buffer = encodeTiff({ H, W, values_HW, bigTiff: true, little: false, rowsPerStrip: H, compression: 1 })
  const image = parseTiffFloat32(buffer, 'y.tif')
  assert.equal(image.H, H)
  assert.equal(image.W, W)
  assert.deepEqual(Array.from(image.data_HW), Array.from(values_HW))
})

test('a compressed TIFF throws naming the file and the tag value', () => {
  const buffer = encodeTiff({ H: 2, W: 2, values_HW: new Float32Array(4), bigTiff: false, little: true, rowsPerStrip: 2, compression: 5 })
  assert.throws(() => parseTiffFloat32(buffer, 'z.tif'), /z\.tif.*Compression 5/)
})

test('normals of a tilted plane are the plane normal, zero at holes', () => {
  const H = 5, W = 5
  const points_zyx_HW3 = new Float32Array(H * W * 3)
  const valid_HW = new Uint8Array(H * W).fill(1)
  for (let v = 0; v < H; v++) {
    for (let u = 0; u < W; u++) points_zyx_HW3.set([0.5 * u + 0.25 * v, v, u], (v * W + u) * 3)
  }
  valid_HW[2 * W + 2] = 0
  points_zyx_HW3.set([-1, -1, -1], (2 * W + 2) * 3)
  const normals_zyx_HW3 = surfaceNormals({ H, W, points_zyx_HW3, valid_HW })
  const expected_zyx = [-1, 0.25, 0.5].map((c) => c / Math.hypot(1, 0.25, 0.5))
  for (let i = 0; i < H * W; i++) {
    const n_zyx = Array.from(normals_zyx_HW3.subarray(i * 3, i * 3 + 3))
    if (!valid_HW[i]) {
      assert.deepEqual(n_zyx, [0, 0, 0])
      continue
    }
    for (let k = 0; k < 3; k++) assert.ok(Math.abs(n_zyx[k] - expected_zyx[k]) < 1e-6, `point ${i}: ${n_zyx}`)
  }
})
