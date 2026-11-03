import { test } from 'node:test'
import assert from 'node:assert/strict'
import { gzipSync } from 'node:zlib'
import { decodeChunk } from '../volume/decode.js'

// arange(64) as 4x4x4 big-endian uint16, numcodecs Blosc(cname="zstd", clevel=5, shuffle=SHUFFLE).encode().
const U16BE_BLOSC_ZSTD = Uint8Array.of(
  2, 1, 145, 2, 128, 0, 0, 0, 128, 0, 0, 0, 105, 0, 0, 0, 20, 0, 0, 0, 81, 0, 0, 0, 40, 181, 47, 253, 32, 128, 69, 2,
  0, 20, 4, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
  28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
  57, 58, 59, 60, 61, 62, 63, 1, 0, 148, 0, 22,
)

test('blosc-zstd big-endian uint16 decodes to arange(64) in native order', async () => {
  const codec = { name: 'blosc', cname: 'zstd', clevel: 5, shuffle: 1, blocksize: 0 }
  const data = await decodeChunk(U16BE_BLOSC_ZSTD, { codec, dtype: 'uint16', littleEndian: false, count: 64 })
  assert.ok(data instanceof Uint16Array)
  assert.deepEqual(Array.from(data), Array.from({ length: 64 }, (_, i) => i))
  assert.equal(data.buffer.byteLength, 128)
})

test('gzip little-endian float32 decodes to arange(64) * 0.5 - 3', async () => {
  const values = Float32Array.from({ length: 64 }, (_, i) => i * 0.5 - 3)
  const view = new DataView(new ArrayBuffer(256))
  values.forEach((v, i) => view.setFloat32(i * 4, v, true))
  const bytes = new Uint8Array(gzipSync(new Uint8Array(view.buffer), { level: 6 }))
  const data = await decodeChunk(bytes, { codec: { name: 'gzip', level: 6 }, dtype: 'float32', littleEndian: true, count: 64 })
  assert.ok(data instanceof Float32Array)
  assert.deepEqual(Array.from(data), Array.from(values))
})

test('raw bytes with the wrong element count throw', async () => {
  await assert.rejects(decodeChunk(new Uint8Array(6), { codec: { name: 'raw' }, dtype: 'uint16', littleEndian: true, count: 4 }), /decoded 6 bytes, expected 4 x 2 \(uint16\)/)
})
