import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readRegion } from '../volume/region.js'

const LEVEL = { shape_zyx: [6, 6, 6], chunk_zyx: [4, 4, 4] }

function fakeVolume() {
  const asked = []
  return {
    asked, dtype: 'uint8', fill: 7, levels: [LEVEL],
    read(level, cz, cy, cx) {
      asked.push(`${cz}/${cy}/${cx}`)
      if (cz === 1 && cy === 1 && cx === 1) return Promise.resolve(null)
      const data = new Uint8Array(64)
      for (let i = 0; i < 64; i++) {
        const [z, y, x] = [cz * 4 + (i >> 4), cy * 4 + ((i >> 2) & 3), cx * 4 + (i & 3)]
        data[i] = Math.max(z, y, x) < 6 ? z * 36 + y * 6 + x : 255
      }
      return Promise.resolve(data)
    },
  }
}

test('a region spanning chunks reads only those chunks and leaves fill where one is missing', async () => {
  const volume = fakeVolume()
  const data_DHW = await readRegion(volume, 0, [2, 3, 3], [3, 3, 3])
  assert.deepEqual(volume.asked.sort(), ['0/0/0', '0/0/1', '0/1/0', '0/1/1', '1/0/0', '1/0/1', '1/1/0', '1/1/1'])

  const expected = []
  for (let z = 2; z < 5; z++) for (let y = 3; y < 6; y++) for (let x = 3; x < 6; x++) {
    expected.push(Math.min(z, y, x) >= 4 ? 7 : z * 36 + y * 6 + x)
  }
  assert.deepEqual(Array.from(data_DHW), expected)
})

test('a box that is not whole voxels inside the level rejects instead of shrinking, and so does a failed chunk', async () => {
  await assert.rejects(readRegion(fakeVolume(), 0, [4, 4, 4], [3, 2, 2]), /not whole voxels inside level 0/)
  await assert.rejects(readRegion(fakeVolume(), 0, [-1, 0, 0], [2, 2, 2]), /not whole voxels inside level 0/)
  await assert.rejects(readRegion(fakeVolume(), 0, [0.5, 0, 0], [2, 2, 2]), /not whole voxels inside level 0/)
  const volume = fakeVolume()
  volume.read = () => Promise.reject(new Error('HTTP 500'))
  await assert.rejects(readRegion(volume, 0, [0, 0, 0], [2, 2, 2]), /HTTP 500/)
})
