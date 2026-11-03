import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createMemoryVolume } from '../volume/memory.js'
import { readRegion } from '../volume/region.js'

const create = () => createMemoryVolume({
  name: 'test', dtype: 'uint8', fill: 7, labels: true,
  levels: [{ shape_zyx: [6, 6, 6], chunk_zyx: [4, 4, 4], downsample_zyx: [1, 1, 1], offset_zyx: [0, 0, 0] }],
})

test('a write spanning chunks reads back exactly, over fill, and each chunk it touched arrives once, changed', async () => {
  const volume = create()
  assert.equal(volume.cached(0, 0, 0, 0), null)
  const data_DHW = Uint8Array.from({ length: 2 * 3 * 2 }, (_, i) => i + 1)
  volume.write(0, [3, 2, 3], [2, 3, 2], data_DHW)
  volume.write(0, [3, 2, 3], [2, 3, 2], data_DHW)

  const whole_DHW = await readRegion(volume, 0, [0, 0, 0], [6, 6, 6])
  const expected = []
  for (let z = 0; z < 6; z++) for (let y = 0; y < 6; y++) for (let x = 0; x < 6; x++) {
    const inside = z >= 3 && z < 5 && y >= 2 && y < 5 && x >= 3 && x < 5
    expected.push(inside ? ((z - 3) * 3 + (y - 2)) * 2 + (x - 3) + 1 : 7)
  }
  assert.deepEqual(Array.from(whole_DHW), expected)

  const arrived = volume.take()
  assert.deepEqual(arrived.map(c => `${c.level}/${c.cz}/${c.cy}/${c.cx}`).sort(), ['0/0/0/0', '0/0/0/1', '0/0/1/0', '0/0/1/1', '0/1/0/0', '0/1/0/1', '0/1/1/0', '0/1/1/1'])
  assert.ok(arrived.every(c => c.data === volume.cached(c.level, c.cz, c.cy, c.cx) && c.data.length === 64))
  assert.ok(arrived.every(c => String(c.changed) === (c.cz === 0 ? '3,4' : '0,1')))
  assert.deepEqual(volume.take(), [])
})

test('a masked write changes only the voxels its mask names, and only the chunks those are in', async () => {
  const volume = create()
  volume.write(0, [0, 3, 0], [1, 2, 3], Uint8Array.of(1, 2, 3, 4, 5, 6), Uint8Array.of(0, 1, 0, 0, 0, 0))
  assert.deepEqual(Array.from(await readRegion(volume, 0, [0, 3, 0], [1, 2, 3])), [7, 2, 7, 7, 7, 7])
  assert.deepEqual(volume.take().map(c => `${c.cz}/${c.cy}/${c.cx} ${c.changed}`), ['0/0/0 0,1'])
  assert.equal(volume.cached(0, 0, 1, 0), null)
  volume.write(0, [1, 0, 0], [3, 1, 1], Uint8Array.of(1, 1, 1), Uint8Array.of(1, 0, 0))
  volume.write(0, [1, 0, 0], [3, 1, 1], Uint8Array.of(1, 1, 1), Uint8Array.of(0, 0, 1))
  assert.deepEqual(volume.take().map(c => `${c.cz}/${c.cy}/${c.cx} ${c.changed}`), ['0/0/0 1,4'])
  assert.throws(() => volume.write(0, [0, 0, 0], [1, 1, 2], new Uint8Array(2), new Uint8Array(3)), /with a mask of 3/)
})

test('a write that is not whole voxels inside the level, or not the size it says, throws', () => {
  const volume = create()
  assert.throws(() => volume.write(0, [4, 4, 4], [3, 2, 2], new Uint8Array(12)), /not whole voxels inside level 0/)
  assert.throws(() => volume.write(0, [0, 0, 0], [2, 2, 2], new Uint8Array(7)), /2×2×2 voxels got 7$/)
})
