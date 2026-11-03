import { test } from 'node:test'
import assert from 'node:assert/strict'
import { planChunks } from '../views/plan.js'

const LEVELS = [
  { index: 0, shape_zyx: [128, 128, 128], chunk_zyx: [32, 32, 32], downsample_zyx: [1, 1, 1], offset_zyx: [0, 0, 0] },
  { index: 1, shape_zyx: [64, 64, 64], chunk_zyx: [32, 32, 32], downsample_zyx: [2, 2, 2], offset_zyx: [0, 0, 0] },
]
const THREE_LEVELS = [
  ...LEVELS,
  { index: 2, shape_zyx: [32, 32, 32], chunk_zyx: [32, 32, 32], downsample_zyx: [4, 4, 4], offset_zyx: [0, 0, 0] },
]
const box = (min_zyx, max_zyx, band = 0) => ({ min_zyx, max_zyx, band })
const plan = (levels, views, extra = {}) =>
  planChunks({ levels, views, minDownsample: 1, slotCapacity: 1000, windowChunks: 64, pinCoarsestChunks: 0, ...extra })

test('a box enumerates chunks at every level from display to coarsest, coarse first', () => {
  const p = plan(LEVELS, [{ boxes: [box([10, 10, 10], [40, 40, 40])], zoom: 1 }])
  assert.deepEqual(p.displayLevels, [0])
  const keys = p.wanted.map(c => `${c.level}/${c.cz}/${c.cy}/${c.cx}`).sort()
  const expected = ['1/0/0/0']
  for (const cz of [0, 1]) for (const cy of [0, 1]) for (const cx of [0, 1]) expected.push(`0/${cz}/${cy}/${cx}`)
  assert.deepEqual(keys, expected.sort())
  assert.equal(p.wanted[0].level, 1)
  assert.ok(Math.abs(p.wanted[0].priority - Math.sqrt(3) * (0.5 - 12.25 / 32)) < 1e-12)
  assert.ok(p.wanted.slice(1).every(c => c.level === 0 && c.priority >= 100 && c.priority < 200))
  for (let i = 1; i < p.wanted.length; i++) assert.ok(p.wanted[i - 1].priority <= p.wanted[i].priority)
})

test('display level is the coarsest downsample within zoom * tolerance, capped below by minDownsample', () => {
  const at = (zoom, minDownsample = 1) =>
    plan(THREE_LEVELS, [{ boxes: [box([0, 0, 0], [8, 8, 8])], zoom }], { minDownsample }).displayLevels[0]
  assert.equal(at(0.5), 0)
  assert.equal(at(1), 0)
  assert.equal(at(2.6), 1)
  assert.equal(at(2.8), 1)
  assert.equal(at(3), 2)
  assert.equal(at(1, 4), 2)
})

test('a level of one plane, pooled in y and x alone, is as coarse as its y and x', () => {
  const plane = [1, 2, 4].map((d, index) => ({ index, shape_zyx: [1, 256 / d, 256 / d], chunk_zyx: [1, 64, 64], downsample_zyx: [1, d, d], offset_zyx: [0, 0, 0] }))
  const at = zoom => plan(plane, [{ boxes: [box([0, 0, 0], [0, 100, 100])], zoom }])
  assert.deepEqual(at(1).displayLevels, [0])
  const p = at(3)
  assert.deepEqual(p.displayLevels, [2])
  assert.deepEqual(p.wanted.map(c => `${c.level}/${c.cz}/${c.cy}/${c.cx}`), ['2/0/0/0'])
})

test('fit rule drops prefetch, then coarsens, then keeps the nearest chunks', () => {
  const views = [{ boxes: [box([0, 0, 0], [62.5, 62.5, 62.5], 0), box([64, 0, 0], [127, 62.5, 62.5], 1)], zoom: 1 }]
  const fits = plan(LEVELS, views, { slotCapacity: 27 })
  assert.equal(fits.wanted.length, 24)
  assert.equal(fits.wanted.filter(c => c.priority >= 1000).length, 8)

  const noPrefetch = plan(LEVELS, views, { slotCapacity: 25 })
  assert.deepEqual(noPrefetch.displayLevels, [0])
  assert.equal(noPrefetch.wanted.length, 16)
  assert.ok(noPrefetch.wanted.every(c => c.priority < 1000))

  const coarsened = plan(LEVELS, views, { slotCapacity: 17 })
  assert.deepEqual(coarsened.displayLevels, [1])
  assert.equal(coarsened.wanted.length, 8)
  assert.ok(coarsened.wanted.every(c => c.level === 1))

  const truncated = plan(LEVELS, views, { slotCapacity: 7 })
  assert.deepEqual(truncated.displayLevels, [1])
  assert.equal(truncated.wanted.length, 6)
  for (let i = 1; i < truncated.wanted.length; i++) assert.ok(truncated.wanted[i - 1].priority <= truncated.wanted[i].priority)
})

test('windows centre on the wanted bounding box and keep only the chunks inside', () => {
  const views = [{ boxes: [box([0, 0, 0], [127, 127, 127])], zoom: 1 }]
  const p = plan(LEVELS, views, { windowChunks: 2 })
  assert.deepEqual(p.windows, [[1, 1, 1], [0, 0, 0]])
  assert.equal(p.wanted.filter(c => c.level === 0).length, 8)
  assert.ok(p.wanted.filter(c => c.level === 0).every(c => [c.cz, c.cy, c.cx].every(i => i === 1 || i === 2)))
  assert.equal(p.wanted.filter(c => c.level === 1).length, 8)

  const single = plan(LEVELS, [{ boxes: [box([70, 70, 70], [80, 80, 80])], zoom: 1 }], { windowChunks: 4 })
  assert.deepEqual(single.windows, [[0, 0, 0], [-1, -1, -1]])
  const wide = plan(LEVELS, [{ boxes: [box([0, 0, 0], [8, 8, 8])], zoom: 4 }], { minDownsample: 1 })
  assert.deepEqual(wide.windows, [null, [-32, -32, -32]])
})

test('fit counts a view\u0027s overlapping boxes only once per chunk', () => {
  const small = box([10, 10, 10], [11, 11, 11])
  const p = plan(LEVELS, [
    { boxes: Array(20).fill(small), zoom: 1 },
    { boxes: [box([0, 0, 0], [62.5, 62.5, 62.5])], zoom: 1 },
  ], { slotCapacity: 10 })
  assert.deepEqual(p.displayLevels, [0, 1])
  assert.equal(p.wanted.filter(c => c.level === 0).length, 1)
})

test('fit drops intermediate fallback levels before coarsening visible detail', () => {
  const p = plan(THREE_LEVELS, [{ boxes: [box([10, 10, 10], [11, 11, 11])], zoom: 1 }], { slotCapacity: 3 })
  assert.deepEqual(p.displayLevels, [0])
  assert.deepEqual(p.wanted.map(c => `${c.level}/${c.cz}/${c.cy}/${c.cx}`), ['2/0/0/0', '0/0/0/0'])
})

test('a small coarsest level is wanted whole, after everything the views asked for', () => {
  const views = [{ boxes: [box([10, 10, 10], [20, 20, 20])], zoom: 1 }]
  const pinned = plan(LEVELS, views, { pinCoarsestChunks: 8 }).wanted.filter(c => c.level === 1)
  assert.equal(pinned.length, 8)
  assert.deepEqual(pinned.map(c => c.views.length), [1, 0, 0, 0, 0, 0, 0, 0])
  assert.ok(pinned[0].priority < 100 && pinned.slice(1).every(c => c.priority === 2000))
  assert.equal(plan(LEVELS, views, { pinCoarsestChunks: 7 }).wanted.filter(c => c.level === 1).length, 1)
})

test('a heading box asks for the coarsest chunks nothing else wants, and counts in no fit', () => {
  const views = [{ boxes: [box([10, 10, 10], [20, 20, 20]), box([10, 10, 10], [100, 20, 20], -1)], zoom: 1 }]
  const p = plan(LEVELS, views, { slotCapacity: 3 })
  assert.deepEqual(p.displayLevels, [0])
  assert.deepEqual(p.wanted.map(c => `${c.level}/${c.cz}/${c.cy}/${c.cx}`), ['1/0/0/0', '0/0/0/0'])
  assert.deepEqual(p.ahead.map(c => `${c.level}/${c.cz}/${c.cy}/${c.cx}`), ['1/1/0/0'])
  assert.ok(p.ahead[0].priority < 100)
  assert.deepEqual(plan(LEVELS, [{ boxes: [box([10, 10, 10], [20, 20, 20])], zoom: 1 }]).ahead, [])
})
