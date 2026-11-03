import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readHash, formatHash } from '../ui/hash.js'

test('hash preserves escaped URL delimiters and small positive zoom', () => {
  const state = readHash('#ct=https://h/a?token=a%2Bb%26part=2&focus=-1.25,2,3&zoom=0.00001&level=2&atlas=1024')
  assert.equal(state.ct, 'https://h/a?token=a+b&part=2')
  assert.deepEqual(state.focus_zyx, [-1.25, 2, 3])
  assert.equal(state.zoom, 0.00001)
  assert.equal(state.level, 2)
  assert.equal(state.atlas, 1024)
  assert.equal(readHash('#zoom=2').atlas, null)
})

test('hash rejects invalid sampling poses and a collapsed window', () => {
  for (const hash of ['#zoom=0', '#zoom=-1', '#level=-1', '#level=1.5', '#atlas=0', '#atlas=1.5', '#atlas=big', '#focus=1,,2', '#black=5&white=5']) {
    assert.throws(() => readHash(hash), /hash /)
  }
})


test('repeated layer URLs keep blend order', () => {
  assert.deepEqual(readHash('#layer=https://h/a?token=x%26y&layer=https://h/b').layers,
    ['https://h/a?token=x&y', 'https://h/b'])
  assert.deepEqual(readHash('#ct=testdata/ct.zarr').layers, [])
})


test('hash restores an orthonormal rotated basis from rounded axes', () => {
  const { axes_zyx } = readHash('#axes=0.8,0.6,0,-0.6,0.8,0,0,0,1')
  assert.deepEqual(axes_zyx, [[0.8, 0.6, 0], [-0.6, 0.8, 0], [0, 0, 1]])
  const scaled = readHash('#axes=0,2,0,-3,0,0,0,0,4').axes_zyx
  assert.deepEqual(scaled, [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
  assert.equal(readHash('#ct=testdata/ct.zarr').axes_zyx, null)
})

test('hash rejects malformed, collapsed and reflected bases', () => {
  for (const axes of ['1,0,0', '1,0,0,0,1,0,0,,1', '1,0,0,2,0,0,0,0,1', '1,0,0,0,1,0,0,0,-1']) {
    assert.throws(() => readHash(`#axes=${axes}`))
  }
})

test('hash lists extension modules in order, none by default', () => {
  assert.deepEqual(readHash('#ct=a&ext=a.js&ext=https://h/b.js?v=1%262').extensions, ['a.js', 'https://h/b.js?v=1&2'])
  assert.deepEqual(readHash('#ct=a').extensions, [])
})


test('a link round-trips both poses, slab, surface image, layer styles and overlay visibility', () => {
  const hash = '#layer=a&layer=b&surfaceZoom=3&uv=20.5,30.5&offset=20&slab=5,7,2,max&overlays=0,0&image=https://h/i.tif%3Fv%3D1%262&imageOpacity=0.4'
    + '&style=labels,0,0.3,0.6,1,0.5,0&style=probability,1,0.8,0.2,0,0.5,1'
  const state = readHash(hash)
  assert.equal(state.surfaceZoom, 3)
  assert.deepEqual(state.slab, { front: 5, back: 7, step: 2, mode: 'max' })
  assert.deepEqual(state.overlays, { outline: false, crosshair: false })
  assert.equal(state.image, 'https://h/i.tif?v=1&2')
  assert.equal(state.imageOpacity, 0.4)
  assert.deepEqual(state.layerStyles[0], { mode: 'labels', visible: false, opacity: 0.3, threshold: 0.6, color: [1, 0.5, 0] })
  assert.deepEqual(readHash(formatHash(state)), state)
  const absent = readHash('#layer=a&layer=b&style=&style=labels,1,1,0,1,0,0')
  assert.deepEqual(readHash(formatHash(absent)), absent)
})

test('invalid surface, slab, style and overlay hash fields fail with their values', () => {
  for (const hash of ['#surfaceZoom=0', '#slab=-1,0,1,mean', '#slab=0,0,0,max', '#slab=1,2,3,unknown',
    '#slab=0,0,1,mean,extra', '#overlays=1,2,0', '#imageOpacity=1.5', '#imageOpacity=-0.1', '#style=labels,1,1,0,1,0,0',
    '#layer=a&style=labels,1,2,0,1,0,0', '#layer=a&style=labels,1,1,0,1,0', '#layer=a&style=other,1,1,0,1,0,0']) {
    assert.throws(() => readHash(hash), /hash /)
  }
})


test('a malformed repeated style reports that value, not the first style', () => {
  assert.throws(() => readHash('#layer=a&layer=b&style=labels,1,1,0,1,0,0&style=oops'),
    { message: 'hash style="oops": expected 6 number(s)' })
})
