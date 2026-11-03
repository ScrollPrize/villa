import { test } from 'node:test'
import assert from 'node:assert/strict'
import { buildOutlineIndex, outlineSegments, outlineNearestPoint, surfacePoint, surfaceBounds } from '../overlays/outline.js'

const MIN_ZYX = [-Infinity, -Infinity, -Infinity]
const MAX_ZYX = [Infinity, Infinity, Infinity]

function gridSurface(H, W, point) {
  const points_zyx_HW3 = new Float32Array(H * W * 3)
  const valid_HW = new Uint8Array(H * W).fill(1)
  for (let v = 0; v < H; v++) for (let u = 0; u < W; u++) points_zyx_HW3.set(point(v, u), (v * W + u) * 3)
  return { H, W, points_zyx_HW3, valid_HW }
}

function coveredRange(segments) {
  const spans = []
  for (let i = 0; i < segments.length; i += 6) spans.push([Math.min(segments[i + 1], segments[i + 4]), Math.max(segments[i + 1], segments[i + 4])])
  spans.sort((a, b) => a[0] - b[0])
  let end = spans[0][0]
  for (const [a, b] of spans) {
    assert.ok(a <= end + 1e-6, `gap before y=${a}`)
    end = Math.max(end, b)
  }
  return [spans[0][0], end]
}

const plane = gridSurface(5, 5, (v, u) => [u, v, u])

test('cutting z = x at z = 2.5 gives segments on x = 2.5 spanning y 0..4', () => {
  const index = buildOutlineIndex(plane)
  assert.equal(index.blockRows, 1)
  assert.equal(index.blockCols, 1)
  assert.deepEqual(Array.from(index.bounds_B6), [0, 0, 0, 4, 4, 4])
  assert.deepEqual(Array.from(index.valid_B), [1])
  assert.deepEqual(Array.from(index.bounds_G6), [0, 0, 0, 4, 4, 4])
  assert.deepEqual(Array.from(index.valid_G), [1])
  const segments = outlineSegments(index, plane, [1, 0, 0], [2.5, 0, 0], MIN_ZYX, MAX_ZYX)
  assert.ok(segments.length > 0 && segments.length % 6 === 0)
  for (let i = 0; i < segments.length; i += 3) {
    assert.ok(Math.abs(segments[i] - 2.5) < 1e-6, `z ${segments[i]}`)
    assert.ok(Math.abs(segments[i + 2] - 2.5) < 1e-6, `x ${segments[i + 2]}`)
  }
  assert.deepEqual(coveredRange(segments), [0, 4])
  assert.deepEqual(Array.from(outlineSegments(index, plane, [0, 0, 1], [0, 0, 2.5], MIN_ZYX, MAX_ZYX)), Array.from(segments))
})

test('a cut exactly through grid points still gives a full outline', () => {
  const segments = outlineSegments(buildOutlineIndex(plane), plane, [1, 0, 0], [2, 0, 0], MIN_ZYX, MAX_ZYX)
  for (let i = 0; i < segments.length; i += 3) assert.equal(segments[i + 2], 2)
  assert.deepEqual(coveredRange(segments), [0, 4])
})

test('a surface parallel to the cut plane gives no segments', () => {
  const flat = gridSurface(5, 5, (v, u) => [1, v, u])
  assert.equal(outlineSegments(buildOutlineIndex(flat), flat, [1, 0, 0], [2.5, 0, 0], MIN_ZYX, MAX_ZYX).length, 0)
  assert.equal(outlineSegments(buildOutlineIndex(flat), flat, [1, 0, 0], [1, 0, 0], MIN_ZYX, MAX_ZYX).length, 0)
})

test('cells touching a hole are skipped and blocks without valid points are marked', () => {
  const holed = gridSurface(20, 12, (v, u) => [u, v, u])
  for (let v = 0; v < 20; v++) for (let u = 0; u < 12; u++) if (v >= 8 && u >= 8) holed.valid_HW[v * 12 + u] = 0
  const index = buildOutlineIndex(holed)
  assert.equal(index.blockRows, 3)
  assert.equal(index.blockCols, 2)
  assert.deepEqual(Array.from(index.valid_B), [1, 1, 1, 0, 1, 0])
  assert.deepEqual(Array.from(index.bounds_B6.subarray(0, 6)), [0, 0, 0, 8, 8, 8])
  const segments = outlineSegments(index, holed, [0, 0, 1], [0, 0, 9.5], MIN_ZYX, MAX_ZYX)
  assert.deepEqual(coveredRange(segments), [0, 7])
})

test('nearest grid point within a reach', () => {
  const index = buildOutlineIndex(plane)
  const near = outlineNearestPoint(index, plane, [2, 3.2, 2.1], 1)
  assert.deepEqual([near.u, near.v], [2, 3])
  assert.ok(Math.abs(near.distance - Math.hypot(0.2, 0.1)) < 1e-9)
  const far = outlineNearestPoint(index, plane, [2, 3.2, 2.1], 0.2)
  assert.deepEqual([far.u, far.v, far.distance], [-1, -1, Infinity])
  const hole = { ...plane, valid_HW: new Uint8Array(plane.valid_HW) }
  hole.valid_HW[3 * 5 + 2] = 0
  assert.deepEqual(outlineNearestPoint(buildOutlineIndex(hole), hole, [2, 3.2, 2.1], 1).v, 4)
  assert.deepEqual(outlineNearestPoint(index, plane, [0, -1, 0], 1), { u: 0, v: 0, distance: 1 })
  assert.deepEqual(outlineNearestPoint(index, plane, [0, 0, 0], 0), { u: 0, v: 0, distance: 0 })
})

test('single-row and single-column grids have searchable points but no outline', () => {
  for (const [H, W] of [[1, 3], [3, 1], [1, 1]]) {
    const surface = gridSurface(H, W, (v, u) => [0, v, u])
    const index = buildOutlineIndex(surface)
    assert.deepEqual(outlineNearestPoint(index, surface, [0, H - 1, W - 1], 0),
      { u: W - 1, v: H - 1, distance: 0 })
    assert.equal(outlineSegments(index, surface, [1, 0, 0], [0, 0, 0], MIN_ZYX, MAX_ZYX).length, 0)
  }
})

test('nearest grid point follows the displaced sheet across group bounds', () => {
  const surface = gridSurface(70, 70, (v, u) => [32, v, u])
  surface.normals_zyx_HW3 = new Float32Array(70 * 70 * 3)
  for (let i = 0; i < 70 * 70; i++) surface.normals_zyx_HW3[i * 3] = -2
  const index = buildOutlineIndex(surface)
  assert.deepEqual(outlineNearestPoint(index, surface, [22, 65, 65], 2, 10), { u: 65, v: 65, distance: 0 })
  assert.deepEqual(outlineNearestPoint(index, surface, [42, 65, 65], 2, -10), { u: 65, v: 65, distance: 0 })
  assert.deepEqual(outlineNearestPoint(index, surface, [22, 65, 65], 2, 0), { u: -1, v: -1, distance: Infinity })
  surface.normals_zyx_HW3.fill(0)
  assert.deepEqual(outlineNearestPoint(index, surface, [32, 65, 65], 0, 10), { u: 65, v: 65, distance: 0 })
})

test('pane bounds cull offscreen blocks and retain blocks touching an edge', () => {
  const surface = gridSurface(25, 5, (v, u) => [u, v, u])
  const index = buildOutlineIndex(surface)
  const segments = outlineSegments(index, surface, [1, 0, 0], [2.5, 0, 0], [2.5, 9, 0], [2.5, 15, 4])
  assert.deepEqual(coveredRange(segments), [8, 16])
  const edge = outlineSegments(index, surface, [1, 0, 0], [2.5, 0, 0], [2.5, 8, 0], [2.5, 8, 4])
  assert.deepEqual(coveredRange(edge), [0, 16])
  assert.equal(outlineSegments(index, surface, [1, 0, 0], [2.5, 0, 0], [2.5, 30, 0], [2.5, 40, 4]).length, 0)
})

test('tilted cut z + y = 4.5 through z = x spans y 0.5..4', () => {
  const normal_zyx = [Math.SQRT1_2, Math.SQRT1_2, 0]
  const segments = outlineSegments(buildOutlineIndex(plane), plane,
    normal_zyx, [2.25, 2.25, 0], MIN_ZYX, MAX_ZYX)
  assert.ok(segments.length > 0)
  for (let i = 0; i < segments.length; i += 3) {
    assert.ok(Math.abs(segments[i] + segments[i + 1] - 4.5) < 1e-6)
    assert.equal(segments[i], segments[i + 2])
  }
  assert.deepEqual(coveredRange(segments), [0.5, 4])
})

test('tilted cut crosses block and group boundaries', () => {
  const surface = gridSurface(70, 70, (v, u) => [0, v, u])
  const index = buildOutlineIndex(surface)
  const segments = outlineSegments(index, surface,
    [0, Math.SQRT1_2, Math.SQRT1_2], [0, 34.25, 34.25], MIN_ZYX, MAX_ZYX)
  for (let i = 0; i < segments.length; i += 3) {
    assert.equal(segments[i], 0)
    assert.ok(Math.abs(segments[i + 1] + segments[i + 2] - 68.5) < 1e-6)
  }
  assert.deepEqual(coveredRange(segments), [0, 68.5])
  assert.equal(outlineSegments(index, surface,
    [0, Math.SQRT1_2, Math.SQRT1_2], [0, 80, 80], MIN_ZYX, MAX_ZYX).length, 0)
})

test('stride four cuts a tilted plane through the two coarse triangles', () => {
  const segments = outlineSegments(buildOutlineIndex(plane), plane,
    [Math.SQRT1_2, Math.SQRT1_2, 0], [2.25, 2.25, 0], MIN_ZYX, MAX_ZYX, 4)
  assert.deepEqual(Array.from(segments), [
    4, 0.5, 4, 2.25, 2.25, 2.25,
    2.25, 2.25, 2.25, 0.5, 4, 0.5,
  ])
})

test('coarse cuts include both edges of the last partial block', () => {
  const surface = gridSurface(12, 12, (v, u) => [u, v, u])
  const segments = outlineSegments(buildOutlineIndex(surface), surface,
    [1, 0, 0], [9.5, 0, 0], MIN_ZYX, MAX_ZYX, 4)
  assert.equal(segments.length, 6 * 6)
  for (let i = 0; i < segments.length; i += 3) {
    assert.equal(segments[i], 9.5)
    assert.equal(segments[i + 2], 9.5)
  }
  assert.deepEqual(coveredRange(segments), [0, 11])
})

test('coarse cuts preserve a hole between the sampled corners', () => {
  const surface = gridSurface(5, 5, (v, u) => [u, v, u])
  surface.valid_HW[2 * 5 + 2] = 0
  const index = buildOutlineIndex(surface)
  assert.deepEqual(Array.from(index.holes_B), [1])
  const segments = outlineSegments(index, surface,
    [1, 0, 0], [2.5, 0, 0], MIN_ZYX, MAX_ZYX, 4)
  assert.deepEqual(Array.from(segments), [
    2.5, 0, 2.5, 2.5, 0.5, 2.5,
    2.5, 0.5, 2.5, 2.5, 1, 2.5,
    2.5, 3, 2.5, 2.5, 3.5, 2.5,
    2.5, 3.5, 2.5, 2.5, 4, 2.5,
  ])
})

test('surface demand includes a ridge between sparse samples and covers displaced normals', () => {
  const ridge = gridSurface(129, 129, (v, u) => [v === 64 && u === 64 ? 300 : 10, v, u])
  const boxes = surfaceBounds(buildOutlineIndex(ridge), [0, 0], [128, 128], 0, [1, 1, 1])
  assert.equal(boxes.length, 64)
  assert.equal(Math.max(...boxes.map(b => b.max_zyx[0])), 301)
  assert.equal(Math.min(...boxes.map(b => b.min_zyx[0])), 9)
  const plane = gridSurface(2, 2, (v, u) => [300, v, u])
  plane.normals_zyx_HW3 = Float32Array.from([-1, 1, 0, -1, -1, 0, -1, 1, 0, -1, -1, 0], n => n * Math.SQRT1_2)
  assert.deepEqual(surfacePoint(plane, 0.5, 0.5, 20), [280, 0.5, 0.5])
  assert.deepEqual(surfaceBounds(buildOutlineIndex(plane), [0, 0], [1, 1], 20, [1, 1, 1]),
    [{ min_zyx: [279, -21, -21], max_zyx: [321, 22, 22], band: 0 }])
  plane.normals_zyx_HW3.fill(0)
  assert.deepEqual(surfacePoint(plane, 0.5, 0.5, 20), [300, 0.5, 0.5])
  assert.equal(surfacePoint(plane, -0.1, 0.5, 0), null)
  plane.valid_HW[3] = 0
  assert.equal(surfacePoint(plane, 0.5, 0.5, 0), null)
})

test('surface demand at the last grid point retains its bordering cells', () => {
  const surface = gridSurface(129, 129, (v, u) => [10, v, u])
  const index = buildOutlineIndex(surface)
  assert.deepEqual(surfaceBounds(index, [128, 128], [128, 128], 0, [1, 1, 1]),
    [{ min_zyx: [9, 119, 119], max_zyx: [11, 129, 129], band: 0 }])
})
