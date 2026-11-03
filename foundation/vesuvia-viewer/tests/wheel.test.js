import { test } from 'node:test'
import assert from 'node:assert/strict'
import { ZOOM_STEP, WHEEL_NOTCH_PX, WHEEL_LINE_PX, wheelZoom } from '../config.js'

test('a notch split into many events zooms as much as one notch event', () => {
  let factor = 1
  for (let i = 0; i < 8; i++) factor *= wheelZoom(WHEEL_NOTCH_PX / 8, 0)
  assert.ok(Math.abs(factor - ZOOM_STEP) < 1e-12)
  assert.equal(wheelZoom(WHEEL_NOTCH_PX, 0), ZOOM_STEP)
  assert.equal(wheelZoom(-WHEEL_NOTCH_PX, 0), 1 / ZOOM_STEP)
})

test('an event zooms at most one ZOOM_STEP, in lines and pages too', () => {
  assert.equal(wheelZoom(10 * WHEEL_NOTCH_PX, 0), ZOOM_STEP)
  assert.equal(wheelZoom(-WHEEL_NOTCH_PX / WHEEL_LINE_PX, 1), 1 / ZOOM_STEP)
  assert.equal(wheelZoom(-30, 1), 1 / ZOOM_STEP)
  assert.equal(wheelZoom(1, 2), ZOOM_STEP)
})

test('turning back undoes the zoom', () => {
  assert.ok(Math.abs(wheelZoom(37, 0) * wheelZoom(-37, 0) - 1) < 1e-12)
})
