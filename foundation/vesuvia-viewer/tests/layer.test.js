import { test } from 'node:test'
import assert from 'node:assert/strict'
import { labelColor } from '../gpu/layer.glsl.js'

test('label 1 takes the golden-ratio hue 0.618, a blue, at the saturation the shader uses', () => {
  const [r, g, b] = labelColor(1)
  assert.ok(Math.abs(r - 0.3) < 1e-3, `red ${r}`)
  assert.ok(Math.abs(g - 0.504) < 1e-3, `green ${g}`)
  assert.equal(b, 1)
})
