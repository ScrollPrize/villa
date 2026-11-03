// Pane pixels are CSS pixels, origin top-left; the focus sits at the centre of the centre pixel, so that pixel samples focus_zyx exactly at zoom 1.
import {
  PREFETCH_SLICES,
  AHEAD_MS,
  ROTATE_RADIANS_PER_PX,
  TILT_DEMAND_TILES,
  OUTLINE_BLOCK,
  OUTLINE_CACHE_MARGIN_PX,
  OUTLINE_PLANE_EPSILON,
  OUTLINE_CELL_PX,
  COLORS,
  ZOOM_RANGE,
  wheelZoom,
  CLICK_SLOP_PX,
  hexRgb,
} from '../config.js'
import { orthonormalizeAxes, tilted, voxelStep } from '../scene.js'
import { LAYER_UNIFORMS, layerGlsl, bindLayer } from '../gpu/layer.glsl.js'
import { FULLSCREEN_VERTEX, startProgram, programReady, fullscreenTriangle, paneRect, uniformLocations } from '../gpu/gl.js'
import { Lines } from '../gpu/lines.js'
import { outlineSegments } from '../overlays/outline.js'

const AXES = {
  z: { name: 'XY', slice: 0, right: 2, down: 1 },
  y: { name: 'XZ', slice: 1, right: 2, down: 0 },
  x: { name: 'YZ', slice: 2, right: 1, down: 0 },
}
const OUTLINE_WIDTH_PX = 2
const CROSSHAIR_WIDTH_PX = 1
const CROSSHAIR_ALPHA = 0.5

const PANE_RGB = hexRgb(COLORS.pane)
const OUTLINE_RGBA = [...hexRgb(COLORS.outline), 1]
const CROSSHAIR_RGBA = [...hexRgb(COLORS.accent), CROSSHAIR_ALPHA]

const UNIFORMS = ['u_size', 'u_center', 'u_focus', 'u_right', 'u_down', 'u_zoom', 'u_ctLevel', 'u_ctMin', 'u_ctMax',
  'u_window', 'u_background']

function rotate(vector_zyx, axis_zyx, angle) {
  const c = Math.cos(angle), s = Math.sin(angle)
  const dot = vector_zyx.reduce((sum, v, a) => sum + v * axis_zyx[a], 0)
  return vector_zyx.map((v, a) => c * v + (1 - c) * dot * axis_zyx[a]
    + s * (axis_zyx[(a + 1) % 3] * vector_zyx[(a + 2) % 3] - axis_zyx[(a + 2) % 3] * vector_zyx[(a + 1) % 3]))
}

function fragmentSource(ct, layers) {
  const layerPart = layers.map((layer, i) => layer.residency.glsl(`layer${i}`) + layerGlsl(`layer${i}`)).join('\n')
  return `#version 300 es
precision highp float;
precision highp int;
${ct.residency.glsl('ct')}
uniform vec2 u_size;
uniform vec2 u_center;
uniform vec3 u_focus;
uniform vec3 u_right, u_down;
uniform float u_zoom;
uniform int u_ctLevel;
uniform vec3 u_ctMin, u_ctMax;
uniform vec2 u_window;
uniform vec3 u_background;
${layerPart}
in vec2 v_pixel;
out vec4 fragColor;

void main() {
  vec3 world_zyx = u_focus + (v_pixel.x - u_center.x) * u_zoom * u_right + (v_pixel.y - u_center.y) * u_zoom * u_down;
  float raw;
  if (any(lessThan(world_zyx, u_ctMin)) || any(greaterThanEqual(world_zyx, u_ctMax)) || !ct_sample(world_zyx, u_ctLevel, raw)) {
    fragColor = vec4(u_background, 1.0);
    return;
  }
  vec3 color = vec3(clamp((raw - u_window.x) / (u_window.y - u_window.x), 0.0, 1.0));
  ${layers.map((layer, i) => `color = layer${i}_blend(color, world_zyx);`).join('\n  ')}
  fragColor = vec4(color, 1.0);
}
`
}

export class SlicePane {
  constructor(element, axis, scene, gl, onChange) {
    this.element = element
    this.rect = element.getBoundingClientRect()
    this.axis = AXES[axis]
    this.scene = scene
    this.gl = gl
    this.onChange = onChange
    this.label = element.querySelector('span.label')
    this.labelText = null
    this.displayLevel = { ct: 0, layers: [] }
    this.program = null
    this.programFor = { ct: null, layers: [] }
    this.loc = null
    this.triangle = fullscreenTriangle(gl)
    this.lines = new Lines(gl)
    this.outlineLines = new Lines(gl)
    this.outline = { surface: null, normal_zyx: null, coordinate: null, stride: null, min_zyx: null, max_zyx: null, segments_S6: null }
    this.outlinePixels = { state: null, segments_S4: null }
    this.cursor = ''
    this.drag = null
    this.pointer = null

    element.addEventListener('contextmenu', event => event.preventDefault())
    element.addEventListener('pointerdown', event => {
      if (event.button !== 0 && event.button !== 2) return
      element.setPointerCapture(event.pointerId)
      if (event.button === 0 && scene.tool !== null && !scene.panKey) {
        this.drag = { id: event.pointerId, tool: scene.tool, world_zyx: this.pointerWorld(event) }
        this.drag.tool.down(this, this.drag.world_zyx, event)
        return
      }
      this.drag = { id: event.pointerId, x: event.clientX, y: event.clientY, focus_zyx: [...scene.focus_zyx], axes_zyx: scene.axes_zyx.map(axis_zyx => [...axis_zyx]), button: event.button, moved: false }
    })
    element.addEventListener('pointermove', event => {
      this.pointer = [event.clientX - this.rect.left, event.clientY - this.rect.top]
      if (scene.tool !== null) scene.dirty = true
      if (this.drag === null || event.pointerId !== this.drag.id) return
      if (this.drag.tool !== undefined) {
        for (const step of event.getCoalescedEvents?.() ?? [event]) {   // every sample of a fast stroke; a page served over plain HTTP has only the event
          this.drag.world_zyx = this.pointerWorld(step)
          this.drag.tool.move(this, this.drag.world_zyx, step)
        }
        return
      }
      const dx = event.clientX - this.drag.x
      const dy = event.clientY - this.drag.y
      if (!this.drag.moved && Math.hypot(dx, dy) < CLICK_SLOP_PX) return
      this.drag.moved = true
      const right_zyx = this.drag.axes_zyx[this.axis.right]
      const down_zyx = this.drag.axes_zyx[this.axis.down]
      if (this.drag.button === 2) {
        scene.axes_zyx = orthonormalizeAxes(this.drag.axes_zyx.map(axis_zyx =>
          rotate(rotate(axis_zyx, down_zyx, dx * ROTATE_RADIANS_PER_PX), right_zyx, dy * ROTATE_RADIANS_PER_PX)))
      } else {
        scene.focus_zyx = this.drag.focus_zyx.map((v, a) => v - scene.zoom * (dx * right_zyx[a] + dy * down_zyx[a]))
      }
      onChange()
    })
    element.addEventListener('pointerup', event => {
      if (this.drag === null || event.pointerId !== this.drag.id) return
      if (this.drag.tool !== undefined) {
        this.drag.tool.up(this, this.pointerWorld(event), event)
      } else if (!this.drag.moved && this.drag.button === 0) {
        scene.focus_zyx = this.pointerWorld(event)
        onChange()
      }
      this.drag = null
    })
    const interrupt = event => {
      if (this.drag === null || event.pointerId !== this.drag.id) return
      if (this.drag.tool !== undefined) this.drag.tool.up(this, this.drag.world_zyx, event)
      this.drag = null
    }
    element.addEventListener('pointerleave', () => {
      this.pointer = null
      if (scene.tool !== null) scene.dirty = true
    })
    element.addEventListener('pointercancel', interrupt)
    element.addEventListener('lostpointercapture', interrupt)
    element.addEventListener('wheel', event => {
      event.preventDefault()
      if (event.deltaY === 0) return
      const direction = Math.sign(event.deltaY)
      if (event.shiftKey) {
        this.step(direction)
      } else {
        const rect = this.rect
        this.zoomAt(event.clientX - rect.left, event.clientY - rect.top, wheelZoom(event.deltaY, event.deltaMode))
      }
    }, { passive: false })
  }

  size() {
    const rect = this.rect
    return [rect.width, rect.height]
  }

  center() {
    const [width, height] = this.size()
    return [Math.floor(width / 2) + 0.5, Math.floor(height / 2) + 0.5]
  }

  voxel(index) {
    const { ct } = this.scene
    return ct === null ? 1 : ct.residency.levels[this.displayLevel.ct].downsample_zyx[index]
  }

  pixelToWorld(px, py) {
    const { focus_zyx, zoom } = this.scene
    const [cx, cy] = this.center()
    const right_zyx = this.scene.axes_zyx[this.axis.right]
    const down_zyx = this.scene.axes_zyx[this.axis.down]
    return focus_zyx.map((v, a) => v + zoom * ((px - cx) * right_zyx[a] + (py - cy) * down_zyx[a]))
  }

  pointerWorld(event) {
    return this.pixelToWorld(event.clientX - this.rect.left, event.clientY - this.rect.top)
  }

  step(count) {
    const { scene } = this
    const normal_zyx = scene.axes_zyx[this.axis.slice]
    const voxel = voxelStep(normal_zyx, [0, 1, 2].map(a => this.voxel(a)))
    scene.focus_zyx = scene.focus_zyx.map((v, a) => v + count * voxel * normal_zyx[a])
    scene.stepDirection = Math.sign(count)
    this.onChange()
  }

  rotateBy(angleRight, angleDown) {
    const { scene } = this
    const right_zyx = scene.axes_zyx[this.axis.right]
    const down_zyx = scene.axes_zyx[this.axis.down]
    scene.axes_zyx = orthonormalizeAxes(scene.axes_zyx.map(axis_zyx =>
      rotate(rotate(axis_zyx, down_zyx, angleDown), right_zyx, angleRight)))
    this.onChange()
  }

  zoomAt(px, py, factor) {
    const { scene } = this
    const world_zyx = this.pixelToWorld(px, py)
    scene.zoom = Math.min(ZOOM_RANGE[1], Math.max(ZOOM_RANGE[0], scene.zoom * factor))
    const [cx, cy] = this.center()
    const right_zyx = scene.axes_zyx[this.axis.right], down_zyx = scene.axes_zyx[this.axis.down]
    scene.focus_zyx = world_zyx.map((v, a) => v - scene.zoom * ((px - cx) * right_zyx[a] + (py - cy) * down_zyx[a]))
    this.onChange()
  }

  worldZoom() {
    return this.scene.zoom
  }

  demand() {
    const { scene } = this
    if (scene.ct === null) return []
    const [width, height] = this.size()
    if (width === 0 || height === 0) return []
    const { focus_zyx, zoom, axes_zyx } = scene
    const normal_zyx = axes_zyx[this.axis.slice], right_zyx = axes_zyx[this.axis.right], down_zyx = axes_zyx[this.axis.down]
    const cx = Math.floor(width / 2) + 0.5, cy = Math.floor(height / 2) + 0.5
    const downsample_zyx = scene.ct.residency.levels[this.displayLevel.ct].downsample_zyx
    const voxel = voxelStep(normal_zyx, downsample_zyx)
    const shift = scene.stepDirection * PREFETCH_SLICES * voxel
    const coarsest = scene.ct.residency.levels.at(-1)
    const ahead_zyx = scene.velocity_zyx.map((v, a) => {
      const reach = coarsest.chunk_zyx[a] * coarsest.downsample_zyx[a]
      return Math.max(-reach, Math.min(reach, v * AHEAD_MS / 1000))
    })
    const tiles = tilted(axes_zyx) ? TILT_DEMAND_TILES : 1
    const boxes = []
    for (let y = 0; y < tiles; y++) for (let x = 0; x < tiles; x++) {
      const min_zyx = [Infinity, Infinity, Infinity], max_zyx = [-Infinity, -Infinity, -Infinity]
      for (let dy = 0; dy <= 1; dy++) for (let dx = 0; dx <= 1; dx++) {
        const px = ((x + dx) * width / tiles - cx) * zoom
        const py = ((y + dy) * height / tiles - cy) * zoom
        for (let a = 0; a < 3; a++) {
          const v = focus_zyx[a] + px * right_zyx[a] + py * down_zyx[a]
          min_zyx[a] = Math.min(min_zyx[a], v - downsample_zyx[a])
          max_zyx[a] = Math.max(max_zyx[a], v + downsample_zyx[a])
        }
      }
      boxes.push({ min_zyx, max_zyx, band: 0 },
        { min_zyx: min_zyx.map((v, a) => v + shift * normal_zyx[a]), max_zyx: max_zyx.map((v, a) => v + shift * normal_zyx[a]), band: 1 })
      if (ahead_zyx.some(v => v !== 0)) {
        boxes.push({ min_zyx: min_zyx.map((v, a) => v + Math.min(0, ahead_zyx[a])), max_zyx: max_zyx.map((v, a) => v + Math.max(0, ahead_zyx[a])), band: -1 })
      }
    }
    return boxes
  }

  project(segments_S6) {
    const { focus_zyx, zoom, axes_zyx } = this.scene
    const right_zyx = axes_zyx[this.axis.right], down_zyx = axes_zyx[this.axis.down]
    const [cx, cy] = this.center()
    const pixels_S4 = new Float32Array(segments_S6.length / 3 * 2)
    for (let p = 0; p < segments_S6.length / 3; p++) {
      let x = 0, y = 0
      for (let a = 0; a < 3; a++) {
        const delta = segments_S6[3 * p + a] - focus_zyx[a]
        x += delta * right_zyx[a]
        y += delta * down_zyx[a]
      }
      pixels_S4[2 * p] = cx + x / zoom
      pixels_S4[2 * p + 1] = cy + y / zoom
    }
    return pixels_S4
  }

  drawSegments(segments_S6, rgba, widthPx) {
    this.lines.draw(paneRect(this.gl.canvas, this.rect), this.project(segments_S6), rgba, widthPx)
  }

  drawPoints(points_zyx_N3, rgba, radiusPx) {
    this.lines.points(paneRect(this.gl.canvas, this.rect), this.project(points_zyx_N3), radiusPx, rgba)
  }

  draw() {
    const { gl, scene, axis } = this
    const rect = paneRect(gl.canvas, this.rect)
    if (rect.width === 0 || rect.height === 0) return
    const cursor = scene.tool !== null && !scene.panKey ? scene.tool.cursor : ''
    if (cursor !== this.cursor) {
      this.cursor = cursor
      this.element.style.cursor = cursor
    }
    gl.enable(gl.SCISSOR_TEST)
    gl.viewport(rect.x, rect.y, rect.width, rect.height)
    gl.scissor(rect.x, rect.y, rect.width, rect.height)
    const normal_zyx = scene.axes_zyx[axis.slice]
    const coordinate = normal_zyx.reduce((sum, n, a) => sum + n * scene.focus_zyx[a], 0)

    if (scene.ct === null) {
      gl.clearColor(PANE_RGB[0], PANE_RGB[1], PANE_RGB[2], 1)
      gl.clear(gl.COLOR_BUFFER_BIT)
      this.setLabel(axis.name)
      return
    }

    if (this.programFor.ct !== scene.ct || this.programFor.layers.length !== scene.layers.length || this.programFor.layers.some((layer, i) => layer !== scene.layers[i])) {
      if (this.program !== null) gl.deleteProgram(this.program)
      this.program = startProgram(gl, FULLSCREEN_VERTEX, fragmentSource(scene.ct, scene.layers))
      this.loc = null
      this.programFor = { ct: scene.ct, layers: [...scene.layers] }
    }
    if (this.loc === null) {
      if (!programReady(gl, this.program)) {
        scene.dirty = true
        return
      }
      this.loc = uniformLocations(gl, this.program, [...UNIFORMS, ...scene.layers.flatMap((layer, i) => LAYER_UNIFORMS.map(key => `layer${i}_${key}`))])
    }
    const { program, loc } = this
    const [width, height] = this.size()
    const [cx, cy] = this.center()
    const right_zyx = scene.axes_zyx[axis.right]
    const down_zyx = scene.axes_zyx[axis.down]
    const ct0 = scene.ct.residency.levels[0]

    gl.useProgram(program)
    gl.uniform2f(loc.u_size, width, height)
    gl.uniform2f(loc.u_center, cx, cy)
    gl.uniform3fv(loc.u_focus, scene.focus_zyx)
    gl.uniform3fv(loc.u_right, right_zyx)
    gl.uniform3fv(loc.u_down, down_zyx)
    gl.uniform1f(loc.u_zoom, scene.zoom)
    gl.uniform1i(loc.u_ctLevel, this.displayLevel.ct)
    gl.uniform3fv(loc.u_ctMin, ct0.offset_zyx)
    gl.uniform3fv(loc.u_ctMax, ct0.offset_zyx.map((o, a) => o + ct0.shape_zyx[a] * ct0.downsample_zyx[a]))
    gl.uniform2f(loc.u_window, scene.window.black, scene.window.white)
    gl.uniform3fv(loc.u_background, PANE_RGB)
    let unit = scene.ct.residency.bind(program, 'ct', 0)
    for (const [i, layer] of scene.layers.entries()) {
      unit = bindLayer(gl, program, loc, `layer${i}`, layer, this.displayLevel.layers[i], unit)
    }
    gl.bindVertexArray(this.triangle)
    gl.drawArrays(gl.TRIANGLES, 0, 3)
    gl.bindVertexArray(null)

    if (scene.overlays.outline && scene.surface !== null) {
      const outline = this.outline
      let stride = 1
      while (stride * 2 <= OUTLINE_BLOCK && stride * 2 * scene.surface.cellSpacing / scene.zoom <= OUTLINE_CELL_PX) stride *= 2
      const corners_zyx = [this.pixelToWorld(-OUTLINE_WIDTH_PX, -OUTLINE_WIDTH_PX),
        this.pixelToWorld(width + OUTLINE_WIDTH_PX, -OUTLINE_WIDTH_PX),
        this.pixelToWorld(-OUTLINE_WIDTH_PX, height + OUTLINE_WIDTH_PX),
        this.pixelToWorld(width + OUTLINE_WIDTH_PX, height + OUTLINE_WIDTH_PX)]
      const min_zyx = [0, 1, 2].map(a => Math.min(...corners_zyx.map(p => p[a])))
      const max_zyx = [0, 1, 2].map(a => Math.max(...corners_zyx.map(p => p[a])))
      if (outline.surface !== scene.surface || stride !== outline.stride || Math.abs(coordinate - outline.coordinate) > OUTLINE_PLANE_EPSILON
        || normal_zyx.some((v, a) => v !== outline.normal_zyx[a])
        || min_zyx.some((v, a) => v < outline.min_zyx[a])
        || max_zyx.some((v, a) => v > outline.max_zyx[a])) {
        const margin = OUTLINE_CACHE_MARGIN_PX * scene.zoom
        outline.surface = scene.surface
        outline.stride = stride
        outline.min_zyx = min_zyx.map(v => v - margin)
        outline.max_zyx = max_zyx.map(v => v + margin)
        outline.normal_zyx = [...normal_zyx]
        outline.coordinate = coordinate
        outline.segments_S6 = outlineSegments(scene.outlineIndex, scene.surface, normal_zyx, scene.focus_zyx, outline.min_zyx, outline.max_zyx, stride)
      }
      const state = [outline.segments_S6, ...scene.focus_zyx, ...right_zyx, ...down_zyx, scene.zoom, width, height]
      if (this.outlinePixels.state === null || state.some((value, i) => value !== this.outlinePixels.state[i])) {
        this.outlinePixels.state = state
        this.outlinePixels.segments_S4 = this.project(outline.segments_S6)
      }
      this.outlineLines.draw(rect, this.outlinePixels.segments_S4, OUTLINE_RGBA, OUTLINE_WIDTH_PX)
    }

    if (scene.overlays.crosshair) {
      this.lines.draw(rect, Float32Array.of(0, cy, width, cy, cx, 0, cx, height), CROSSHAIR_RGBA, CROSSHAIR_WIDTH_PX)
    }

    const loading = scene.ct.residency.stats(this.viewIndex).missing > 0 || scene.layers.some(layer => layer.residency.stats(this.viewIndex).missing > 0)
    this.setLabel(`${axis.name}  ${'zyx'[axis.slice]} ${coordinate.toFixed(1)}${tilted(scene.axes_zyx) ? ', tilted' : ''}${loading ? ', loading' : ''}`)
  }

  setLabel(text) {
    if (text === this.labelText) return
    this.labelText = text
    this.label.textContent = text
  }
}
