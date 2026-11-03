// u is the grid column, v the row; texel (x = u, y = v) of the RGB32F textures holds the ZYX point or normal.
import {
  COLORS,
  CLICK_SLOP_PX,
  ZOOM_RANGE,
  wheelZoom,
  SLAB_MAX_SAMPLES,
  MARKER_REACH_CELLS,
  DTYPE_MAX,
  hexRgb,
} from '../config.js'
import { LAYER_UNIFORMS, layerGlsl, bindLayer } from '../gpu/layer.glsl.js'
import { FULLSCREEN_VERTEX, startProgram, programReady, fullscreenTriangle, paneRect, uniformLocations } from '../gpu/gl.js'
import { Lines } from '../gpu/lines.js'
import { NORMAL_EPSILON, outlineNearestPoint, surfacePoint, surfaceBounds } from '../overlays/outline.js'

const MARKER_RADIUS_PX = 4
const SLAB_MODES = { mean: 0, max: 1, min: 2 }

const PANE_RGB = hexRgb(COLORS.pane)
const MARKER_RGBA = [...hexRgb(COLORS.accent), 1]

const UNIFORMS = ['u_size', 'u_center_uv', 'u_zoom', 'u_offset', 'u_slabStart', 'u_slabStep', 'u_slabCount', 'u_mode',
  'u_ctLevel', 'u_window', 'u_background', 'u_points', 'u_normals', 'u_valid']
const IMAGE_UNIFORMS = ['u_imageLevel', 'u_imageOpacity', 'u_imageScale', 'u_scale_uv']

function fragmentSource(ct, layers, image) {
  const layerPart = layers.map((layer, i) => layer.residency.glsl(`layer${i}`) + layerGlsl(`layer${i}`)).join('\n')
  const imagePart = image === null ? '' : `${image.residency.glsl('image')}
uniform int u_imageLevel;
uniform float u_imageOpacity;
uniform float u_imageScale;
uniform vec2 u_scale_uv;`
  return `#version 300 es
precision highp float;
precision highp int;
${ct.residency.glsl('ct')}
uniform vec2 u_size;
uniform vec2 u_center_uv;
uniform float u_zoom;
uniform float u_offset;
uniform float u_slabStart;
uniform float u_slabStep;
uniform int u_slabCount;
uniform int u_mode;               // 0 mean, 1 max, 2 min
uniform int u_ctLevel;
uniform vec2 u_window;
uniform vec3 u_background;
uniform highp sampler2D u_points;
uniform highp sampler2D u_normals;
uniform highp usampler2D u_valid;
${imagePart}
${layerPart}
in vec2 v_pixel;
out vec4 fragColor;

void main() {
  fragColor = vec4(u_background, 1.0);
  vec2 uv = u_center_uv + (v_pixel - u_size * 0.5) * u_zoom;
  ivec2 size = textureSize(u_points, 0);
  if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(size - 1)))) return;
  ivec2 g0 = ivec2(floor(uv));
  ivec2 g1 = min(g0 + 1, size - 1);
  ivec2 g01 = ivec2(g1.x, g0.y);
  ivec2 g10 = ivec2(g0.x, g1.y);
  if (texelFetch(u_valid, g0, 0).r == 0u || texelFetch(u_valid, g01, 0).r == 0u
      || texelFetch(u_valid, g10, 0).r == 0u || texelFetch(u_valid, g1, 0).r == 0u) return;
  vec2 f = uv - vec2(g0);
  vec4 w = vec4((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
  vec3 p_zyx = w[0] * texelFetch(u_points, g0, 0).xyz + w[1] * texelFetch(u_points, g01, 0).xyz
             + w[2] * texelFetch(u_points, g10, 0).xyz + w[3] * texelFetch(u_points, g1, 0).xyz;
  vec3 n_zyx = w[0] * texelFetch(u_normals, g0, 0).xyz + w[1] * texelFetch(u_normals, g01, 0).xyz
             + w[2] * texelFetch(u_normals, g10, 0).xyz + w[3] * texelFetch(u_normals, g1, 0).xyz;
  n_zyx /= max(length(n_zyx), ${NORMAL_EPSILON});

  float acc = u_mode == 1 ? -1e30 : (u_mode == 2 ? 1e30 : 0.0);
  int found = 0;
  for (int i = 0; i < u_slabCount; i++) {
    float value;
    if (!ct_sample(p_zyx + n_zyx * (u_offset + u_slabStart + float(i) * u_slabStep), u_ctLevel, value)) continue;
    acc = u_mode == 1 ? max(acc, value) : (u_mode == 2 ? min(acc, value) : acc + value);
    found++;
  }
  bool shown = found > 0;
  vec3 color = u_background;
  if (shown) {
    float value = u_mode == 0 ? acc / float(found) : acc;
    color = vec3(clamp((value - u_window[0]) / (u_window[1] - u_window[0]), 0.0, 1.0));
  }
  ${image === null ? '' : `float pixel;
  if (u_imageOpacity > 0.0 && image_sample(vec3(0.0, uv.y / u_scale_uv.y, uv.x / u_scale_uv.x), u_imageLevel, pixel)) {
    color = mix(color, vec3(clamp(pixel * u_imageScale, 0.0, 1.0)), u_imageOpacity);
    shown = true;
  }`}
  if (!shown) return;
  ${layers.map((layer, i) => `color = layer${i}_blend(color, p_zyx + n_zyx * u_offset);`).join('\n  ')}
  fragColor = vec4(color, 1.0);
}
`
}

export class SurfacePane {
  constructor(element, scene, gl, onChange) {
    this.element = element
    this.rect = element.getBoundingClientRect()
    this.scene = scene
    this.gl = gl
    this.onChange = onChange
    this.label = element.querySelector('span.label')
    this.labelText = null
    this.vao = fullscreenTriangle(gl)
    this.lines = new Lines(gl)
    this.displayLevel = { ct: 0, layers: [], image: 0 }
    this.program = null
    this.programFor = { ct: null, layers: [], image: null }
    this.loc = null
    this.textures = null
    this.textureSurface = null
    this.image = null
    this.drag = null
    this.pointer = null
    this.cursor = ''
    this.marker = { surface: null, offset: NaN, focus_zyx: [NaN, NaN, NaN], near: null }

    element.addEventListener('contextmenu', event => event.preventDefault())
    element.addEventListener('pointerdown', event => {
      if (event.button !== 0) return
      element.setPointerCapture(event.pointerId)
      if (scene.surfaceTool !== null && !scene.panKey) {
        this.drag = { id: event.pointerId, tool: scene.surfaceTool, uv: this.pointerUv(event) }
        this.drag.tool.down(this, this.drag.uv, event)
        return
      }
      this.drag = { id: event.pointerId, x: event.clientX, y: event.clientY, moved: false }
    })
    element.addEventListener('pointermove', event => {
      this.pointer = [event.clientX - this.rect.left, event.clientY - this.rect.top]
      if (scene.surfaceTool !== null) scene.dirty = true
      const drag = this.drag
      if (drag === null || event.pointerId !== drag.id) return
      if (drag.tool !== undefined) {
        for (const step of event.getCoalescedEvents?.() ?? [event]) {   // every sample of a fast stroke; a page served over plain HTTP has only the event
          drag.uv = this.pointerUv(step)
          drag.tool.move(this, drag.uv, step)
        }
        return
      }
      const dx = event.clientX - drag.x
      const dy = event.clientY - drag.y
      if (!drag.moved && Math.hypot(dx, dy) < CLICK_SLOP_PX) return
      drag.moved = true
      drag.x = event.clientX
      drag.y = event.clientY
      const view = scene.surfaceView
      view.center_uv[0] -= dx * view.zoom
      view.center_uv[1] -= dy * view.zoom
      this.onChange()
    })
    element.addEventListener('pointerup', event => {
      const drag = this.drag
      if (drag === null || event.pointerId !== drag.id) return
      this.drag = null
      element.releasePointerCapture(event.pointerId)
      if (drag.tool !== undefined) {
        drag.tool.up(this, this.pointerUv(event), event)
        return
      }
      if (drag.moved || scene.surface === null) return
      const rect = this.rect
      const world_zyx = this.pixelToWorld(event.clientX - rect.left, event.clientY - rect.top)
      if (world_zyx === null) return
      scene.focus_zyx = world_zyx
      this.onChange()
    })
    const interrupt = event => {
      if (this.drag === null || event.pointerId !== this.drag.id) return
      if (this.drag.tool !== undefined) this.drag.tool.up(this, this.drag.uv, event)
      this.drag = null
    }
    element.addEventListener('pointerleave', () => {
      this.pointer = null
      if (scene.surfaceTool !== null) scene.dirty = true
    })
    element.addEventListener('pointercancel', interrupt)
    element.addEventListener('lostpointercapture', interrupt)
    element.addEventListener('wheel', event => {
      event.preventDefault()
      const direction = Math.sign(event.deltaY)
      if (direction === 0) return
      if (event.shiftKey) {
        this.step(direction)
      } else {
        const rect = this.rect
        this.zoomAt(event.clientX - rect.left, event.clientY - rect.top, wheelZoom(event.deltaY, event.deltaMode))
      }
    }, { passive: false })
  }

  center() {
    return [Math.round(this.rect.width) / 2, Math.round(this.rect.height) / 2]
  }

  pixelToUv(px, py) {
    const { center_uv, zoom } = this.scene.surfaceView
    const [cx, cy] = this.center()
    return [center_uv[0] + (px - cx) * zoom, center_uv[1] + (py - cy) * zoom]
  }

  uvToPixel(u, v) {
    const { center_uv, zoom } = this.scene.surfaceView
    const [cx, cy] = this.center()
    return [(u - center_uv[0]) / zoom + cx, (v - center_uv[1]) / zoom + cy]
  }

  pointerUv(event) {
    return this.pixelToUv(event.clientX - this.rect.left, event.clientY - this.rect.top)
  }

  project(uv_P2) {
    const { center_uv, zoom } = this.scene.surfaceView
    const [cx, cy] = this.center()
    const pixels_P2 = new Float32Array(uv_P2.length)
    for (let i = 0; i < uv_P2.length; i += 2) {
      pixels_P2[i] = (uv_P2[i] - center_uv[0]) / zoom + cx
      pixels_P2[i + 1] = (uv_P2[i + 1] - center_uv[1]) / zoom + cy
    }
    return pixels_P2
  }

  drawSegments(segments_uv_S4, rgba, widthPx) {
    this.lines.draw(paneRect(this.gl.canvas, this.rect), this.project(segments_uv_S4), rgba, widthPx)
  }

  drawPoints(points_uv_N2, rgba, radiusPx) {
    this.lines.points(paneRect(this.gl.canvas, this.rect), this.project(points_uv_N2), radiusPx, rgba)
  }

  step(count) {
    this.scene.surfaceView.offset += count
    this.onChange()
  }

  zoomAt(px, py, factor) {
    const view = this.scene.surfaceView
    const [u, v] = this.pixelToUv(px, py)
    const [cx, cy] = this.center()
    view.zoom = Math.min(ZOOM_RANGE[1], Math.max(ZOOM_RANGE[0], view.zoom * factor))
    view.center_uv[0] = u - (px - cx) * view.zoom
    view.center_uv[1] = v - (py - cy) * view.zoom
    this.onChange()
  }

  pixelToWorld(px, py) {
    return this.scene.surface === null ? null : this.uvToWorld(...this.pixelToUv(px, py))
  }

  uvToWorld(u, v) {
    return surfacePoint(this.scene.surface, u, v, this.scene.surfaceView.offset)
  }

  worldZoom() {
    const { surface, surfaceView, zoom } = this.scene
    return surface === null ? zoom : surfaceView.zoom * surface.cellSpacing
  }

  demand() {
    const { scene } = this
    const surface = scene.surface
    if (surface === null || scene.ct === null) return []
    if (this.rect.width === 0 || this.rect.height === 0) return []
    const { H, W } = surface
    const { center_uv, zoom, offset, slab } = scene.surfaceView
    const [cx, cy] = this.center()
    const halfW = cx * zoom
    const halfH = cy * zoom
    const u0 = Math.max(0, Math.floor(center_uv[0] - halfW))
    const u1 = Math.min(W - 1, Math.ceil(center_uv[0] + halfW))
    const v0 = Math.max(0, Math.floor(center_uv[1] - halfH))
    const v1 = Math.min(H - 1, Math.ceil(center_uv[1] + halfH))
    if (u0 > u1 || v0 > v1) return []
    const front = slab.mode === 'single' ? 0 : slab.front
    const back = slab.mode === 'single' ? 0 : slab.back
    const reach = Math.max(Math.abs(offset - front), Math.abs(offset + back))
    const d_zyx = scene.ct.residency.levels[this.displayLevel.ct].downsample_zyx
    return surfaceBounds(scene.outlineIndex, [u0, v0], [u1, v1], reach, d_zyx)
  }

  imageDemand() {
    const { scene } = this
    const { center_uv, zoom, imageOpacity } = scene.surfaceView
    const [su, sv] = scene.surface.scale_uv
    const [cx, cy] = this.center()
    const box = (reach, band) => ({ band,
      min_zyx: [0, (center_uv[1] - reach * cy * zoom) / sv, (center_uv[0] - reach * cx * zoom) / su],
      max_zyx: [0, (center_uv[1] + reach * cy * zoom) / sv, (center_uv[0] + reach * cx * zoom) / su] })
    const shown = imageOpacity > 0 && this.rect.width > 0 && this.rect.height > 0
    return { boxes: shown ? [box(1, 0), box(2, 1)] : [], zoom: zoom / Math.max(su, sv) }
  }

  draw() {
    const { gl, scene } = this
    const rect = paneRect(gl.canvas, this.rect)
    if (rect.width === 0 || rect.height === 0) return
    const cursor = scene.surfaceTool !== null && !scene.panKey ? scene.surfaceTool.cursor : ''
    if (cursor !== this.cursor) {
      this.cursor = cursor
      this.element.style.cursor = cursor
    }
    gl.enable(gl.SCISSOR_TEST)
    gl.viewport(rect.x, rect.y, rect.width, rect.height)
    gl.scissor(rect.x, rect.y, rect.width, rect.height)

    const { center_uv, zoom, offset, slab, imageOpacity } = scene.surfaceView
    let slabStart = 0
    let slabStep = 1
    let slabCount = 1
    let mode = SLAB_MODES.mean
    if (slab.mode !== 'single') {
      mode = SLAB_MODES[slab.mode]
      slabStart = -slab.front
      slabStep = slab.step
      slabCount = Math.floor((slab.front + slab.back) / slab.step + 1e-6) + 1
      if (slabCount > SLAB_MAX_SAMPLES) {
        slabCount = SLAB_MAX_SAMPLES
        slabStep = (slab.front + slab.back) / (SLAB_MAX_SAMPLES - 1)
      }
    }
    const loading = scene.ct !== null && (scene.ct.residency.stats(this.viewIndex).missing > 0
      || scene.layers.some(layer => layer.residency.stats(this.viewIndex).missing > 0)
      || (scene.image !== null && scene.image.residency.stats().missing > 0))
    const slabText = slab.mode === 'single' ? 'single' : `${-slab.front}..+${slab.back} step ${+slabStep.toFixed(2)} ${slab.mode}`
    const text = `u ${center_uv[0].toFixed(1)} v ${center_uv[1].toFixed(1)}, offset ${offset}, slab ${slabText}${loading ? ', loading' : ''}`
    if (text !== this.labelText) {
      this.labelText = text
      this.label.textContent = text
    }

    const surface = scene.surface
    if (this.image !== null && (surface === null || scene.ct === null
      || this.image.width !== rect.width || this.image.height !== rect.height)) {
      gl.deleteFramebuffer(this.image.framebuffer)
      gl.deleteRenderbuffer(this.image.buffer)
      this.image = null
    }
    if (this.textureSurface !== surface && this.textures !== null) {
      for (const texture of Object.values(this.textures)) gl.deleteTexture(texture)
      this.textures = null
      this.textureSurface = null
    }
    if (scene.ct !== null && (this.programFor.ct !== scene.ct || this.programFor.image !== scene.image
      || this.programFor.layers.length !== scene.layers.length || scene.layers.some((layer, i) => layer !== this.programFor.layers[i]))) {
      if (this.program !== null) gl.deleteProgram(this.program)
      this.program = startProgram(gl, FULLSCREEN_VERTEX, fragmentSource(scene.ct, scene.layers, scene.image))
      this.loc = null
      this.programFor = { ct: scene.ct, layers: [...scene.layers], image: scene.image }
    }
    if (surface === null || scene.ct === null) {
      gl.clearColor(PANE_RGB[0], PANE_RGB[1], PANE_RGB[2], 1)
      gl.clear(gl.COLOR_BUFFER_BIT)
      return
    }
    if (this.loc === null) {
      if (!programReady(gl, this.program)) {
        scene.dirty = true
        return
      }
      this.loc = uniformLocations(gl, this.program, [...UNIFORMS, ...(scene.image === null ? [] : IMAGE_UNIFORMS),
        ...scene.layers.flatMap((layer, i) => LAYER_UNIFORMS.map(name => `layer${i}_${name}`))])
    }

    if (this.textureSurface !== surface) {
      const { H, W } = surface
      this.textures = {}
      for (const [name, internal, format, type, data] of [
        ['points', gl.RGB32F, gl.RGB, gl.FLOAT, surface.points_zyx_HW3],
        ['normals', gl.RGB32F, gl.RGB, gl.FLOAT, surface.normals_zyx_HW3],
        ['valid', gl.R8UI, gl.RED_INTEGER, gl.UNSIGNED_BYTE, surface.valid_HW],
      ]) {
        const texture = gl.createTexture()
        gl.bindTexture(gl.TEXTURE_2D, texture)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
        gl.texStorage2D(gl.TEXTURE_2D, 1, internal, W, H)
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, W, H, format, type, data)
        this.textures[name] = texture
      }
      this.textureSurface = surface
    }

    if (this.image === null) {
      const maxSize = gl.getParameter(gl.MAX_RENDERBUFFER_SIZE)
      if (rect.width > maxSize || rect.height > maxSize) throw new Error(`surface image ${rect.width}x${rect.height} exceeds MAX_RENDERBUFFER_SIZE ${maxSize}`)
      const buffer = gl.createRenderbuffer()
      gl.bindRenderbuffer(gl.RENDERBUFFER, buffer)
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.RGBA8, rect.width, rect.height)
      const framebuffer = gl.createFramebuffer()
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer)
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, buffer)
      const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER)
      gl.bindFramebuffer(gl.FRAMEBUFFER, null)
      gl.bindRenderbuffer(gl.RENDERBUFFER, null)
      if (status !== gl.FRAMEBUFFER_COMPLETE) {
        gl.deleteFramebuffer(framebuffer)
        gl.deleteRenderbuffer(buffer)
        throw new Error(`surface framebuffer ${rect.width}x${rect.height} incomplete: ${status}`)
      }
      this.image = { framebuffer, buffer, width: rect.width, height: rect.height, state: null }
    }
    const state = [surface, this.program, scene.ct.residency.revisionAt(this.displayLevel.ct),
      Math.round(this.rect.width), Math.round(this.rect.height), center_uv[0], center_uv[1], zoom, offset,
      slabStart, slabStep, slabCount, mode, this.displayLevel.ct,
      scene.window.black, scene.window.white]
    if (scene.image !== null) state.push(scene.image, scene.image.residency.revisionAt(this.displayLevel.image), this.displayLevel.image, imageOpacity)
    for (const [i, layer] of scene.layers.entries()) {
      const { style } = layer
      state.push(layer, layer.residency.revisionAt(this.displayLevel.layers[i]), this.displayLevel.layers[i],
        style.visible, style.opacity, style.mode, style.threshold, ...style.color)
    }
    const image = this.image
    if (image.state === null || state.length !== image.state.length || state.some((value, i) => value !== image.state[i])) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, image.framebuffer)
      gl.viewport(0, 0, rect.width, rect.height)
      gl.scissor(0, 0, rect.width, rect.height)

      const { program, loc } = this
      gl.useProgram(program)
      let unit = scene.ct.residency.bind(program, 'ct', 0)
      for (const [i, layer] of scene.layers.entries()) {
        unit = bindLayer(gl, program, loc, `layer${i}`, layer, this.displayLevel.layers[i], unit)
      }
      if (scene.image !== null) {
        unit = scene.image.residency.bind(program, 'image', unit)
        gl.uniform1i(loc.u_imageLevel, this.displayLevel.image)
        gl.uniform1f(loc.u_imageOpacity, imageOpacity)
        gl.uniform1f(loc.u_imageScale, 1 / DTYPE_MAX[scene.image.volume.dtype])
        gl.uniform2fv(loc.u_scale_uv, surface.scale_uv)
      }
      for (const [name, texture] of [['u_points', this.textures.points], ['u_normals', this.textures.normals], ['u_valid', this.textures.valid]]) {
        gl.activeTexture(gl.TEXTURE0 + unit)
        gl.bindTexture(gl.TEXTURE_2D, texture)
        gl.uniform1i(loc[name], unit)
        unit += 1
      }
      gl.uniform2f(loc.u_size, Math.round(this.rect.width), Math.round(this.rect.height))
      gl.uniform2f(loc.u_center_uv, center_uv[0], center_uv[1])
      gl.uniform1f(loc.u_zoom, zoom)
      gl.uniform1f(loc.u_offset, offset)
      gl.uniform1f(loc.u_slabStart, slabStart)
      gl.uniform1f(loc.u_slabStep, slabStep)
      gl.uniform1i(loc.u_slabCount, slabCount)
      gl.uniform1i(loc.u_mode, mode)
      gl.uniform1i(loc.u_ctLevel, this.displayLevel.ct)
      gl.uniform2f(loc.u_window, scene.window.black, scene.window.white)
      gl.uniform3fv(loc.u_background, PANE_RGB)
      gl.bindVertexArray(this.vao)
      gl.drawArrays(gl.TRIANGLES, 0, 3)
      gl.bindVertexArray(null)

      image.state = state
    }
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, image.framebuffer)
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null)
    gl.scissor(rect.x, rect.y, rect.width, rect.height)
    gl.clear(gl.COLOR_BUFFER_BIT) // a blit alone may not initialize a discarded default drawing buffer
    gl.blitFramebuffer(0, 0, rect.width, rect.height, rect.x, rect.y, rect.x + rect.width, rect.y + rect.height,
      gl.COLOR_BUFFER_BIT, gl.NEAREST)
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null)
    gl.viewport(rect.x, rect.y, rect.width, rect.height)

    const focus_zyx = scene.focus_zyx
    const marker = this.marker
    if (marker.surface !== surface || marker.offset !== offset || focus_zyx.some((value, a) => value !== marker.focus_zyx[a])) {
      marker.surface = surface
      marker.offset = offset
      marker.focus_zyx = [...focus_zyx]
      marker.near = outlineNearestPoint(scene.outlineIndex, surface, focus_zyx, MARKER_REACH_CELLS * surface.cellSpacing, offset)
    }
    if (marker.near.u < 0) return
    this.lines.points(rect, Float32Array.from(this.uvToPixel(marker.near.u, marker.near.v)), MARKER_RADIUS_PX, MARKER_RGBA)
  }
}
