import {
  CONCURRENCY,
  UPLOAD_BYTES_PER_FRAME,
  UPLOAD_CALLS_PER_FRAME,
  WORKERS,
  REPLAN_MS,
  RETRY_MS,
  WINDOW_CHUNKS,
  PIN_COARSEST_CHUNKS,
  CT_ATLAS_BYTES,
  LAYER_ATLAS_BYTES,
  MEMORY_ATLAS_BYTES,
  EXTENSION_ORIGINS,
  MAX_LAYERS,
  LAYER_COLORS,
  LAYER_OPACITY,
  LAYER_THRESHOLD,
  LAYER_BRICK_SIZE,
  CT_CACHE_BYTES,
  LAYER_CACHE_BYTES,
  IMAGE_ATLAS_BYTES,
  IMAGE_CACHE_BYTES,
  IMAGE_BRICK_SIZE,
  SURFACE_MAX_POINTS,
  COLORS,
  hexRgb,
  AUTO_WINDOW_SAMPLES,
  AUTO_WINDOW_PERCENTILES,
  DTYPE_MAX,
} from './config.js'
import { createScene } from './scene.js'
import { createContext, layerCapacity } from './gpu/gl.js'
import { Residency } from './gpu/residency.js'
import { planChunks } from './views/plan.js'
import { SlicePane } from './views/slice-pane.js'
import { SurfacePane } from './views/surface-pane.js'
import { openVolume } from './volume/volume.js'
import { readVolumeMetadata } from './volume/ngff.js'
import { readTiffImage } from './volume/tiff.js'
import { readHash, writeHash } from './ui/hash.js'
import { openSurface } from './overlays/open-surface.js'
import { setupControls } from './ui/controls.js'
import { setupKeys } from './ui/keys.js'
import { setupStatus } from './ui/status.js'

const BACKGROUND_RGB = hexRgb(COLORS.background)

const scene = createScene()
let urls = { ct: null, surface: null, image: null, layers: [] }
let linkedLayers = []
let atlasMiB = null
let extensionUrls = []
const extensions = []
let ctPlan = null
let replanTimer = null
let lastReplan = -Infinity
let plannedFocus_zyx = [0, 0, 0]
let retryTimer = null
let windowPending = false
let loadId = 0
const queues = []
let firstSource = 0

function refresh() {
  controls.sync(urls)
  onChange()
}

function hashState() {
  return {
    ...urls, layerStyles: linkedLayers.map(layer => layer.style), extensions: extensionUrls, focus_zyx: scene.focus_zyx, axes_zyx: scene.axes_zyx, zoom: scene.zoom, uv: scene.surfaceView.center_uv,
    surfaceZoom: scene.surfaceView.zoom, slab: scene.surfaceView.slab, overlays: scene.overlays,
    offset: scene.surfaceView.offset, black: scene.window.black, white: scene.window.white, level: scene.levelCap, atlas: atlasMiB,
    imageOpacity: urls.image === null ? null : scene.surfaceView.imageOpacity,
  }
}

function onChange() {
  scene.dirty = true
  if (!scene.loading) writeHash(hashState())
  const wait = REPLAN_MS - (performance.now() - lastReplan)
  if (wait <= 0) replan()
  else if (replanTimer === null) replanTimer = setTimeout(replan, wait)
}

function replan() {
  clearTimeout(replanTimer)
  replanTimer = null
  if (scene.ct === null) return
  const now = performance.now()
  scene.velocity_zyx = scene.focus_zyx.map((v, a) => (v - plannedFocus_zyx[a]) * 1000 / (now - lastReplan))
  plannedFocus_zyx = scene.focus_zyx
  lastReplan = now
  const views = panes.map(pane => ({ boxes: pane.demand(), zoom: pane.worldZoom() }))
  const minDownsample = Math.max(...scene.ct.residency.levels[scene.levelCap].downsample_zyx)
  let retry = false
  for (const [index, source] of [scene.ct, ...scene.layers].entries()) {
    const { residency } = source
    const plan = planChunks({ levels: residency.levels, views, minDownsample, slotCapacity: residency.slotCount, windowChunks: WINDOW_CHUNKS, pinCoarsestChunks: PIN_COARSEST_CHUNKS })
    residency.ensure(plan)
    if (index === 0) ctPlan = plan
    panes.forEach((pane, i) => {
      if (index === 0) pane.displayLevel.ct = plan.displayLevels[i]
      else pane.displayLevel.layers[index - 1] = plan.displayLevels[i]
    })
    if (source.volume.stats().failed > 0 && residency.stats().missing > 0) retry = true
  }
  if (scene.image !== null) {
    const { volume, residency } = scene.image
    const plan = planChunks({ levels: residency.levels, views: [surfacePane.imageDemand()], minDownsample: 1, slotCapacity: residency.slotCount, windowChunks: WINDOW_CHUNKS, pinCoarsestChunks: PIN_COARSEST_CHUNKS })
    residency.ensure(plan)
    surfacePane.displayLevel.image = plan.displayLevels[0]
    if (volume.stats().failed > 0 && residency.stats().missing > 0) retry = true
  }
  clearTimeout(retryTimer)
  retryTimer = retry ? setTimeout(replan, RETRY_MS) : null
  scene.dirty = true
}

function frame() {
  requestAnimationFrame(frame)   // first, so a throwing frame does not stop the loop
  const width = Math.round(innerWidth * devicePixelRatio)
  const height = Math.round(innerHeight * devicePixelRatio)
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width
    canvas.height = height
    scene.dirty = true
  }
  let changed = false
  const budget = { bytes: UPLOAD_BYTES_PER_FRAME, calls: UPLOAD_CALLS_PER_FRAME }
  const sources = [scene.ct, ...scene.layers, scene.image].filter(source => source !== null)
  for (let i = 0; i < sources.length; i++) {
    if (sources[(firstSource + i) % sources.length].residency.tick(budget)) changed = true
  }
  firstSource = sources.length === 0 ? 0 : (firstSource + 1) % sources.length
  if (changed || scene.dirty) {
    scene.dirty = false
    gl.disable(gl.SCISSOR_TEST)
    gl.viewport(0, 0, width, height)
    gl.clearColor(BACKGROUND_RGB[0], BACKGROUND_RGB[1], BACKGROUND_RGB[2], 1)
    gl.clear(gl.COLOR_BUFFER_BIT)
    for (const pane of panes) drawPane(pane)
  }
  if (windowPending && panes.every(pane => scene.ct.residency.stats(pane.viewIndex).missing === 0) && autoWindow()) windowPending = false
}

function drawPane(pane) {
  pane.draw()
  for (const extension of extensions) {
    if (pane !== surfacePane) extension.draw(pane)
    else extension.drawSurface?.(pane)
  }
}

function layerStyle(volume, index) {
  return { visible: true, opacity: LAYER_OPACITY, color: [...LAYER_COLORS[index % LAYER_COLORS.length]],
    threshold: LAYER_THRESHOLD, mode: volume.meta.labels ? 'labels' : 'probability' }
}

async function openSource(id, metadata, cacheBytes, atlasBytes, brickSize = null) {
  const meta = await metadata
  if (id !== loadId) return null
  const volume = openVolume(meta, { cacheBytes, concurrency: CONCURRENCY, workers: WORKERS, queues, onError: status.error })
  try {
    return { volume, residency: new Residency(gl, volume, { atlasBytes, brickSize }) }
  } catch (error) {
    volume.close()
    throw error
  }
}

async function load(state) {
  state = { ...readHash('#'), ...state }
  if (state.layers.length > maxLayers) {
    status.error(`Requested ${state.layers.length} prediction layers; ${maxLayers} is the most here (MAX_LAYERS in config.js, and what this GPU can sample at once)`)
    return
  }
  const id = ++loadId
  scene.loading = true
  status.error('')
  clearTimeout(replanTimer)
  replanTimer = null
  lastReplan = -Infinity
  clearTimeout(retryTimer)
  retryTimer = null
  windowPending = false
  scene.dirty = true
  for (const pane of panes) pane.displayLevel = { ct: 0, layers: [], image: 0 }
  for (const source of [scene.ct, ...scene.layers, scene.image]) {
    if (source === null) continue
    source.volume.close()
    source.residency.close()
  }
  scene.ct = scene.surface = scene.image = null
  scene.layers = []
  scene.axes_zyx = state.axes_zyx ?? [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
  scene.surfaceView = { center_uv: state.uv ?? [0, 0], zoom: state.surfaceZoom ?? 1,
    offset: state.offset ?? 0, slab: state.slab ?? { front: 0, back: 0, step: 1, mode: 'mean' }, imageOpacity: state.imageOpacity ?? 1 }
  scene.overlays = state.overlays ?? { outline: true, crosshair: true }
  scene.outlineIndex = null
  ctPlan = null
  urls = { ct: state.ct, surface: state.surface, image: state.image, layers: state.layers }
  linkedLayers = state.layers.map((url, i) => ({ style: state.layerStyles[i] ?? null }))
  if (state.atlas !== null) atlasMiB = state.atlas
  const errors = []
  const settle = async (name, promise) => {
    try {
      return await promise
    } catch (error) {
      console.error(error)
      errors.push(`${name}: ${error.message}`)
      if (id === loadId) status.error(errors.join('; '))
      return null
    }
  }
  await Promise.all([
    state.ct === null ? null : settle('ct', (async () => {
      const ct = await openSource(id, readVolumeMetadata(state.ct, { world: null }), CT_CACHE_BYTES, atlasMiB === null ? CT_ATLAS_BYTES : atlasMiB * 2 ** 20)
      if (ct === null) return
      scene.ct = ct
      const { shape_zyx, offset_zyx, downsample_zyx } = ct.volume.levels[0]
      const [width, height] = slicePanes.xy.size()
      scene.focus_zyx = state.focus_zyx ?? shape_zyx.map((s, a) => offset_zyx[a] + (s * downsample_zyx[a] - 1) / 2)
      scene.zoom = state.zoom ?? Math.max(shape_zyx[2] * downsample_zyx[2] / width, shape_zyx[1] * downsample_zyx[1] / height)
      scene.levelCap = Math.min(state.level ?? 0, ct.volume.levels.length - 1)
      scene.window = { black: state.black ?? 0, white: state.white ?? DTYPE_MAX[ct.volume.dtype] }
      windowPending = ct.volume.dtype !== 'uint8' && state.black === null && state.white === null
      refresh()
      for (const [index, url] of state.layers.entries()) {
        if (id !== loadId) return
        await settle(`layer ${index + 1}`, (async () => {
          const layer = await openSource(id, readVolumeMetadata(url, { world: ct.volume.meta }),
            LAYER_CACHE_BYTES / state.layers.length, LAYER_ATLAS_BYTES / state.layers.length, LAYER_BRICK_SIZE)
          if (layer === null) return
          layer.style = state.layerStyles[index] ?? layerStyle(layer.volume, index)
          linkedLayers[index] = layer
          scene.layers.push(layer)
          for (const pane of panes) pane.displayLevel.layers.push(0)
          refresh()
        })())
      }
    })()),
    state.surface === null ? null : settle('surface', (async () => {
      const { surface, index } = await openSurface(state.surface, { maxPoints: SURFACE_MAX_POINTS })
      const maxSize = gl.getParameter(gl.MAX_TEXTURE_SIZE)
      if (surface.W > maxSize || surface.H > maxSize) throw new Error(`surface grid ${surface.W}x${surface.H} exceeds MAX_TEXTURE_SIZE ${maxSize}`)
      if (id !== loadId) return
      scene.surface = surface
      scene.outlineIndex = index
      scene.surfaceView.center_uv = state.uv ?? [(surface.W - 1) / 2, (surface.H - 1) / 2]
      refresh()
      if (state.image === null) return
      await settle('image', (async () => {
        const image = await openSource(id, readTiffImage(state.image), IMAGE_CACHE_BYTES, IMAGE_ATLAS_BYTES, IMAGE_BRICK_SIZE)
        if (image === null) return
        const [, height, width] = image.volume.levels[0].shape_zyx
        const [su, sv] = surface.scale_uv
        if (Math.abs(width * su - surface.W) >= 1 || Math.abs(height * sv - surface.H) >= 1) {
          image.volume.close()
          image.residency.close()
          throw new Error(`${height}×${width} pixels are not this surface's ${(surface.H / sv).toFixed(1)}×${(surface.W / su).toFixed(1)}`)
        }
        scene.image = image
        refresh()
      })())
    })()),
  ])
  if (id !== loadId) return
  clearTimeout(replanTimer)
  replan()
  scene.loading = false
  controls.sync(urls)
  writeHash(hashState())
}

function autoWindow() {
  const plan = ctPlan
  if (plan === null || plan.wanted.length === 0) return false
  const { volume } = scene.ct
  const finest = Math.min(...plan.wanted.map(c => c.level))
  const chunks = plan.wanted.filter(c => c.level === finest)
    .map(c => volume.cached(c.level, c.cz, c.cy, c.cx))
    .filter(data => data !== undefined && data !== null)
  const total = chunks.reduce((n, data) => n + data.length, 0)
  if (total === 0) return false
  const stride = Math.max(1, Math.ceil(total / AUTO_WINDOW_SAMPLES))
  const samples = new Float32Array(Math.ceil(total / stride))
  let count = 0
  let start = 0
  for (const data of chunks) {
    let i = start
    for (; i < data.length; i += stride) if (Number.isFinite(data[i])) samples[count++] = data[i]
    start = i - data.length
  }
  if (count === 0) return false
  const sorted = samples.subarray(0, count).sort()
  const black = sorted[Math.floor(AUTO_WINDOW_PERCENTILES[0] * (count - 1))]
  const white = sorted[Math.floor(AUTO_WINDOW_PERCENTILES[1] * (count - 1))]
  scene.window = { black, white: white > black ? white : black + 1 }
  refresh()
  return true
}

function stats() {
  let wanted = 0, missing = 0, pending = 0, inFlight = 0, residentBytes = 0, slotsUsed = 0, slotCount = 0
  for (const source of [scene.ct, ...scene.layers, scene.image]) {
    if (source === null) continue
    const r = source.residency.stats()
    wanted += r.wanted
    missing += r.missing
    pending += r.pending
    residentBytes += r.residentBytes
    slotsUsed += r.slotsUsed
    slotCount += r.slotCount
    inFlight += source.volume.stats().inFlight
  }
  const displayLevels = Object.values(slicePanes).map(pane => pane.displayLevel.ct)
  return { wanted, missing, pending, inFlight, residentMiB: residentBytes / 2 ** 20, slotsUsed, slotCount, displayLevels }
}

for (const [name, hex] of Object.entries(COLORS)) document.documentElement.style.setProperty(`--${name}`, hex)
const canvas = document.getElementById('canvas')
let gl
try {
  gl = createContext(canvas)
} catch (error) {
  document.getElementById('error').textContent = error.message
  throw error
}
const maxLayers = layerCapacity(gl, MAX_LAYERS)
const slicePanes = {
  xy: new SlicePane(document.getElementById('pane-xy'), 'z', scene, gl, onChange),
  xz: new SlicePane(document.getElementById('pane-xz'), 'y', scene, gl, onChange),
  yz: new SlicePane(document.getElementById('pane-yz'), 'x', scene, gl, onChange),
}
const surfacePane = new SurfacePane(document.getElementById('pane-surface'), scene, gl, onChange)
const panes = [slicePanes.xy, slicePanes.xz, slicePanes.yz, surfacePane]
panes.forEach((pane, i) => { pane.viewIndex = i })
const status = setupStatus({ scene, stats })
const controls = setupControls({ scene, onChange, autoWindow, load, extensions })
const keyboard = setupKeys({
  scene, onChange, refresh, panes: { ...slicePanes, surface: surfacePane }, grid: document.getElementById('grid'), extensions,
  showHelp: controls.showHelp, togglePanel: controls.togglePanel,
})

const resizeObserver = new ResizeObserver(() => {
  let changed = false
  for (const pane of panes) {
    const rect = pane.element.getBoundingClientRect()
    if (['left', 'top', 'width', 'height'].some(key => rect[key] !== pane.rect[key])) {
      pane.rect = rect
      changed = true
    }
  }
  if (changed) onChange()
})
resizeObserver.observe(document.getElementById('grid'))
for (const pane of panes) resizeObserver.observe(pane.element)

const viewer = {
  scene, gl, onChange, load, stats, error: status.error, panes: { ...slicePanes, surface: surfacePane }, maxLayers,
}
window.vesuvia = viewer

viewer.extend = create => {
  const extension = create(viewer)
  const taken = Object.keys(extension.keys ?? {}).find(key => keyboard.keys.has(key) || extensions.some(other => key in (other.keys ?? {})))
  if (taken !== undefined) {
    extension.dispose()
    throw new Error(`the key ${taken} is bound already`)
  }
  extensions.push(extension)
  document.getElementById('extensions').append(extension.panel)
  scene.dirty = true
  return () => {
    if (!extensions.includes(extension)) return
    extensions.splice(extensions.indexOf(extension), 1)
    extension.panel.remove()
    extension.dispose()
    scene.dirty = true
  }
}

viewer.addLayer = (volume, { index = scene.layers.length, style } = {}) => {
  if (scene.ct === null || scene.loading) throw new Error('a layer goes over an open CT; none is open')
  if (scene.layers.length === maxLayers) throw new Error(`${maxLayers} layers are open; the renderer blends no more here`)
  if (!Number.isInteger(index) || index < 0 || index > scene.layers.length) throw new Error(`layer index ${index} is outside 0..${scene.layers.length}`)
  if (style === undefined) style = layerStyle(volume, index)
  const layer = { volume, residency: new Residency(gl, volume, { atlasBytes: MEMORY_ATLAS_BYTES }), style }
  scene.layers.splice(index, 0, layer)
  for (const pane of panes) pane.displayLevel.layers.splice(index, 0, 0)
  refresh()
  return () => {
    const index = scene.layers.indexOf(layer)
    if (index === -1) return
    scene.layers.splice(index, 1)
    for (const pane of panes) pane.displayLevel.layers.splice(index, 1)
    layer.residency.close()
    refresh()
  }
}

async function allowed(href) {
  const { origin } = new URL(href)
  if (origin === location.origin || EXTENSION_ORIGINS.includes(origin)) return true
  const dialog = document.getElementById('extension-dialog')
  document.getElementById('extension-url').textContent = href
  dialog.returnValue = ''
  dialog.showModal()
  return new Promise(resolve => dialog.addEventListener('close', () => resolve(dialog.returnValue === 'load'), { once: true }))
}

async function extendFrom(moduleUrls) {
  const loaded = document.getElementById('extensions-loaded')
  const names = []
  const skipped = []
  const failures = []
  for (const url of moduleUrls) {
    try {
      const href = new URL(url, location.href).href
      if (!await allowed(href)) {
        skipped.push(`ext skipped: ${url}`)
        continue
      }
      const module = await import(href)
      if (typeof module.default !== 'function') throw new Error('no default export function')
      viewer.extend(module.default)
      names.push(new URL(url, location.href).pathname.split('/').at(-1))
    } catch (error) {
      console.error(error)
      failures.push(`ext failed: ${url}: ${error.message}`)
    }
  }
  const parts = [...(names.length > 0 ? [`ext: ${names.join(', ')}`] : []), ...skipped, ...failures]
  loaded.textContent = parts.join(', ')
  loaded.title = moduleUrls.join('\n')
  loaded.classList.toggle('error', failures.length > 0)
}

canvas.addEventListener('webglcontextlost', () => status.error('WebGL context lost; reload the page'))
requestAnimationFrame(frame)
try {
  window.addEventListener('hashchange', () => location.reload())   // a link pasted over this one stays in this document, and an ext= loads only at startup
  const initial = readHash()
  extensionUrls = initial.extensions
  extendFrom(extensionUrls)
  if (initial.ct !== null) load(initial)
} catch (error) {
  status.error(error.message)
}
