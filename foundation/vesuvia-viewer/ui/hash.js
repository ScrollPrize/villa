import { orthonormalizeAxes, tilted } from '../scene.js'
import { HASH_DEBOUNCE_MS } from '../config.js'

const URL_KEYS = ['ct', 'surface', 'image']
const DECIMALS = { focus: 2, uv: 2, offset: 2, black: 3, white: 3, level: 0, atlas: 0, imageOpacity: 2 }
const fixed = (value, decimals) => String(Number(value.toFixed(decimals)))

const encode = text => text.replace(/[%&#+\s]/g, c => encodeURIComponent(c))

export function readHash(hash = location.hash) {
  const params = new URLSearchParams(hash.replace(/^#/, ''))
  const numbers = (key, count, text = params.get(key), shown = text) => {
    if (text === null) return null
    const parts = text.split(',')
    const values = parts.map(Number)
    if (parts.some(v => v.trim() === '') || values.length !== count || values.some(v => !Number.isFinite(v))) {
      throw new Error(`hash ${key}=${JSON.stringify(shown)}: expected ${count} number(s)`)
    }
    return count === 1 ? values[0] : values
  }
  const axes = numbers('axes', 9)
  const state = {
    ct: params.get('ct'),
    surface: params.get('surface'),
    image: params.get('image'),
    layers: params.getAll('layer'),
    extensions: params.getAll('ext'),
    focus_zyx: numbers('focus', 3),
    axes_zyx: axes === null ? null : orthonormalizeAxes([axes.slice(0, 3), axes.slice(3, 6), axes.slice(6, 9)]),
    zoom: numbers('zoom', 1),
    surfaceZoom: numbers('surfaceZoom', 1),
    slab: null,
    layerStyles: [],
    overlays: null,
    uv: numbers('uv', 2),
    offset: numbers('offset', 1),
    black: numbers('black', 1),
    white: numbers('white', 1),
    level: numbers('level', 1),
    atlas: numbers('atlas', 1),
    imageOpacity: numbers('imageOpacity', 1),
  }
  for (const key of ['zoom', 'surfaceZoom']) {
    if (state[key] !== null && state[key] <= 0) throw new Error(`hash ${key}=${state[key]}: must be positive`)
  }
  if (params.has('slab')) {
    const parts = params.get('slab').split(',')
    const [front, back, step] = numbers('slab', 3, parts.slice(0, 3).join(','), params.get('slab'))
    const mode = parts[3]
    if (parts.length !== 4 || front < 0 || back < 0 || step <= 0 || !['single', 'mean', 'max', 'min'].includes(mode)) {
      throw new Error(`hash slab=${params.get('slab')}: expected front,back,positive step,single|mean|max|min`)
    }
    state.slab = { front, back, step, mode }
  }
  state.layerStyles = params.getAll('style').map(text => {
    if (text === '') return null
    const [mode, ...parts] = text.split(',')
    const [visible, opacity, threshold, ...color] = numbers('style', 6, parts.join(','), text)
    if (!['probability', 'labels'].includes(mode) || ![0, 1].includes(visible)
      || [opacity, threshold, ...color].some(v => v < 0 || v > 1)) {
      throw new Error(`hash style=${text}: expected mode,visibility,opacity,threshold,r,g,b in 0..1`)
    }
    return { mode, visible: visible === 1, opacity, threshold, color }
  })
  if (state.layerStyles.length !== 0 && state.layerStyles.length !== state.layers.length) {
    throw new Error(`hash style count ${state.layerStyles.length}: expected ${state.layers.length} layer styles`)
  }
  const overlays = numbers('overlays', 2)
  if (overlays !== null) {
    if (overlays.some(v => ![0, 1].includes(v))) throw new Error(`hash overlays=${overlays}: expected outline,crosshair as 0 or 1`)
    state.overlays = { outline: !!overlays[0], crosshair: !!overlays[1] }
  }
  if (state.level !== null && (!Number.isSafeInteger(state.level) || state.level < 0)) throw new Error(`hash level=${state.level}: must be a non-negative integer`)
  if (state.atlas !== null && (!Number.isSafeInteger(state.atlas) || state.atlas < 1)) throw new Error(`hash atlas=${state.atlas}: must be a positive integer, MiB`)
  if (state.imageOpacity !== null && !(state.imageOpacity >= 0 && state.imageOpacity <= 1)) throw new Error(`hash imageOpacity=${state.imageOpacity}: must be in 0..1`)
  if (state.black !== null && state.white !== null && state.black >= state.white) throw new Error(`hash window ${state.black}..${state.white}: white must exceed black`)
  return state
}

export function formatHash(state) {
  const parts = []
  for (const key of URL_KEYS) if (state[key] !== null) parts.push(`${key}=${encode(state[key])}`)
  for (const url of state.layers) parts.push(`layer=${encode(url)}`)
  for (const style of state.layerStyles) {
    parts.push(`style=${style === null ? '' : [style.mode, Number(style.visible), style.opacity, style.threshold, ...style.color].join(',')}`)
  }
  for (const url of state.extensions) parts.push(`ext=${encode(url)}`)
  if (state.focus_zyx !== null) parts.push(`focus=${state.focus_zyx.map(v => fixed(v, DECIMALS.focus)).join(',')}`)
  if (state.axes_zyx !== null && tilted(state.axes_zyx)) {
    parts.push(`axes=${state.axes_zyx.flat().map(v => fixed(v, 4)).join(',')}`)
  }
  for (const key of ['zoom', 'surfaceZoom']) if (state[key] !== null) parts.push(`${key}=${Number(state[key].toPrecision(4))}`)
  if (state.uv !== null) parts.push(`uv=${state.uv.map(v => fixed(v, DECIMALS.uv)).join(',')}`)
  if (state.slab !== null) {
    const { front, back, step, mode } = state.slab
    parts.push(`slab=${front},${back},${step},${mode}`)
  }
  if (state.overlays !== null) parts.push(`overlays=${['outline', 'crosshair'].map(key => Number(state.overlays[key])).join(',')}`)
  for (const key of ['offset', 'black', 'white', 'level', 'atlas', 'imageOpacity']) {
    if (state[key] !== null) parts.push(`${key}=${fixed(state[key], DECIMALS[key])}`)
  }
  return parts.join('&')
}

let timer = null
let written = null

export function writeHash(state) {
  clearTimeout(timer)
  timer = setTimeout(() => {
    const next = formatHash(state)
    if (next === written) return
    written = next
    history.replaceState(null, '', `#${next}`)
  }, HASH_DEBOUNCE_MS)
}
