import { ZOOM_STEP, STEP_RANGE, PAGE_STEPS } from '../config.js'
import { heldKeys } from './held.js'

// Letters and digits act by physical position, so a remapped layout keeps the
// hand shape; every other key acts by what it prints.
const binding = event => event.code.startsWith('Key') || event.code.startsWith('Digit') ? event.code : event.key

export function setupKeys({ scene, onChange, refresh, panes, grid, extensions, showHelp, togglePanel }) {
  let active = panes.xy
  let solo = null
  let pointer = null
  let imageOpacityShown = 1
  active.element.classList.add('active')

  // A hidden pane is not resized, so every rect is read back here; the panes
  // left with an empty rect draw nothing and want no chunks.
  const setSolo = pane => {
    solo = pane
    for (const other of Object.values(panes)) other.element.hidden = solo !== null && other !== solo
    grid.classList.toggle('solo', solo !== null)
    for (const other of Object.values(panes)) other.rect = other.element.getBoundingClientRect()
    onChange()
  }
  const focusPane = pane => {
    if (pane === active) return
    active.element.classList.remove('active')
    active = pane
    active.element.classList.add('active')
    if (solo !== null) setSolo(pane)
  }
  for (const pane of Object.values(panes)) {
    pane.element.addEventListener('pointerdown', () => focusPane(pane))
    pane.element.addEventListener('wheel', () => focusPane(pane), { passive: true })
    pane.element.addEventListener('pointermove', event => {
      pointer = { pane, x: event.clientX - pane.rect.left, y: event.clientY - pane.rect.top }
    })
    pane.element.addEventListener('pointerleave', () => { if (pointer !== null && pointer.pane === pane) pointer = null })
  }
  const zoomActive = factor => {
    const [x, y] = pointer !== null && pointer.pane === active ? [pointer.x, pointer.y] : active.center()
    active.zoomAt(x, y, factor)
  }
  const setStep = delta => {
    scene.stepSize = Math.min(STEP_RANGE[1], Math.max(STEP_RANGE[0], scene.stepSize + delta))
    refresh()
  }
  const held = heldKeys({
    step: direction => active.step(direction * scene.stepSize),
    zoom: zoomActive,
    turn: (right, down) => { if (active !== panes.surface) active.rotateBy(right, down) },
  })
  const hold = event => held.press(event.code, event.repeat)

  const actions = {
    KeyD: hold,
    KeyF: hold,
    KeyI: hold,
    KeyO: hold,
    KeyG: () => setStep(-1),
    KeyH: () => setStep(1),
    KeyR: () => {
      if (pointer === null) return
      focusPane(pointer.pane)
      const world_zyx = pointer.pane.pixelToWorld(pointer.x, pointer.y)
      if (world_zyx === null) return
      scene.focus_zyx = world_zyx
      onChange()
    },
    KeyV: () => {
      scene.overlays.crosshair = !scene.overlays.crosshair
      refresh()
    },
    KeyL: () => {
      const view = scene.surfaceView
      const imageShown = scene.image !== null && view.imageOpacity > 0
      const show = scene.layers.every(layer => !layer.style.visible) && !imageShown
      for (const layer of scene.layers) layer.style.visible = show
      if (imageShown) [imageOpacityShown, view.imageOpacity] = [view.imageOpacity, 0]
      else if (show && scene.image !== null) view.imageOpacity = imageOpacityShown
      refresh()
    },
    KeyZ: () => setSolo(solo === active ? null : active),
    Digit0: () => { scene.axes_zyx = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]; onChange() },
    Digit1: () => focusPane(panes.xy),
    Digit2: () => focusPane(panes.xz),
    Digit3: () => focusPane(panes.yz),
    Digit4: () => focusPane(panes.surface),
    ArrowLeft: hold,
    ArrowRight: hold,
    ArrowUp: hold,
    ArrowDown: hold,
    PageUp: () => active.step(-PAGE_STEPS),
    PageDown: () => active.step(PAGE_STEPS),
    '+': () => zoomActive(1 / ZOOM_STEP),
    '=': () => zoomActive(1 / ZOOM_STEP),
    '-': () => zoomActive(ZOOM_STEP),
    Tab: togglePanel,
    '?': showHelp,
    Escape: () => { scene.tool = scene.surfaceTool = null; scene.dirty = true },
    ' ': () => { scene.panKey = true; scene.dirty = true },
  }
  document.addEventListener('keydown', event => {
    const target = event.target
    if (event.altKey) return
    if (target instanceof HTMLInputElement || target instanceof HTMLSelectElement || target instanceof HTMLTextAreaElement) return
    if (document.querySelector('dialog[open]') !== null) return
    const key = (event.ctrlKey || event.metaKey ? 'Ctrl+' : '') + binding(event)
    const owner = [scene.tool, scene.surfaceTool, ...extensions].find(owner => owner?.keys !== undefined && key in owner.keys)
    const action = actions[key] ?? (event.repeat ? undefined : owner?.keys[key].run)
    if (action === undefined) return
    if (key === ' ' && scene.tool === null && scene.surfaceTool === null) return
    event.preventDefault()
    action(event)
  })
  document.addEventListener('keyup', event => {
    const key = binding(event)
    if (key === ' ') {
      scene.panKey = false
      scene.dirty = true
    }
    held.release(key)
  })
  window.addEventListener('blur', () => {
    scene.panKey = false
    scene.dirty = true
    held.stop()
  })

  return { keys: new Set(Object.keys(actions)) }
}
