import { STATUS_MS, DTYPE_MAX, hexRgb } from '../config.js'
import { element } from './element.js'

const FLOAT_STEP = 0.001

const rgbHex = rgb => `#${rgb.map(c => Math.round(c * 255).toString(16).padStart(2, '0')).join('')}`

export function setupControls({ scene, onChange, autoWindow, load, extensions }) {
  const panel = element('panel')
  const openDialog = element('open-dialog')
  const helpDialog = element('help-dialog')
  const openInputs = { ct: element('open-ct'), surface: element('open-surface'), image: element('open-image'), layers: element('open-layer') }

  const bind = (id, apply) => {
    const input = element(id)
    input.addEventListener('input', () => {
      if (!input.checkValidity()) return
      apply(input)
      onChange()
    })
    return input
  }
  const inputs = {
    black: bind('ct-black', input => {
      scene.window.black = Number(input.value)
      if (scene.window.white <= scene.window.black) scene.window.white = scene.window.black + Number(input.step)
    }),
    white: bind('ct-white', input => {
      scene.window.white = Number(input.value)
      if (scene.window.black >= scene.window.white) scene.window.black = scene.window.white - Number(input.step)
    }),
    offset: bind('surface-offset', input => { scene.surfaceView.offset = Number(input.value) }),
    front: bind('surface-front', input => { scene.surfaceView.slab.front = Number(input.value) }),
    back: bind('surface-back', input => { scene.surfaceView.slab.back = Number(input.value) }),
    step: bind('surface-step', input => { scene.surfaceView.slab.step = Number(input.value) }),
    slabMode: bind('surface-mode', input => { scene.surfaceView.slab.mode = input.value }),
    outline: bind('surface-outline', input => { scene.overlays.outline = input.checked }),
    imageOpacity: bind('surface-image-opacity', input => { scene.surfaceView.imageOpacity = Number(input.value) }),
    crosshair: bind('view-crosshair', input => { scene.overlays.crosshair = input.checked }),
    stepSize: bind('view-step', input => { scene.stepSize = Number(input.value) }),
    level: bind('quality-level', input => { scene.levelCap = Number(input.value) }),
  }
  const readouts = { black: element('ct-black-value'), white: element('ct-white-value') }
  for (const [name, readout] of Object.entries(readouts)) {
    inputs[name].addEventListener('input', () => { readout.value = inputs[name].value })
    readout.addEventListener('change', () => {
      inputs[name].value = readout.value
      inputs[name].dispatchEvent(new Event('input'))
    })
  }

  element('ct-auto').addEventListener('click', autoWindow)
  element('open-button').addEventListener('click', () => openDialog.showModal())
  element('open-cancel').addEventListener('click', () => openDialog.close())
  element('empty-open').addEventListener('click', () => openDialog.showModal())
  const showHelp = () => {
    const row = (tag, ...texts) => {
      const tr = document.createElement('tr')
      tr.append(...texts.map(text => Object.assign(document.createElement(tag), { textContent: text, colSpan: 3 - texts.length })))
      return tr
    }
    const owners = [...extensions.map(extension => [extension.panel.querySelector('h2')?.textContent ?? 'Extension', extension.keys]),
      ...[scene.tool, scene.surfaceTool].filter(tool => tool !== null).map(tool => [tool.status ?? 'Tool', tool.keys])]
    element('help-extensions').replaceChildren(...owners.filter(([, keys]) => keys !== undefined).flatMap(([heading, keys]) => [
      row('th', heading),
      ...Object.entries(keys).map(([key, { help }]) => {
        const tr = row('td', '', help)
        tr.firstChild.append(Object.assign(document.createElement('kbd'), { textContent: key.replace(/(Key|Digit)/, '').toLowerCase() }))
        return tr
      })]))
    helpDialog.showModal()
  }
  element('help-button').addEventListener('click', showHelp)
  panel.addEventListener('click', event => {
    if (event.target instanceof HTMLHeadingElement && event.target.parentElement instanceof HTMLElement) event.target.parentElement.classList.toggle('folded')
  })
  element('help-close').addEventListener('click', () => helpDialog.close())
  const togglePanel = () => {
    panel.hidden = !panel.hidden
    onChange()
  }
  element('panel-button').addEventListener('click', togglePanel)
  element('open-form').addEventListener('submit', () => {
    const value = input => input.value.trim() === '' ? null : input.value.trim()
    load({
      ct: value(openInputs.ct), surface: value(openInputs.surface), image: value(openInputs.image), layers: openInputs.layers.value.split('\n').map(url => url.trim()).filter(Boolean),
    })
  })

  let lastOffset = scene.surfaceView.offset
  setInterval(() => {
    const offset = scene.surfaceView.offset
    if (offset === lastOffset) return
    lastOffset = offset
    if (inputs.offset.value === '' || Number(inputs.offset.value) !== offset) inputs.offset.value = offset
  }, STATUS_MS)

  let layerInputs = []
  return {
    showHelp,
    togglePanel,
    sync(urls) {
      element('layers-section').hidden = scene.layers.length === 0
      element('surface-section').hidden = scene.surface === null
      element('surface-image-row').hidden = scene.image === null
      for (const [name, input] of Object.entries(openInputs)) input.value = name === 'layers' ? urls.layers.join('\n') : urls[name] ?? ''
      if (scene.ct !== null) {
        const { dtype, levels } = scene.ct.volume
        for (const input of [inputs.black, inputs.white]) {
          input.min = dtype === 'float32' ? '' : 0
          input.max = dtype === 'float32' ? '' : DTYPE_MAX[dtype]
          input.step = dtype === 'float32' ? FLOAT_STEP : 1
        }
        inputs.level.replaceChildren(...levels.map((level, i) => new Option(`level ${i} (1/${level.downsample_zyx[0]})`, i)))
        inputs.level.value = scene.levelCap
      }
      inputs.black.value = scene.window.black
      inputs.white.value = scene.window.white
      if (layerInputs.length !== scene.layers.length || layerInputs.some((row, i) => row.layer !== scene.layers[i])) {
        element('layers').replaceChildren()
        layerInputs = scene.layers.map((layer, i) => {
          const section = document.createElement('div')
          section.innerHTML = `<div class="layer"><input type="checkbox" title="Visible"><button type="button"></button><input type="range" min="0" max="1" step="0.01" title="Opacity"></div>
            <div hidden>
              <label class="row"><span>Colour</span><input type="color"></label>
              <label class="row"><span>Threshold</span><input type="range" min="0" max="0.99" step="0.01"><output></output></label>
              <label class="row"><span>Mode</span><select><option value="probability">probability</option><option value="labels">labels</option></select></label>
            </div>`
          const name = section.querySelector('button')
          name.textContent = new URL(layer.volume.url, location.href).pathname.split('/').filter(Boolean).at(-1)
          name.title = layer.volume.url
          name.addEventListener('click', () => { section.lastElementChild.hidden = !section.lastElementChild.hidden })
          const [visible, opacity, color, threshold, mode] = section.querySelectorAll('input, select')
          opacity.id = `layer-${i}-opacity`
          color.id = `layer-${i}-color`
          threshold.id = `layer-${i}-threshold`
          mode.id = `layer-${i}-mode`
          visible.id = `layer-${i}-visible`
          section.addEventListener('input', () => {
            layer.style.opacity = Number(opacity.value)
            layer.style.color = hexRgb(color.value)
            layer.style.threshold = Number(threshold.value)
            layer.style.mode = mode.value
            layer.style.visible = visible.checked
            threshold.nextElementSibling.value = threshold.value
            color.disabled = threshold.disabled = mode.value === 'labels'
            onChange()
          })
          element('layers').append(section)
          return { layer, opacity, color, threshold, mode, visible }
        })
      }
      for (const row of layerInputs) {
        const { style } = row.layer
        row.opacity.value = style.opacity
        row.color.value = rgbHex(style.color)
        row.threshold.value = style.threshold
        row.mode.value = style.mode
        row.visible.checked = style.visible
        row.threshold.nextElementSibling.value = row.threshold.value
        row.color.disabled = row.threshold.disabled = style.mode === 'labels'
      }
      inputs.offset.value = scene.surfaceView.offset
      inputs.front.value = scene.surfaceView.slab.front
      inputs.back.value = scene.surfaceView.slab.back
      inputs.step.value = scene.surfaceView.slab.step
      inputs.slabMode.value = scene.surfaceView.slab.mode
      inputs.outline.checked = scene.overlays.outline
      inputs.imageOpacity.value = scene.surfaceView.imageOpacity
      inputs.crosshair.checked = scene.overlays.crosshair
      inputs.stepSize.value = scene.stepSize
      for (const [name, readout] of Object.entries(readouts)) readout.value = inputs[name].value
    },
  }
}
