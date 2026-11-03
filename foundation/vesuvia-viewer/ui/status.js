import { STATUS_MS } from '../config.js'
import { element } from './element.js'

export function setupStatus({ scene, stats }) {
  const status = element('status')
  const hint = element('hint')
  const errorText = element('error')
  const atlasBar = element('atlas-bar').firstElementChild
  const atlasText = element('atlas-text')

  setInterval(() => {
    const tool = scene.tool ?? scene.surfaceTool
    const hintText = scene.ct === null ? (scene.loading ? 'Loading…' : '')
      : tool !== null ? tool.status ?? 'Tool' : ''
    if (hint.textContent !== hintText) hint.textContent = hintText
    document.body.classList.toggle('no-ct', scene.ct === null && !scene.loading)
    if (scene.ct === null) {
      status.textContent = ''
      return
    }
    const s = stats()
    const text = `${s.residentMiB.toFixed(0)} MiB resident, ${s.inFlight} in flight, level ${Math.min(...s.displayLevels)}`
    if (status.textContent !== text) status.textContent = text
    const atlasPercent = 100 * s.slotsUsed / s.slotCount
    atlasBar.style.width = `${atlasPercent.toFixed(1)}%`
    const percent = `${atlasPercent.toFixed(0)}%`
    const title = `${s.slotsUsed} of ${s.slotCount} atlas slots`
    if (atlasText.textContent !== percent) atlasText.textContent = percent
    if (atlasText.title !== title) atlasText.title = title
  }, STATUS_MS)

  return {
    error(message) {
      errorText.textContent = message
    },
  }
}
