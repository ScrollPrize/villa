// Another index goes in this module's own URL: `?index=<url>`.

const DEFAULT_INDEX = 'https://raw.githubusercontent.com/ScrollPrize/villa/main/scrollprize.org/src/data/atlasDataAccess.json'
const CT_TYPE = 'ome-zarr'
const SURFACE_TYPE = 'tifxyz-transformed'
const LAYER_TYPES = ['surface-prediction-zarr', 'ink-detection-3d-zarr']

async function fetchJson(url, signal) {
  const response = await fetch(url, { signal })
  if (!response.ok) throw new Error(`HTTP ${response.status} ${url}`)
  return response.json()
}

function asset(record, rewrites) {
  const origins = record.origins.flatMap(origin =>
    origin.access_roots.map(root => `${root.url.replace(/\/$/, '')}/${origin.path.replace(/^\//, '')}`))
  const url = origins
    .map(origin => {
      const rewrite = rewrites.find(entry => origin.startsWith(entry.from))
      return rewrite === undefined ? origin : rewrite.to + origin.slice(rewrite.from.length)
    })
    .find(origin => origin.startsWith('https:'))
  if (url === undefined) throw new Error(`catalog ${record.type}: no public HTTPS origin in ${origins.join(', ')}`)
  return { name: url.replace(/\/$/, '').split('/').at(-1), url }
}

export function catalogEntries(metadata, access) {
  if (metadata.samples === null || typeof metadata.samples !== 'object') {
    throw new Error(`catalog metadata: samples ${JSON.stringify(metadata.samples)} is not an object`)
  }
  const { rewrites } = access.dataAccess
  const entries = []
  for (const [sample, { volumes, segments }] of Object.entries(metadata.samples)) {
    for (const [scan, volume] of Object.entries(volumes)) {
      const record = volume.data.find(item => item.type === CT_TYPE)
      if (record === undefined) continue
      const { name, url } = asset(record, rewrites)
      const layers = volume.data.filter(item => LAYER_TYPES.includes(item.type)).map(item => asset(item, rewrites))
      const surfaces = Object.values(segments).flatMap(segment => segment.data
        .filter(item => item.type === SURFACE_TYPE && item.parameters.target_volume === scan)
        .map(item => asset(item, rewrites)))
      entries.push({ sample, name, label: `${volume.properties.pixel_size_um} µm, ${scan}`, ct: url, layers, surfaces })
    }
  }
  return entries
}

export default function createCatalog(viewer) {
  const indexUrl = new URL(new URL(import.meta.url).searchParams.get('index') ?? DEFAULT_INDEX, location.href).href
  const reading = new AbortController()
  let entries = []
  let entry = null

  const panel = document.createElement('section')
  panel.innerHTML = `<h2>Catalog</h2>
    <label class="row"><span>Volume</span><select class="wide"></select></label>
    <label class="row"><span>Surface</span><select class="wide"></select></label>
    <div class="checks"></div>
    <div class="buttons"><button type="button">Open</button></div>
    <p class="note"></p>`
  const [volumeSelect, surfaceSelect] = panel.querySelectorAll('select')
  const checks = panel.querySelector('.checks')
  const openButton = panel.querySelector('button')
  const note = panel.querySelector('p')
  volumeSelect.id = 'catalog-volume'
  surfaceSelect.id = 'catalog-surface'
  openButton.id = 'catalog-open'
  note.id = 'catalog-note'

  function describe(message = '') {
    const boxes = [...checks.querySelectorAll('input')]
    const chosen = boxes.filter(box => box.checked).length
    for (const box of boxes) box.disabled = !box.checked && chosen === viewer.maxLayers
    openButton.disabled = entry === null
    if (message !== '') note.textContent = message
    else if (entry !== null) note.textContent = `${entries.length} volumes, ${entry.surfaces.length} surfaces`
  }

  function choose() {
    entry = entries[Number(volumeSelect.value)]
    surfaceSelect.replaceChildren(new Option('none', ''), ...entry.surfaces.map((surface, index) => {
      const option = new Option(surface.name, index)
      option.title = surface.url
      return option
    }))
    checks.replaceChildren(...entry.layers.map((layer, index) => {
      const row = document.createElement('label')
      row.className = 'check'
      row.title = layer.url
      row.innerHTML = `<input type="checkbox" id="catalog-layer-${index}" ${index < viewer.maxLayers ? 'checked' : ''}><span></span>`
      row.lastElementChild.textContent = layer.name
      return row
    }))
    describe()
  }

  volumeSelect.addEventListener('change', choose)
  checks.addEventListener('change', () => describe())

  openButton.addEventListener('click', async () => {
    const boxes = [...checks.querySelectorAll('input')]
    const layers = entry.layers.filter((layer, index) => boxes[index].checked).map(layer => layer.url)
    const surface = surfaceSelect.value === '' ? null : entry.surfaces[Number(surfaceSelect.value)].url
    openButton.blur()   // so Space pans instead of pressing the button again
    describe(`Opening ${entry.name}…`)
    await viewer.load({ ct: entry.ct, surface, layers })
    describe()
  })

  async function read() {
    describe(`Reading ${indexUrl}…`)
    try {
      const access = await fetchJson(indexUrl, reading.signal)
      const metadata = await fetchJson(new URL(access.metadataUrl, indexUrl).href, reading.signal)
      entries = catalogEntries(metadata, access)
      if (entries.length === 0) throw new Error(`${indexUrl}: no volume in this catalog has a CT`)
      const groups = []
      for (const [index, volume] of entries.entries()) {
        if (groups.length === 0 || groups.at(-1).label !== volume.sample) {
          const group = document.createElement('optgroup')
          group.label = volume.sample
          groups.push(group)
        }
        const option = new Option(volume.label, index)
        option.title = volume.ct
        groups.at(-1).append(option)
      }
      volumeSelect.replaceChildren(...groups)
      choose()
    } catch (error) {
      if (reading.signal.aborted) return
      console.error(error)
      viewer.error(`catalog: ${error.message}`)
      describe(error.message)
    }
  }
  read()

  return {
    panel,
    draw() {},
    dispose() {
      reading.abort()
    },
  }
}
