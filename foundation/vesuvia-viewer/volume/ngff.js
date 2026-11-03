// World coordinates are CT level-0 indices; offset_zyx is the block origin, so texel g is centred at offset + g*d + (d-1)/2. An NGFF translation names the centre of voxel zero; a level without one (the scroll pyramids on S3, mean-pooled) has its blocks start at the world origin.

const DOWNSAMPLE_TOLERANCE = 1e-6

const DTYPES_V2 = { '|u1': 'uint8', '<u2': 'uint16', '>u2': 'uint16', '<f4': 'float32', '>f4': 'float32' }
const DTYPES_V3 = new Set(['uint8', 'uint16', 'float32'])
const COMPRESSORS = new Set(['blosc', 'zstd', 'gzip', 'zlib', 'lz4'])
const BLOSC_SHUFFLE_V3 = { noshuffle: 0, shuffle: 1, bitshuffle: 2 }
const UINT_MAX = { uint8: 255, uint16: 65535 }

// zarr v2 lets fill_value be null (no fill, read as 0); both versions spell a
// float NaN as the string "NaN". Displayed as 0 either way.
function fillValue(value, dtype, url) {
  if (value === null) return 0
  if (Number.isFinite(value) && (dtype === 'float32' || (Number.isInteger(value) && value >= 0 && value <= UINT_MAX[dtype]))) return value
  if (value === 'NaN' && dtype === 'float32') return 0
  throw new Error(`${url}: fill_value ${JSON.stringify(value)} not supported for ${dtype}`)
}

async function fetchJsonDefault(url) {
  const response = await fetch(url)
  if (response.status === 404) return null
  if (!response.ok) throw new Error(`HTTP ${response.status} ${url}`)
  return response.json()
}

function nearestInteger(value, what) {
  const rounded = Math.round(value)
  if (!Number.isFinite(value) || Math.abs(value - rounded) > DOWNSAMPLE_TOLERANCE) throw new Error(`${what} ${value} is not an integer`)
  return rounded
}

function assertThreeAxes(axes, where) {
  if (!Array.isArray(axes)) throw new Error(`${where}: expected 3 axes, got ${JSON.stringify(axes)}`)
  if (axes.length !== 3) throw new Error(`${where}: expected 3 axes, got ${axes.length} [${axes.join(', ')}]`)
}

function vector(values, where, positive = false) {
  assertThreeAxes(values, where)
  if (values.some(v => !Number.isFinite(v) || (positive && v <= 0))) {
    throw new Error(`${where}: invalid values ${JSON.stringify(values)}`)
  }
  return values
}

function integers(values, where) {
  vector(values, where, true)
  if (!values.every(Number.isSafeInteger)) throw new Error(`${where}: ${JSON.stringify(values)} must be integers`)
  return values
}

function childUrl(url, path) {
  const child = new URL(url)
  child.pathname = `${child.pathname.replace(/\/+$/, '')}/${path}`
  return child.href
}

function parseArrayV2(doc, url) {
  if (doc.order !== 'C') throw new Error(`${url}: order ${JSON.stringify(doc.order)} not supported, only C`)
  if (doc.filters !== null && !(Array.isArray(doc.filters) && doc.filters.length === 0)) {
    throw new Error(`${url}: filters ${JSON.stringify(doc.filters)} not supported`)
  }
  const dtype = DTYPES_V2[doc.dtype]
  if (dtype === undefined) throw new Error(`${url}: dtype ${JSON.stringify(doc.dtype)} not supported`)
  integers(doc.shape, `${url}: shape`)
  integers(doc.chunks, `${url}: chunks`)
  if (doc.dimension_separator !== undefined && !['.', '/'].includes(doc.dimension_separator)) throw new Error(`${url}: dimension_separator ${doc.dimension_separator} not supported`)
  let codec = { name: 'raw' }
  if (doc.compressor !== null) {
    const { id, ...configuration } = doc.compressor
    if (!COMPRESSORS.has(id)) throw new Error(`${url}: compressor ${JSON.stringify(id)} not supported`)
    codec = { name: id, ...configuration }
  }
  return {
    shape_zyx: doc.shape, chunk_zyx: doc.chunks, dtype, littleEndian: doc.dtype[0] !== '>', codec,
    fill: fillValue(doc.fill_value, dtype, url), separator: doc.dimension_separator ?? '.', keyPrefix: '',
  }
}

function parseArrayV3(doc, url) {
  if (!DTYPES_V3.has(doc.data_type)) throw new Error(`${url}: data_type ${JSON.stringify(doc.data_type)} not supported`)
  integers(doc.shape, `${url}: shape`)
  integers(doc.chunk_grid.configuration.chunk_shape, `${url}: chunk_shape`)
  if (doc.dimension_names?.some((name, a) => name !== null && name !== 'zyx'[a])) throw new Error(`${url}: axes [${doc.dimension_names}] not supported; expected z,y,x`)
  if (doc.chunk_grid.name !== 'regular') throw new Error(`${url}: chunk_grid ${doc.chunk_grid.name} not supported`)
  if (doc.codecs.some(c => c.name.startsWith('sharding'))) throw new Error(`${url}: sharding not supported`)
  const [bytes, ...rest] = doc.codecs
  if (bytes.name !== 'bytes') throw new Error(`${url}: first codec must be bytes, got ${bytes.name}`)
  if (rest.length > 1) throw new Error(`${url}: codecs [${doc.codecs.map(c => c.name).join(', ')}] not supported`)
  let codec = { name: 'raw' }
  if (rest.length === 1) {
    const { name, configuration } = rest[0]
    if (!COMPRESSORS.has(name)) throw new Error(`${url}: codec ${JSON.stringify(name)} not supported`)
    codec = { name, ...configuration }
    if (name === 'blosc') codec.shuffle = BLOSC_SHUFFLE_V3[configuration.shuffle]
  }
  const { name: encoding, configuration: encodingConfig } = doc.chunk_key_encoding
  if (encoding !== 'default' && encoding !== 'v2') throw new Error(`${url}: chunk_key_encoding ${encoding} not supported`)
  const separator = encodingConfig?.separator ?? (encoding === 'default' ? '/' : '.')
  // uint8 stores may omit endian; it only matters for wider types.
  const endian = bytes.configuration?.endian ?? (doc.data_type === 'uint8' ? 'little' : undefined)
  if (endian !== 'little' && endian !== 'big') throw new Error(`${url}: bytes endian ${JSON.stringify(endian)} not supported`)
  if (!['.', '/'].includes(separator)) throw new Error(`${url}: separator ${separator} not supported`)
  return {
    shape_zyx: doc.shape, chunk_zyx: doc.chunk_grid.configuration.chunk_shape, dtype: doc.data_type,
    littleEndian: endian === 'little', codec, fill: fillValue(doc.fill_value, doc.data_type, url),
    separator, keyPrefix: encoding === 'default' ? `c${separator}` : '',
  }
}

export async function readVolumeMetadata(url, { world = null, fetchJson = fetchJsonDefault }) {
  const source = new URL(url, globalThis.location?.href)
  source.pathname = source.pathname.replace(/\/+$/, '')
  const base = source.href
  const v3 = await fetchJson(childUrl(base, 'zarr.json'))
  const version = v3 === null ? 2 : 3

  const multiscaleOf = attrs => (attrs?.ome?.multiscales ?? attrs?.multiscales)?.[0]
  const node = async (nodeUrl, readAttrs = true) => {
    if (version === 3) {
      const doc = nodeUrl === base ? v3 : await fetchJson(childUrl(nodeUrl, 'zarr.json'))
      if (doc === null) return null
      return { array: doc.node_type === 'array' ? parseArrayV3(doc, nodeUrl) : null, attrs: doc.attributes ?? {} }
    }
    const zattrs = readAttrs ? await fetchJson(childUrl(nodeUrl, '.zattrs')) : null
    if (multiscaleOf(zattrs) !== undefined) return { array: null, attrs: zattrs }
    const zarray = await fetchJson(childUrl(nodeUrl, '.zarray'))
    if (zarray === null && zattrs === null && await fetchJson(childUrl(nodeUrl, '.zgroup')) === null) return null
    return { array: zarray === null ? null : parseArrayV2(zarray, nodeUrl), attrs: zattrs ?? {} }
  }

  const root = await node(base)
  if (root === null) throw new Error(`no zarr array or group at ${base}`)
  const multiscale = multiscaleOf(root.attrs)

  const arrays = []
  if (root.array !== null) {
    arrays.push({ url: base, array: root.array, attrs: root.attrs, transforms: null })
  } else if (multiscale !== undefined) {
    if (multiscale.axes !== undefined) {
      const axes = multiscale.axes.map(axis => typeof axis === 'string' ? axis : axis.name)
      assertThreeAxes(axes, base)
      if (axes.join(',') !== 'z,y,x') throw new Error(`${base}: axes [${axes}] not supported; expected z,y,x`)
    }
    if (multiscale.coordinateTransformations !== undefined) throw new Error(`${base}: multiscale-level coordinateTransformations not supported`)
    arrays.push(...await Promise.all(multiscale.datasets.map(async ({ path, coordinateTransformations }) => {
      const levelUrl = childUrl(base, path)
      const level = await node(levelUrl, coordinateTransformations == null)
      if (level === null || level.array === null) throw new Error(`multiscale dataset ${path} is not an array at ${levelUrl}`)
      return { url: levelUrl, array: level.array, attrs: level.attrs, transforms: coordinateTransformations ?? null }
    })))
  } else {
    for (let index = 0; ; index++) {
      const level = await node(childUrl(base, String(index)))
      if (level === null || level.array === null) break
      arrays.push({ url: childUrl(base, String(index)), array: level.array, attrs: level.attrs, transforms: null })
    }
    if (arrays.length === 0) throw new Error(`${base}: no multiscales and no array "0"`)
  }

  // Registration without transforms: roi_origin_zyx + level on the
  // array, else on its parent group; else identity. Origin is in level-`level` voxels.
  const registered = a => a.roi_origin_zyx !== undefined && a.level !== undefined
  let attrs = root.attrs
  if (root.array !== null && !registered(attrs)) {
    const parent = new URL(base)
    parent.pathname = parent.pathname.replace(/\/?[^/]+\/?$/, '')
    attrs = (await node(parent.href))?.attrs ?? {}
  }

  if (arrays.length === 0) throw new Error(`${base}: empty multiscale datasets`)
  const scale_zyx = arrays[0].transforms?.find(t => t.type === 'scale')?.scale ?? [1, 1, 1]
  const translation_zyx = arrays[0].transforms?.find(t => t.type === 'translation')?.translation ?? [0, 0, 0]
  const worldScale_zyx = vector(world === null ? scale_zyx : world.scale_zyx, `${base}: world scale`, true)
  const worldTranslation_zyx = vector(world === null ? translation_zyx : world.translation_zyx, `${base}: world translation`)
  const { shape_zyx: shape0_zyx, dtype, fill } = arrays[0].array
  const levels = arrays.map(({ url: levelUrl, array, attrs: arrayAttrs, transforms }) => {
    if (array.dtype !== dtype) throw new Error(`${levelUrl}: dtype ${array.dtype} differs from level 0 (${dtype})`)
    let downsample_zyx, offset_zyx
    if (transforms !== null) {
      if (transforms.length < 1 || transforms.length > 2 || transforms[0].type !== 'scale' || (transforms.length === 2 && transforms[1].type !== 'translation')) {
        throw new Error(`${levelUrl}: coordinateTransformations ${JSON.stringify(transforms)} not supported`)
      }
      const scale = vector(transforms[0].scale, `${levelUrl}: scale`, true)
      const translation = transforms.length === 2 ? vector(transforms[1].translation, `${levelUrl}: translation`) : null
      downsample_zyx = scale.map((s, a) => nearestInteger(s / worldScale_zyx[a], `${levelUrl}: downsample axis ${a}`))
      offset_zyx = downsample_zyx.map((d, a) => translation === null
        ? (0 - worldTranslation_zyx[a]) / worldScale_zyx[a]
        : (translation[a] - worldTranslation_zyx[a]) / worldScale_zyx[a] - (d - 1) / 2)
    } else {
      const registration = registered(arrayAttrs) ? arrayAttrs : attrs
      const baseDownsample = registered(registration) ? 2 ** registration.level : 1
      const baseOffset_zyx = registered(registration) ? vector(registration.roi_origin_zyx, `${levelUrl}: roi_origin_zyx`).map(o => o * baseDownsample) : [0, 0, 0]
      downsample_zyx = shape0_zyx.map((s, a) => baseDownsample * Math.round(s / array.shape_zyx[a]))
      offset_zyx = baseOffset_zyx
    }
    if (downsample_zyx.some(d => !Number.isSafeInteger(d) || d < 1)) throw new Error(`${levelUrl}: downsample ${downsample_zyx} must be positive integers`)
    const { separator, keyPrefix } = array
    return {
      shape_zyx: array.shape_zyx, chunk_zyx: array.chunk_zyx, downsample_zyx, offset_zyx,
      chunkUrl: (cz, cy, cx) => childUrl(levelUrl, `${keyPrefix}${[cz, cy, cx].join(separator)}`),
      codec: array.codec, littleEndian: array.littleEndian,
    }
  })

  // NGFF marks a volume of integer ids with an `image-label` attribute beside its multiscales.
  const labels = (root.attrs.ome ?? root.attrs)['image-label'] !== undefined
  return { url: base, version, dtype, fill, labels, scale_zyx, translation_zyx, levels }
}
