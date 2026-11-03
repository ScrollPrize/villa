// The JPEG an eighth the size that S3 keeps beside a prediction TIFF (downsampled/<name>-ds8.jpg) has a pixel the mean of 8×8, and its 2×2 means down to one tile are the coarser levels.
import { fetchFile, fetchRange } from './mirrors.js'

const TAG = {
  width: 256, height: 257, bits: 258, compression: 259, photometric: 262, fillOrder: 266, samples: 277, predictor: 317,
  tileWidth: 322, tileHeight: 323, tileOffsets: 324, tileByteCounts: 325, sampleFormat: 339,
}
const ABSENT = { bits: 1, compression: 1, fillOrder: 1, samples: 1, predictor: 1, sampleFormat: 1 }
const CODECS = { 1: { name: 'raw' }, 8: { name: 'zlib' }, 32946: { name: 'zlib' } }   // by Compression: none, deflate, deflate's older code
const DS8 = 8
const STRIP_ROWS = 512     // rows of the JPEG drawn and read back at once: a canvas of all of it is 4 bytes a pixel

export function ds8Url(url) {
  const jpeg = new URL(url)
  const match = /^(.*\/)([^/]+)\.tiff?$/.exec(jpeg.pathname)
  if (match === null) return null
  jpeg.pathname = `${match[1]}downsampled/${match[2]}-ds8.jpg`
  jpeg.hash = ''
  return jpeg.href
}

export async function readTiffLayout(at, url) {
  const view = bytes => new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const header = await at(0, 8)
  const order = String.fromCharCode(header[0], header[1])
  if (order !== 'II' && order !== 'MM') throw new Error(`${url} is not a TIFF`)
  const little = order === 'II'
  if (view(header).getUint16(2, little) !== 42) throw new Error(`${url} is not a classic TIFF; a BigTIFF is not read here`)
  const ifd = view(header).getUint32(4, little)
  const count = view(await at(ifd, 2)).getUint16(0, little)
  const entries = view(await at(ifd + 2, count * 12 + 4))
  if (entries.getUint32(count * 12, little) !== 0) throw new Error(`${url} holds more than one page`)

  const read = async name => {
    for (let i = 0; i < count; i++) {
      if (entries.getUint16(12 * i, little) !== TAG[name]) continue
      const type = entries.getUint16(12 * i + 2, little)
      if (type !== 3 && type !== 4) throw new Error(`${url}: tag ${TAG[name]} (${name}) has type ${type}, not SHORT or LONG`)
      const size = type === 3 ? 2 : 4
      const n = entries.getUint32(12 * i + 4, little)
      const bytes = n * size <= 4 ? new Uint8Array(entries.buffer, entries.byteOffset + 12 * i + 8, 4) : await at(entries.getUint32(12 * i + 8, little), n * size)
      const values = view(bytes)
      return Array.from({ length: n }, (_, k) => size === 2 ? values.getUint16(2 * k, little) : values.getUint32(4 * k, little))
    }
    return undefined
  }
  const tags = Object.fromEntries(await Promise.all(Object.keys(TAG).map(async name => [name, await read(name)])))
  const value = name => {
    const values = tags[name] ?? (name in ABSENT ? [ABSENT[name]] : undefined)
    if (values === undefined) throw new Error(`${url} has no ${name} tag (${TAG[name]})`)
    return values[0]
  }

  if (value('samples') !== 1 || value('bits') !== 8 || value('sampleFormat') !== 1) {
    throw new Error(`${url} holds ${value('samples')} samples of ${value('bits')} bits in format ${value('sampleFormat')}; one uint8 is read here`)
  }
  if (value('photometric') !== 1) throw new Error(`${url} has photometric ${value('photometric')}; grey from black at 0 (1) is read here`)
  if (value('predictor') !== 1) throw new Error(`${url} has predictor ${value('predictor')}; none (1) is read here`)
  if (value('fillOrder') !== 1) throw new Error(`${url} has fill order ${value('fillOrder')}; bytes as stored (1) are read here`)
  if (tags.tileWidth === undefined || tags.tileHeight === undefined) throw new Error(`${url} is in strips; tiles are read here`)
  const codec = CODECS[value('compression')]
  if (codec === undefined) throw new Error(`${url} has compression ${value('compression')}; none (1) and deflate (8, 32946) are read here`)
  const [height, width, tileHeight, tileWidth] = ['height', 'width', 'tileHeight', 'tileWidth'].map(value)
  const tiles = Math.ceil(height / tileHeight) * Math.ceil(width / tileWidth)
  const offsets = tags.tileOffsets
  const counts = tags.tileByteCounts
  if (offsets?.length !== tiles || counts?.length !== tiles) throw new Error(`${url} has ${offsets?.length} tile offsets and ${counts?.length} byte counts for ${tiles} tiles`)
  if (counts.includes(0)) throw new Error(`${url} has a tile of no bytes`)
  return { height, width, tileHeight, tileWidth, offsets, counts, codec }
}

export function pooled(pixels_HW, [height, width]) {
  const shape_HW = [Math.ceil(height / 2), Math.ceil(width / 2)]
  const out_HW = new Uint8Array(shape_HW[0] * shape_HW[1])
  for (let y = 0; y < shape_HW[0]; y++) {
    const top = 2 * y * width
    const bottom = Math.min(2 * y + 1, height - 1) * width
    for (let x = 0; x < shape_HW[1]; x++) {
      const left = 2 * x
      const right = Math.min(2 * x + 1, width - 1)
      out_HW[y * shape_HW[1] + x] = (pixels_HW[top + left] + pixels_HW[top + right] + pixels_HW[bottom + left] + pixels_HW[bottom + right] + 2) >> 2
    }
  }
  return { pixels_HW: out_HW, shape_HW }
}

export async function readTiffImage(url) {
  const source = new URL(url, globalThis.location?.href).href
  const { height, width, tileHeight, tileWidth, offsets, counts, codec } = await readTiffLayout((offset, length) => fetchRange(source, offset, length), url)
  const across = Math.ceil(width / tileWidth)
  const levels = [{
    shape_zyx: [1, height, width], chunk_zyx: [1, tileHeight, tileWidth], downsample_zyx: [1, 1, 1], offset_zyx: [0, 0, 0],
    chunkUrl: () => source, chunkRange: (cz, cy, cx) => [offsets[cy * across + cx], counts[cy * across + cx]], codec, littleEndian: true,
  }]

  const tilesAlone = { url, dtype: 'uint8', fill: 0, levels }
  const jpeg = ds8Url(source)
  if (jpeg === null) return tilesAlone
  let bytes
  try {
    bytes = await fetchFile(jpeg)
  } catch (error) {
    if (error.status === 404) return tilesAlone
    throw error
  }
  const bitmap = await createImageBitmap(new Blob([bytes]), { colorSpaceConversion: 'none' })
  let shape_HW = [Math.ceil(height / DS8), Math.ceil(width / DS8)]
  const [jpegHeight, jpegWidth] = [bitmap.height, bitmap.width]
  if (jpegHeight !== shape_HW[0] || jpegWidth !== shape_HW[1]) {
    bitmap.close()
    throw new Error(`${jpeg} is ${jpegHeight}×${jpegWidth} pixels, not the TIFF's ${height}×${width} / ${DS8}`)
  }
  let pixels_HW = new Uint8Array(jpegHeight * jpegWidth)
  const context = new OffscreenCanvas(jpegWidth, STRIP_ROWS).getContext('2d', { willReadFrequently: true })
  for (let y = 0; y < jpegHeight; y += STRIP_ROWS) {
    const rows = Math.min(STRIP_ROWS, jpegHeight - y)
    context.drawImage(bitmap, 0, y, jpegWidth, rows, 0, 0, jpegWidth, rows)
    const rgba = context.getImageData(0, 0, jpegWidth, rows).data
    for (let i = 0; i < rows * jpegWidth; i++) pixels_HW[y * jpegWidth + i] = rgba[4 * i]
  }
  bitmap.close()

  for (let downsample = DS8; ; downsample *= 2) {
    const [h, w] = shape_HW
    const held_HW = pixels_HW
    levels.push({
      shape_zyx: [1, h, w], chunk_zyx: [1, tileHeight, tileWidth], downsample_zyx: [1, downsample, downsample], offset_zyx: [0, 0, 0],
      heldChunk(cz, cy, cx) {
        const chunk = new Uint8Array(tileHeight * tileWidth)
        const x0 = cx * tileWidth
        for (let y = 0; y < tileHeight && cy * tileHeight + y < h; y++) {
          const row = (cy * tileHeight + y) * w
          chunk.set(held_HW.subarray(row + x0, row + Math.min(w, x0 + tileWidth)), y * tileWidth)
        }
        return chunk
      },
    })
    if (h <= tileHeight && w <= tileWidth) break
    const next = pooled(pixels_HW, shape_HW)
    pixels_HW = next.pixels_HW
    shape_HW = next.shape_HW
  }
  return { url, dtype: 'uint8', fill: 0, levels }
}
