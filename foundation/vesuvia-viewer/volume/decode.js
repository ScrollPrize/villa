import { Blosc, Zstd, GZip, Zlib, LZ4 } from '../vendor/codecs.js'
import { DTYPE_ARRAY } from '../config.js'

const CODECS = { blosc: Blosc, zstd: Zstd, gzip: GZip, zlib: Zlib, lz4: LZ4 }

export async function decodeChunk(bytes, { codec, dtype, littleEndian, count }) {
  const { name, ...configuration } = codec
  if (name !== 'raw') bytes = await CODECS[name].fromConfig(configuration).decode(bytes)
  const Typed = DTYPE_ARRAY[dtype]
  const width = Typed.BYTES_PER_ELEMENT
  if (bytes.byteLength !== count * width) {
    throw new Error(`decoded ${bytes.byteLength} bytes, expected ${count} x ${width} (${dtype})`)
  }
  if (bytes.byteOffset !== 0 || bytes.byteLength !== bytes.buffer.byteLength) bytes = bytes.slice()
  if (!littleEndian && width === 2) {
    for (let i = 0; i < bytes.length; i += 2) {
      const b0 = bytes[i]; bytes[i] = bytes[i + 1]; bytes[i + 1] = b0
    }
  }
  if (!littleEndian && width === 4) {
    for (let i = 0; i < bytes.length; i += 4) {
      const b0 = bytes[i], b1 = bytes[i + 1]
      bytes[i] = bytes[i + 3]; bytes[i + 1] = bytes[i + 2]; bytes[i + 2] = b1; bytes[i + 3] = b0
    }
  }
  return new Typed(bytes.buffer)
}
