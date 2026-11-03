import { decodeChunk } from './decode.js'
import { fetchRange } from './mirrors.js'

onmessage = async ({ data: { id, url, range, codec, dtype, littleEndian, count } }) => {
  try {
    if (range !== null) {
      const data = await decodeChunk(await fetchRange(url, ...range), { codec, dtype, littleEndian, count })
      postMessage({ id, status: 'ok', data }, [data.buffer])
      return
    }
    const response = await fetch(url)
    if (response.status === 404) {
      postMessage({ id, status: 'missing' })
      return
    }
    if (!response.ok) {
      postMessage({ id, status: 'error', message: `HTTP ${response.status} ${url}` })
      return
    }
    const bytes = new Uint8Array(await response.arrayBuffer())
    const data = await decodeChunk(bytes, { codec, dtype, littleEndian, count })
    postMessage({ id, status: 'ok', data }, [data.buffer])
  } catch (err) {
    postMessage({ id, status: 'error', message: `${url}: ${err.message}` })
  }
}
