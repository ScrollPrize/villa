// S3 serves a bucket under four hostnames of its region, and a browser holds 6 HTTP/1.1 connections to each; one connection carries about 1 MiB/s of a far bucket.
import { PART_BYTES } from '../config.js'

// 32-bit FNV-1a, then MurmurHash3's final mix: the low bits of FNV over chunk indices repeat along a tilted plane, which would send a pane's chunks to one mirror.
const FNV_BASIS = 0x811c9dc5
const FNV_PRIME = 0x01000193
const MIX = [[16, 0x85ebca6b], [13, 0xc2b2ae35], [16, 1]]

// [bucket.]s3.[dualstack.]<region>.amazonaws.com. The global bucket.s3.amazonaws.com names no region and does not match.
const S3_HOST = /^(?:([a-z0-9-]+)\.)?s3\.(?:dualstack\.)?([a-z]{2}(?:-[a-z]+)+-\d)\.amazonaws\.com$/

export function mirrors(url) {
  const { protocol, hostname, origin, pathname, search } = new URL(url)
  const match = S3_HOST.exec(hostname)
  if (match === null || search !== '') return [{ url, origin }]   // a query may be a signature, which covers the hostname
  const region = match[2]
  const bucket = match[1] === undefined ? pathname.split('/')[1] : match[1]
  if (!/^[a-z0-9-]+$/.test(bucket)) return [{ url, origin }]   // a dotted bucket has no certificate as a hostname
  const path = match[1] === undefined ? pathname.slice(bucket.length + 1) : pathname
  return ['', 'dualstack.'].flatMap(stack => {
    const host = `${protocol}//${bucket}.s3.${stack}${region}.amazonaws.com`
    const shared = `${protocol}//s3.${stack}${region}.amazonaws.com`
    return [{ url: host + path, origin: host }, { url: `${shared}/${bucket}${path}`, origin: shared }]
  })
}

export function mirror(url, range = null) {
  const places = mirrors(url)
  const name = range === null ? url : `${url} ${range}`
  let hash = FNV_BASIS
  for (let i = 0; i < name.length; i++) hash = Math.imul(hash ^ name.charCodeAt(i), FNV_PRIME)
  for (const [shift, multiplier] of MIX) hash = Math.imul(hash ^ (hash >>> shift), multiplier)
  return places[(hash >>> 0) % places.length]
}

// Two ranges of one URL in flight together leave neither in the browser's cache, so a file over PART_BYTES comes in one ranged part per mirror.
export async function fetchFile(url) {
  const places = mirrors(url)
  let size = 0
  if (places.length > 1) {
    const head = await fetch(url, { method: 'HEAD' })
    if (!head.ok) throw Object.assign(new Error(`${url}: HTTP ${head.status}`), { status: head.status })
    size = Number(head.headers.get('content-length'))
  }
  const count = Math.min(places.length, Math.ceil(size / PART_BYTES))
  if (count <= 1) {
    const response = await fetch(url)
    if (!response.ok) throw Object.assign(new Error(`${url}: HTTP ${response.status}`), { status: response.status })
    return response.arrayBuffer()
  }
  const step = Math.ceil(size / count)
  const bytes = new Uint8Array(size)
  await Promise.all(places.slice(0, count).map(async (place, i) => {
    bytes.set(await fetchRange(place.url, i * step, Math.min(step, size - i * step)), i * step)
  }))
  return bytes.buffer
}

export async function fetchRange(url, first, length) {
  const range = `bytes=${first}-${first + length - 1}`
  const response = await fetch(url, { headers: { Range: range } })
  const bytes = new Uint8Array(await response.arrayBuffer())
  if (response.status !== 206 || bytes.length !== length) throw new Error(`${url} ${range}: HTTP ${response.status}, ${bytes.length} bytes`)
  return bytes
}
