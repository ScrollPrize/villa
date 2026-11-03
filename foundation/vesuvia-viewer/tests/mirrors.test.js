import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mirrors, mirror } from '../volume/mirrors.js'

test('an S3 URL that names its region has four mirrors, whichever of them it is', () => {
  const expected = [
    { url: 'https://scrolls.s3.us-east-1.amazonaws.com/a.zarr/0/1/2/3', origin: 'https://scrolls.s3.us-east-1.amazonaws.com' },
    { url: 'https://s3.us-east-1.amazonaws.com/scrolls/a.zarr/0/1/2/3', origin: 'https://s3.us-east-1.amazonaws.com' },
    { url: 'https://scrolls.s3.dualstack.us-east-1.amazonaws.com/a.zarr/0/1/2/3', origin: 'https://scrolls.s3.dualstack.us-east-1.amazonaws.com' },
    { url: 'https://s3.dualstack.us-east-1.amazonaws.com/scrolls/a.zarr/0/1/2/3', origin: 'https://s3.dualstack.us-east-1.amazonaws.com' },
  ]
  for (const { url } of expected) assert.deepEqual(mirrors(url), expected)
})

test('a URL with no region, a dotted bucket, a query or another host is its own only mirror', () => {
  for (const url of [
    'https://scrolls.s3.amazonaws.com/a.zarr/0/1/2/3',
    'https://s3.eu-west-1.amazonaws.com/scrolls.example.org/a.zarr/0/1/2/3',
    'https://scrolls.s3.us-east-1.amazonaws.com/a.zarr/0/1/2/3?X-Amz-Signature=5f',
    'http://localhost:8080/local.zarr/0/0/0/0',
  ]) assert.deepEqual(mirrors(url), [{ url, origin: new URL(url).origin }])
})

test('the chunks of a tilted plane spread over the four mirrors, each chunk always on the same one', () => {
  const counts = new Map()
  for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) {
    const url = `https://scrolls.s3.us-east-1.amazonaws.com/a.zarr/3/${30 - y - x}/${y}/${x}`
    assert.deepEqual(mirror(url), mirror(url))
    counts.set(mirror(url).origin, (counts.get(mirror(url).origin) ?? 0) + 1)
  }
  assert.deepEqual([...counts.values()].sort((a, b) => a - b), [52, 64, 65, 75])
  assert.deepEqual(mirror('http://localhost:8080/local.zarr/0/0/0/0'), { url: 'http://localhost:8080/local.zarr/0/0/0/0', origin: 'http://localhost:8080' })
})
