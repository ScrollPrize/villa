import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readVolumeMetadata } from '../volume/ngff.js'

const ONE = [1, 1, 1]
const fromDocs = docs => async url => (url in docs ? docs[url] : null)
const zarrayV2 = (shape, extra) => ({
  zarr_format: 2, shape, chunks: [64, 64, 64], dtype: '|u1', order: 'C', filters: null, fill_value: null,
  compressor: { id: 'blosc', cname: 'zstd', clevel: 5, shuffle: 1, blocksize: 0 }, ...extra,
})

test('v2 multiscale group without translations: levels, block origins at zero, chunk urls, blosc', async () => {
  const B = 'http://h/ct.zarr'
  const docs = {
    [`${B}/.zattrs`]: { multiscales: [{
      axes: [{ name: 'z' }, { name: 'y' }, { name: 'x' }],
      datasets: [0, 1, 2].map(i => ({ path: `${i}`, coordinateTransformations: [{ type: 'scale', scale: [2 ** i, 2 ** i, 2 ** i] }] })),
    }] },
    [`${B}/0/.zarray`]: zarrayV2([300, 200, 100], { dimension_separator: '/' }),
    [`${B}/1/.zarray`]: zarrayV2([150, 100, 50], { dimension_separator: '/' }),
    [`${B}/2/.zarray`]: zarrayV2([75, 50, 25], { dimension_separator: '/' }),
  }
  const meta = await readVolumeMetadata(`${B}/`, { world: { scale_zyx: ONE, translation_zyx: [0, 0, 0] }, fetchJson: fromDocs(docs) })
  assert.equal(meta.version, 2)
  assert.equal(meta.dtype, 'uint8')
  assert.equal(meta.fill, 0)
  assert.equal(meta.labels, false)
  assert.deepEqual(meta.levels.map(l => l.downsample_zyx), [[1, 1, 1], [2, 2, 2], [4, 4, 4]])
  assert.deepEqual(meta.levels.map(l => l.offset_zyx), [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
  assert.deepEqual(meta.levels[2].shape_zyx, [75, 50, 25])
  assert.equal(meta.levels[1].chunkUrl(3, 2, 1), `${B}/1/3/2/1`)
  assert.deepEqual(meta.levels[0].codec, { name: 'blosc', cname: 'zstd', clevel: 5, shuffle: 1, blocksize: 0 })
  assert.equal(meta.levels[0].littleEndian, true)
})

test('v2 single array: dot separator, big-endian uint16, numeric fill', async () => {
  const B = 'http://h/plain.zarr'
  const docs = { [`${B}/.zarray`]: zarrayV2([10, 20, 30], { dtype: '>u2', compressor: null, fill_value: 7 }) }
  const meta = await readVolumeMetadata(B, { world: { scale_zyx: ONE, translation_zyx: [0, 0, 0] }, fetchJson: fromDocs(docs) })
  assert.equal(meta.levels.length, 1)
  assert.equal(meta.dtype, 'uint16')
  assert.equal(meta.fill, 7)
  assert.equal(meta.levels[0].littleEndian, false)
  assert.deepEqual(meta.levels[0].codec, { name: 'raw' })
  assert.deepEqual(meta.levels[0].downsample_zyx, [1, 1, 1])
  assert.equal(meta.levels[0].chunkUrl(0, 1, 2), `${B}/0.1.2`)
})

test('an image-label attribute marks a labels volume, v2 and v3 ome', async () => {
  const B = 'http://h/labels.zarr'
  const multiscales = [{ datasets: [{ path: '0', coordinateTransformations: [{ type: 'scale', scale: ONE }] }] }]
  const v2 = { [`${B}/.zattrs`]: { multiscales, 'image-label': {} }, [`${B}/0/.zarray`]: zarrayV2([64, 64, 64]) }
  assert.equal((await readVolumeMetadata(B, { world: { scale_zyx: ONE, translation_zyx: [0, 0, 0] }, fetchJson: fromDocs(v2) })).labels, true)
  const array = {
    zarr_format: 3, node_type: 'array', shape: [64, 64, 64], data_type: 'uint8', fill_value: 0,
    chunk_grid: { name: 'regular', configuration: { chunk_shape: [64, 64, 64] } },
    chunk_key_encoding: { name: 'default', configuration: { separator: '/' } },
    codecs: [{ name: 'bytes', configuration: { endian: 'little' } }],
  }
  const v3 = {
    [`${B}/zarr.json`]: { zarr_format: 3, node_type: 'group', attributes: { ome: { version: '0.5', multiscales, 'image-label': {} } } },
    [`${B}/0/zarr.json`]: array,
  }
  assert.equal((await readVolumeMetadata(B, { world: { scale_zyx: ONE, translation_zyx: [0, 0, 0] }, fetchJson: fromDocs(v3) })).labels, true)
})

test('v3 group with ome 0.5 attrs: default key encoding, bytes + zstd, translation', async () => {
  const B = 'http://h/pred.zarr'
  const arrayV3 = shape => ({
    zarr_format: 3, node_type: 'array', shape, data_type: 'uint8', fill_value: 0,
    chunk_grid: { name: 'regular', configuration: { chunk_shape: [32, 32, 32] } },
    chunk_key_encoding: { name: 'default', configuration: { separator: '/' } },
    codecs: [{ name: 'bytes', configuration: { endian: 'little' } }, { name: 'zstd', configuration: { level: 3, checksum: false } }],
  })
  const docs = {
    [`${B}/zarr.json`]: { zarr_format: 3, node_type: 'group', attributes: { ome: { version: '0.5', multiscales: [{
      axes: [{ name: 'z' }, { name: 'y' }, { name: 'x' }],
      datasets: [
        { path: '0', coordinateTransformations: [{ type: 'scale', scale: [2, 2, 2] }, { type: 'translation', translation: [100, 200, 300] }] },
        { path: '1', coordinateTransformations: [{ type: 'scale', scale: [4, 4, 4] }, { type: 'translation', translation: [100, 200, 300] }] },
      ],
    }] } } },
    [`${B}/0/zarr.json`]: arrayV3([64, 64, 64]),
    [`${B}/1/zarr.json`]: arrayV3([32, 32, 32]),
  }
  const meta = await readVolumeMetadata(B, { world: { scale_zyx: ONE, translation_zyx: [0, 0, 0] }, fetchJson: fromDocs(docs) })
  assert.equal(meta.version, 3)
  assert.deepEqual(meta.levels.map(l => l.downsample_zyx), [[2, 2, 2], [4, 4, 4]])
  assert.deepEqual(meta.levels[1].offset_zyx, [98.5, 198.5, 298.5])
  assert.deepEqual(meta.levels[0].chunk_zyx, [32, 32, 32])
  assert.equal(meta.levels[0].chunkUrl(1, 0, 2), `${B}/0/c/1/0/2`)
  assert.deepEqual(meta.levels[0].codec, { name: 'zstd', level: 3, checksum: false })
})

test('group attrs register an array opened at url/probs', async () => {
  const B = 'http://h/infer.zarr'
  const docs = {
    [`${B}/.zattrs`]: { roi_origin_zyx: [10, 20, 30], level: 2 },
    [`${B}/.zgroup`]: { zarr_format: 2 },
    [`${B}/probs/.zarray`]: zarrayV2([8, 8, 8], { dtype: '<f4', compressor: { id: 'zstd', level: 1 } }),
  }
  const meta = await readVolumeMetadata(`${B}/probs`, { world: { scale_zyx: ONE, translation_zyx: [0, 0, 0] }, fetchJson: fromDocs(docs) })
  assert.equal(meta.dtype, 'float32')
  assert.deepEqual(meta.levels[0].downsample_zyx, [4, 4, 4])
  assert.deepEqual(meta.levels[0].offset_zyx, [40, 80, 120])
  assert.deepEqual(meta.levels[0].codec, { name: 'zstd', level: 1 })
  assert.equal(meta.levels[0].chunkUrl(0, 0, 0), `${B}/probs/0.0.0`)
})

test('raises: four axes, sharding, non-integer downsample', async () => {
  const B = 'http://h/bad.zarr'
  const four = { [`${B}/.zarray`]: zarrayV2([1, 2, 3, 4], { chunks: [1, 1, 1, 1] }) }
  await assert.rejects(readVolumeMetadata(B, { world: { scale_zyx: ONE, translation_zyx: [0, 0, 0] }, fetchJson: fromDocs(four) }), /expected 3 axes, got 4 \[1, 2, 3, 4\]/)

  const sharded = { [`${B}/zarr.json`]: {
    zarr_format: 3, node_type: 'array', shape: [8, 8, 8], data_type: 'uint8', fill_value: 0,
    chunk_grid: { name: 'regular', configuration: { chunk_shape: [8, 8, 8] } },
    chunk_key_encoding: { name: 'default' },
    codecs: [{ name: 'sharding_indexed', configuration: {} }],
  } }
  await assert.rejects(readVolumeMetadata(B, { world: { scale_zyx: ONE, translation_zyx: [0, 0, 0] }, fetchJson: fromDocs(sharded) }), /sharding not supported/)

  const fractional = {
    [`${B}/.zattrs`]: { multiscales: [{ datasets: [{ path: '0', coordinateTransformations: [{ type: 'scale', scale: [1, 1, 1.5] }] }] }] },
    [`${B}/0/.zarray`]: zarrayV2([8, 8, 8]),
  }
  await assert.rejects(readVolumeMetadata(B, { world: { scale_zyx: ONE, translation_zyx: [0, 0, 0] }, fetchJson: fromDocs(fractional) }), /downsample axis 2 1.5 is not an integer/)
})

test('fill_value: null and float NaN read as 0; anything else raises with the value', async () => {
  const B = 'http://h/fill.zarr'
  const open = extra => readVolumeMetadata(B, { world: { scale_zyx: ONE, translation_zyx: [0, 0, 0] }, fetchJson: fromDocs({ [`${B}/.zarray`]: zarrayV2([8, 8, 8], extra) }) })
  assert.equal((await open({ fill_value: null })).fill, 0)
  assert.equal((await open({ dtype: '<f4', fill_value: 'NaN' })).fill, 0)
  assert.equal((await open({ dtype: '<f4', fill_value: -0.5 })).fill, -0.5)
  for (const [dtype, max] of [['|u1', 255], ['<u2', 65535]]) {
    assert.equal((await open({ dtype, fill_value: max })).fill, max)
    for (const fill_value of [-1, 0.5, max + 1]) {
      await assert.rejects(open({ dtype, fill_value }), { message: `http://h/fill.zarr: fill_value ${fill_value} not supported for ${dtype === '|u1' ? 'uint8' : 'uint16'}` })
    }
  }
  await assert.rejects(open({ fill_value: 'NaN' }), /fill_value "NaN" not supported for uint8/)
  await assert.rejects(open({ fill_value: undefined }), /fill_value undefined not supported/)
  await assert.rejects(open({ fill_value: '0' }), /fill_value "0" not supported/)
  await assert.rejects(open({ fill_value: { v: 1 } }), /fill_value {"v":1} not supported/)
})

test('filters must be null or an empty array', async () => {
  const B = 'http://h/filters.zarr'
  const open = filters => readVolumeMetadata(B, { world: { scale_zyx: ONE, translation_zyx: [0, 0, 0] }, fetchJson: fromDocs({ [`${B}/.zarray`]: zarrayV2([8, 8, 8], { filters }) }) })
  assert.equal((await open([])).levels.length, 1)
  await assert.rejects(open({ id: 'delta' }), /filters {"id":"delta"} not supported/)
  await assert.rejects(open([{ id: 'delta' }]), /filters \[{"id":"delta"}\] not supported/)
  await assert.rejects(open(undefined), /filters undefined not supported/)
})

test('multiscale-level coordinateTransformations are rejected', async () => {
  const B = 'http://h/transformed.zarr'
  const docs = {
    [`${B}/.zattrs`]: { multiscales: [{
      coordinateTransformations: [{ type: 'scale', scale: [3, 4, 5] }],
      datasets: [{ path: '0', coordinateTransformations: [{ type: 'scale', scale: [1, 1, 1] }] }],
    }] },
    [`${B}/0/.zarray`]: zarrayV2([8, 8, 8]),
  }
  await assert.rejects(readVolumeMetadata(B, { world: { scale_zyx: ONE, translation_zyx: [0, 0, 0] }, fetchJson: fromDocs(docs) }), /multiscale-level coordinateTransformations not supported/)
})

test('a group opened by URL prefers its array registration', async () => {
  const B = 'http://h/registered.zarr'
  const docs = {
    [`${B}/.zattrs`]: { roi_origin_zyx: [1, 2, 3], level: 1 },
    [`${B}/0/.zarray`]: zarrayV2([8, 8, 8]),
    [`${B}/0/.zattrs`]: { roi_origin_zyx: [10, 20, 30], level: 2 },
  }
  const meta = await readVolumeMetadata(B, { world: { scale_zyx: ONE, translation_zyx: [0, 0, 0] }, fetchJson: fromDocs(docs) })
  assert.deepEqual(meta.levels[0].downsample_zyx, [4, 4, 4])
  assert.deepEqual(meta.levels[0].offset_zyx, [40, 80, 120])
})

test('metadata and chunk paths precede URL query parameters', async () => {
  const B = 'http://h/plain.zarr'
  const docs = { [`${B}/.zarray?token=a%2Bb`]: zarrayV2([8, 8, 8]) }
  const meta = await readVolumeMetadata(`${B}?token=a%2Bb`, { world: { scale_zyx: ONE, translation_zyx: [0, 0, 0] }, fetchJson: fromDocs(docs) })
  assert.equal(meta.levels[0].chunkUrl(1, 2, 3), `${B}/1.2.3?token=a%2Bb`)
})

test('invalid chunk shapes and zero or nonfinite scales fail at metadata', async () => {
  const B = 'http://h/bad.zarr'
  await assert.rejects(readVolumeMetadata(B, { world: { scale_zyx: ONE, translation_zyx: [0, 0, 0] }, fetchJson: fromDocs({
    [`${B}/.zarray`]: zarrayV2([8, 8, 8], { chunks: [0, 8, 8] }),
  }) }), /chunks.*\[0,8,8\]/)
  for (const scale of [[0, 1, 1], [NaN, 1, 1]]) {
    await assert.rejects(readVolumeMetadata(B, { world: { scale_zyx: ONE, translation_zyx: [0, 0, 0] }, fetchJson: fromDocs({
      [`${B}/.zattrs`]: { multiscales: [{ datasets: [{ path: '0', coordinateTransformations: [{ type: 'scale', scale }] }] }] },
      [`${B}/0/.zarray`]: zarrayV2([8, 8, 8]),
    }) }), /scale: invalid values/)
  }
})

test('NGFF mean-pyramid centres stay aligned relative to a translated CT origin', async () => {
  const B = 'http://h/mean.zarr'
  const origin_zyx = [100, 200, 300], scale_zyx = [8, 8, 4]
  const docs = {
    [`${B}/.zattrs`]: { multiscales: [{ datasets: [1, 4].map((d, i) => ({ path: String(i), coordinateTransformations: [
      { type: 'scale', scale: scale_zyx.map(s => s * d) },
      { type: 'translation', translation: origin_zyx.map((t, a) => t + (d - 1) / 2 * scale_zyx[a]) },
    ] })) }] },
    [`${B}/0/.zarray`]: zarrayV2([64, 64, 64]),
    [`${B}/1/.zarray`]: zarrayV2([16, 16, 16]),
  }
  const ct = await readVolumeMetadata(B, { fetchJson: fromDocs(docs) })
  assert.deepEqual(ct.translation_zyx, origin_zyx)
  assert.deepEqual(ct.levels.map(l => l.offset_zyx), [[0, 0, 0], [0, 0, 0]])
  const layer = await readVolumeMetadata(B, { world: ct, fetchJson: fromDocs(docs) })
  assert.deepEqual(layer.levels.map(l => l.offset_zyx), [[0, 0, 0], [0, 0, 0]])
  const coarse = layer.levels[1]
  assert.equal(coarse.offset_zyx[2] + 3 * coarse.downsample_zyx[2] + (coarse.downsample_zyx[2] - 1) / 2, 13.5)
})

test('a scale-only layer on a translated CT starts its blocks at its own origin', async () => {
  const B = 'http://h/plain-layer.zarr'
  const docs = {
    [`${B}/.zattrs`]: { multiscales: [{ datasets: [1, 2].map((d, i) => ({ path: String(i), coordinateTransformations: [{ type: 'scale', scale: [d * 8, d * 8, d * 4] }] })) }] },
    [`${B}/0/.zarray`]: zarrayV2([16, 16, 16]),
    [`${B}/1/.zarray`]: zarrayV2([8, 8, 8]),
  }
  const layer = await readVolumeMetadata(B, { world: { scale_zyx: [8, 8, 4], translation_zyx: [80, 160, 40] }, fetchJson: fromDocs(docs) })
  assert.deepEqual(layer.levels.map(l => l.offset_zyx), [[-10, -20, -10], [-10, -20, -10]])
})

test('a translation transform must provide its vector', async () => {
  const B = 'http://h/missing-translation.zarr'
  const docs = {
    [`${B}/.zattrs`]: { multiscales: [{ datasets: [{ path: '0', coordinateTransformations: [
      { type: 'scale', scale: [1, 1, 1] }, { type: 'translation' },
    ] }] }] },
    [`${B}/0/.zarray`]: zarrayV2([8, 8, 8]),
  }
  await assert.rejects(readVolumeMetadata(B, { fetchJson: fromDocs(docs) }), /translation.*undefined/)
})
