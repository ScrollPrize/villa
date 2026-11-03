import { test } from 'node:test'
import assert from 'node:assert/strict'
import { catalogEntries } from '../extensions/catalog.js'

const ACCESS = { dataAccess: { rewrites: [{ from: 's3://open/', to: 'https://data.example/' }] } }
const bucket = path => ({ origins: [{ path, access_roots: [{ type: 's3', url: 's3://open' }] }] })
const METADATA = {
  samples: {
    Scroll1: {
      volumes: {
        900: { properties: { pixel_size_um: 2.4 }, data: [
          { type: 'ome-zarr', ...bucket('Scroll1/volumes/900-2.4um.zarr/') },
          { type: 'surface-prediction-zarr', ...bucket('Scroll1/predictions/900-surface-m7.zarr/') },
          { type: 'normal-grids', ...bucket('Scroll1/predictions/900-surface-m7.normal-grids/') },
        ] },
        901: { properties: { pixel_size_um: 8 }, data: [
          { type: 'ome-zarr', origins: [{ path: '/Scroll1/volumes/901-8um.zarr/', access_roots: [{ type: 'https', url: 'https://mirror.example/' }] }] },
        ] },
      },
      segments: {
        s1: { data: [
          { type: 'tifxyz-normalized', ...bucket('Scroll1/segments/s1/mesh/normalized.tifxyz/') },
          { type: 'tifxyz-transformed', parameters: { target_volume: '900' }, ...bucket('Scroll1/segments/s1/mesh/s1-on-900-2.4um.tifxyz/') },
        ] },
        s2: { data: [
          { type: 'tifxyz-transformed', parameters: { target_volume: '901' }, ...bucket('Scroll1/segments/s2/mesh/s2-on-901-8um.tifxyz/') },
        ] },
      },
    },
    Scroll2: { volumes: { 902: { data: [{ type: 'photo', ...bucket('Scroll2/photo.jpg') }] } }, segments: {} },
  },
}

test('every volume with a CT becomes one entry carrying its own predictions and surfaces', () => {
  assert.deepEqual(catalogEntries(METADATA, ACCESS), [
    {
      sample: 'Scroll1',
      name: '900-2.4um.zarr',
      label: '2.4 µm, 900',
      ct: 'https://data.example/Scroll1/volumes/900-2.4um.zarr/',
      layers: [{ name: '900-surface-m7.zarr', url: 'https://data.example/Scroll1/predictions/900-surface-m7.zarr/' }],
      surfaces: [{ name: 's1-on-900-2.4um.tifxyz', url: 'https://data.example/Scroll1/segments/s1/mesh/s1-on-900-2.4um.tifxyz/' }],
    },
    {
      sample: 'Scroll1',
      name: '901-8um.zarr',
      label: '8 µm, 901',
      ct: 'https://mirror.example/Scroll1/volumes/901-8um.zarr/',
      layers: [],
      surfaces: [{ name: 's2-on-901-8um.tifxyz', url: 'https://data.example/Scroll1/segments/s2/mesh/s2-on-901-8um.tifxyz/' }],
    },
  ])
})

test('a record with no public HTTPS origin and a document without samples are reported, not skipped', () => {
  const metadata = structuredClone(METADATA)
  metadata.samples.Scroll1.volumes[900].data[0].origins[0].access_roots = [{ type: 's3', url: 's3://closed' }]
  assert.throws(() => catalogEntries(metadata, ACCESS), /ome-zarr: no public HTTPS origin in s3:\/\/closed\/Scroll1\/volumes\/900-2.4um.zarr\//)
  assert.throws(() => catalogEntries({ samples: null }, ACCESS), /samples null is not an object/)
})
