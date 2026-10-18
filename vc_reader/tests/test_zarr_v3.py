"""Value-exact v3 chunks/shards through the existing native reader and caches."""

import copy
import ctypes
import ctypes.util
import json
import multiprocessing
from pathlib import Path
import pickle
import unittest

import numpy as np

from test_source_storage import FixtureCase, RemoteFixtureCase, process_read
from vc_reader import Affine, Encoding, Selection, Source, Storage, VolumeRegistration, CacheMissError
from vc_reader.reader import ZarrReader, ZarrGroup
from vc_reader.reader.zarr_meta import ArrayMeta
from vc_reader.reader import S3Credentials


def crc32c(data):
    crc = 0xffffffff
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82f63b78 if crc & 1 else 0)
    return crc ^ 0xffffffff


def fork_read(reader, connection):
    try:
        connection.send(process_read(reader))
    except BaseException as error:
        connection.send(error)
    finally:
        connection.close()


def blosc(data):
    library = ctypes.CDLL(ctypes.util.find_library('blosc'))
    library.blosc_compress_ctx.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int]
    output = ctypes.create_string_buffer(len(data)+16)
    size = library.blosc_compress_ctx(3, 2, 8, len(data), data, output, len(output), b'zstd', 0, 1)
    if size <= 0:
        raise RuntimeError('fixture Blosc compression failed')
    return output.raw[:size]


def write_v3(path, *, sharded=True, index_end=True, compressed=False, endian='little', missing=(1, 2)):
    path.mkdir(parents=True)
    dtype = np.dtype('<u8' if endian == 'little' else '>u8')
    data = np.arange(12*20, dtype=np.uint64).reshape(12, 20).astype(dtype)+np.array(2**40, dtype=dtype)
    data = data.astype(dtype)
    shape, chunk = data.shape, (3, 5)
    outer = (6, 10) if sharded else chunk
    codecs = [dict(name='bytes', configuration=dict(endian=endian))]
    if compressed:
        codecs.append(dict(name='blosc', configuration=dict(typesize=8, cname='zstd', clevel=3, shuffle='bitshuffle', blocksize=0)))
    if sharded:
        codecs = [dict(name='sharding_indexed', configuration=dict(chunk_shape=chunk, codecs=codecs,
            index_codecs=[dict(name='bytes', configuration=dict(endian='little')), dict(name='crc32c')],
            index_location='end' if index_end else 'start'))]
    metadata = dict(zarr_format=3, node_type='array', shape=shape, data_type='uint64',
        chunk_grid=dict(name='regular', configuration=dict(chunk_shape=outer)),
        chunk_key_encoding=dict(name='default', configuration=dict(separator='/')),
        fill_value=7, codecs=codecs, attributes={}, dimension_names=['y', 'x'])
    (path/'zarr.json').write_text(json.dumps(metadata))
    for sy, sx in np.ndindex(tuple(n//c for n, c in zip(shape, outer))):
        body = bytearray()
        index = []
        for iy, ix in np.ndindex((2, 2) if sharded else (1, 1)):
            cy, cx = sy*(2 if sharded else 1)+iy, sx*(2 if sharded else 1)+ix
            if (cy, cx) == missing:
                index.extend([2**64-1, 2**64-1])
                continue
            payload = data[cy*3:(cy+1)*3, cx*5:(cx+1)*5].tobytes()
            if compressed:
                payload = blosc(payload)
            index.extend([len(body)+(68 if sharded and not index_end else 0), len(payload)])
            body.extend(payload)
        if not body:
            continue
        if sharded:
            encoded = np.array(index, dtype='<u8').tobytes()
            encoded += crc32c(encoded).to_bytes(4, 'little')
            body = body+encoded if index_end else encoded+body
        target = path/'c'/str(sy)/str(sx)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    if missing is not None:
        cy, cx = missing
        data[cy*3:(cy+1)*3, cx*5:(cx+1)*5] = 7
    return data, metadata


class V3Tests(FixtureCase):
    @unittest.skipUnless('fork' in multiprocessing.get_all_start_methods(), 'fork is unavailable')
    def test_inherited_native_reader_reopens_after_fork(self):
        path = self.root/'fork'
        expected, _ = write_v3(path, compressed=True)
        with Source('fork', Storage(str(path), 'zarr')).open(memory_cache=4096) as reader:
            reader.read((0, 0), (3, 4))
            context = multiprocessing.get_context('fork')
            parent, child = context.Pipe()
            process = context.Process(target=fork_read, args=(reader, child))
            process.start()
            child.close()
            try:
                self.assertTrue(parent.poll(20), 'forked native reader did not return')
                result = parent.recv()
                if isinstance(result, BaseException):
                    raise result
                np.testing.assert_array_equal(result, expected[:3, :4])
            finally:
                parent.close()
                process.join(5)
                if process.is_alive():
                    process.terminate()
                    process.join()
            self.assertEqual(process.exitcode, 0)
            np.testing.assert_array_equal(reader.read((0, 0), (3, 4)), expected[:3, :4])

    def test_local_shards_layouts_codecs_endian_missing_and_padding(self):
        self.assertEqual(crc32c(b'123456789'), 0xe3069283)
        for sharded, index_end, compressed, endian in ((False, True, False, 'little'),
                (True, True, False, 'little'), (True, False, True, 'little'), (True, True, True, 'big')):
            with self.subTest(sharded=sharded, index_end=index_end, compressed=compressed, endian=endian):
                path = self.root/f'{sharded}-{index_end}-{compressed}-{endian}'
                expected, metadata = write_v3(path, sharded=sharded, index_end=index_end, compressed=compressed, endian=endian)
                with ZarrReader.open(path, threads=2) as reader:
                    np.testing.assert_array_equal(reader.read((0, 0), expected.shape), expected)
                    padded = np.full((6, 7), 7, dtype=expected.dtype)
                    padded[1:, 2:] = expected[:5, :5]
                    np.testing.assert_array_equal(reader.read((-1, -2), (6, 7)), padded)
                with Source('v3', Storage(str(path), 'zarr')).open() as reader:
                    np.testing.assert_array_equal(reader.read((2, 4), (8, 11)), expected[2:10, 4:15])
                    restored = pickle.loads(pickle.dumps(reader))
                    np.testing.assert_array_equal(restored.read((0, 0), (3, 4)), expected[:3, :4])
                    restored.close()

    def test_group_metadata_selection_and_invalid_pipelines(self):
        group = self.root/'group'
        expected, metadata = write_v3(group/'0')
        attrs = dict(ome=dict(version='0.5', multiscales=[dict(axes=[dict(name=a, type='space', unit='micrometer') for a in 'yx'],
            datasets=[dict(path='0', coordinateTransformations=[dict(type='scale', scale=[2., 2.])])])]))
        (group/'zarr.json').write_text(json.dumps(dict(zarr_format=3, node_type='group', attributes=attrs)))
        with ZarrGroup.open(group) as reader:
            self.assertEqual(reader.keys(), ['0'])
            np.testing.assert_array_equal(reader.level(0).read((0, 0), expected.shape), expected)
        source = Source('group', Storage(str(group), 'zarr', selection=Selection(level=0)),
            registration=VolumeRegistration('s', 'v', mode='storage', storage_frame='physical',
                                             storage_units='micrometer', affine=Affine.identity()))
        with source.open() as reader:
            self.assertEqual(reader.registration.affine.linear, ((2., 0., 0.), (0., 2., 0.), (0., 0., 1.)))
        for mutate in (lambda m: m['codecs'][0]['configuration'].update(index_codecs=[dict(name='gzip')]),
                       lambda m: m['codecs'][0]['configuration'].update(chunk_shape=[4, 5]),
                       lambda m: m.update(storage_transformers=[dict(name='unknown')]),
                       lambda m: m.update(codecs=[dict(name='unknown')])):
            invalid = copy.deepcopy(metadata)
            mutate(invalid)
            with self.assertRaises(ValueError):
                ArrayMeta.from_zarray(invalid)

    def test_corrupt_index_and_invalid_offsets_fail(self):
        path = self.root/'corrupt'
        write_v3(path)
        shard = path/'c/0/0'
        original = shard.read_bytes()
        corrupt = bytearray(original)
        corrupt[-1] ^= 1
        shard.write_bytes(corrupt)
        with ZarrReader.open(path) as reader:
            with self.assertRaisesRegex(RuntimeError, 'CRC32C'):
                reader.read((0, 0), (1, 1))
        corrupt = bytearray(original)
        corrupt[-68:-60] = (2**63).to_bytes(8, 'little')
        corrupt[-4:] = crc32c(corrupt[-68:-4]).to_bytes(4, 'little')
        shard.write_bytes(corrupt)
        with ZarrReader.open(path) as reader:
            with self.assertRaisesRegex(RuntimeError, 'offset/length'):
                reader.read((0, 0), (1, 1))


class V3RemoteTests(RemoteFixtureCase):
    def test_rejected_ranges_and_changed_shards(self):
        write_v3(self.root/'v3')
        for prefix, message in (('ignore-range', 'ignored HTTP Range'), ('bad-range', 'Content-Range')):
            with ZarrReader.open(self.url+f'/{prefix}/v3') as reader:
                with self.assertRaisesRegex(RuntimeError, message):
                    reader.read((0, 0), (1, 1))
        with ZarrReader.open(self.url+'/v3') as reader:
            reader.read((0, 0), (1, 1))
            shard = self.root/'v3/c/0/0'
            data = bytearray(shard.read_bytes())
            data[1] ^= 1
            shard.write_bytes(data)
            with self.assertRaisesRegex(RuntimeError, '412'):
                reader.read((4, 0), (1, 1))


    def test_remote_ranges_auth_offline_and_workers(self):
        expected, _ = write_v3(self.root/'v3', compressed=True)
        source = Source('remote-v3', Storage(self.url+'/basic/v3', 'zarr'))
        with source.open(cache_dir=self.cache, cache_max_bytes=2048, auth=('user', 'password')) as reader:
            crop = reader.read((1, 1), (2, 3))
            np.testing.assert_array_equal(crop, expected[1:3, 1:4])
            stats = reader.stats()['zarr']
            self.assertEqual(stats['remote_fetches'], 2)  # One index and one compressed inner chunk.
            self.assertLess(stats['remote_bytes'], (self.root/'v3/c/0/0').stat().st_size)
            reader.read((4, 1), (2, 3))
            self.assertEqual(reader.stats()['zarr']['remote_fetches'], 3)  # Reused index.
        before = dict(self.counts)
        with source.open(cache_dir=self.cache, offline=True) as reader:
            np.testing.assert_array_equal(reader.read((1, 1), (2, 3)), crop)
            with self.assertRaises(CacheMissError):
                reader.read((10, 15), (1, 1))
            with multiprocessing.get_context('spawn').Pool(2) as pool:
                for result in pool.map(process_read, [reader, reader]):
                    np.testing.assert_array_equal(result, expected[:3, :4])
        self.assertEqual(before, self.counts)
        s3 = Source('s3-v3', Storage('s3://bucket/v3', 'zarr'))
        keys = S3Credentials('example', 'test-secret', region='us-east-1', endpoint_url=self.url)
        with s3.open(auth=keys) as reader:
            np.testing.assert_array_equal(reader.read((0, 0), expected.shape), expected)

    def test_remote_group_companions_offline(self):
        attrs = dict(multiscales=[dict(axes=['y', 'x'], datasets=[dict(path='0')])])
        for name in ('labels', 'mask'):
            write_v3(self.root/name/'0')
            (self.root/name/'zarr.json').write_text(json.dumps(dict(zarr_format=3, node_type='group', attributes=attrs)))
        from vc_reader import Companion
        mask = Source('mask', Storage(self.url+'/mask', 'zarr', selection=Selection(level=0)), role='label')
        source = Source('labels', Storage(self.url+'/labels', 'zarr', selection=Selection(level=0)),
                         companions=(Companion('supervision', mask, True),))
        with source.open(cache_dir=self.cache) as reader:
            expected = reader.read((0, 0), (3, 4))
            np.testing.assert_array_equal(reader.companions['supervision'].read((0, 0), (3, 4)), expected)
        before = dict(self.counts)
        with source.open(cache_dir=self.cache, offline=True) as reader:
            np.testing.assert_array_equal(reader.read((0, 0), (3, 4)), expected)
            np.testing.assert_array_equal(reader.companions['supervision'].read((0, 0), (3, 4)), expected)
        self.assertEqual(before, self.counts)


if __name__ == '__main__':
    unittest.main()
