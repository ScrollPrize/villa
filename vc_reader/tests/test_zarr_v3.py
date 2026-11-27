"""Value-exact v3 chunks/shards through the existing C++ reader and caches."""

import copy
import ctypes
import ctypes.util
import hashlib
import json
import multiprocessing
from pathlib import Path
import pickle
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
import zlib

import numpy as np

from test_source_storage import FixtureCase, RemoteFixtureCase, process_read
from build_support import load_build
load_build()

from vc_reader import Affine, Encoding, Selection, Source, Storage, VolumeRegistration, CacheMissError
from vc_reader.reader import MemoryCache, ZarrReader, ZarrGroup, ZarrWriter
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
    def test_inherited_chunk_reader_reopens_after_fork(self):
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
                self.assertTrue(parent.poll(20), 'forked C++ reader did not return')
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
                    np.testing.assert_array_equal(reader.read_region((0, 0), expected.shape), expected)
                    padded = np.full((6, 7), 7, dtype=expected.dtype)
                    padded[1:, 2:] = expected[:5, :5]
                    np.testing.assert_array_equal(reader.read_region((-1, -2), (6, 7)), padded)
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
            np.testing.assert_array_equal(reader.level(0).read_region((0, 0), expected.shape), expected)
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
                reader.read_region((0, 0), (1, 1))
        corrupt = bytearray(original)
        corrupt[-68:-60] = (2**63).to_bytes(8, 'little')
        corrupt[-4:] = crc32c(corrupt[-68:-4]).to_bytes(4, 'little')
        shard.write_bytes(corrupt)
        with ZarrReader.open(path) as reader:
            with self.assertRaisesRegex(RuntimeError, 'offset/length'):
                reader.read_region((0, 0), (1, 1))


def write_custom_shard_array(path, data, chunk_shape, shard_shape, missing_chunks=()):
    path.mkdir(parents=True, exist_ok=True)
    missing_set = set(missing_chunks)
    dtype = np.dtype(data.dtype)
    endian = 'big' if dtype.byteorder == '>' else 'little'
    if dtype.itemsize == 1:
        inner_codecs = [dict(name='bytes')]
    else:
        inner_codecs = [dict(name='bytes', configuration=dict(endian=endian))]
    codecs = [dict(name='sharding_indexed', configuration=dict(
        chunk_shape=list(chunk_shape),
        codecs=inner_codecs,
        index_codecs=[dict(name='bytes', configuration=dict(endian='little')), dict(name='crc32c')],
        index_location='end',
    ))]
    dtype_name = {
        np.dtype('u1'): 'uint8',
        np.dtype('<u8'): 'uint64',
    }[dtype]
    metadata = dict(
        zarr_format=3,
        node_type='array',
        shape=list(data.shape),
        data_type=dtype_name,
        chunk_grid=dict(name='regular', configuration=dict(chunk_shape=list(shard_shape))),
        chunk_key_encoding=dict(name='default', configuration=dict(separator='/')),
        fill_value=7,
        codecs=codecs,
        attributes={},
        dimension_names=[f'd{i}' for i in range(data.ndim)],
    )
    (path / 'zarr.json').write_text(json.dumps(metadata))
    shard_grid = tuple(s // sh for s, sh in zip(data.shape, shard_shape))
    inner_grid = tuple(sh // ch for sh, ch in zip(shard_shape, chunk_shape))
    expected = data.copy()
    for s_idx in np.ndindex(shard_grid):
        body = bytearray()
        index = []
        for i_idx in np.ndindex(inner_grid):
            c_idx = tuple(si * ig + ii for si, ig, ii in zip(s_idx, inner_grid, i_idx))
            slices = tuple(slice(ci * ch, (ci + 1) * ch) for ci, ch in zip(c_idx, chunk_shape))
            if c_idx in missing_set:
                index.extend([2**64 - 1, 2**64 - 1])
                expected[slices] = 7
                continue
            payload = data[slices].tobytes()
            index.extend([len(body), len(payload)])
            body.extend(payload)
        encoded = np.array(index, dtype='<u8').tobytes()
        encoded += crc32c(encoded).to_bytes(4, 'little')
        body.extend(encoded)
        target = path / 'c' / '/'.join(str(x) for x in s_idx)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes(body))
    return expected


class V3RemoteTests(RemoteFixtureCase):
    def test_nan_fill_cache_reopens_and_rejects_metadata_changes(self):
        path = self.root/'nan'
        path.mkdir()
        metadata = dict(zarr_format=3, node_type='array', shape=[2, 2], data_type='float32',
            chunk_grid=dict(name='regular', configuration=dict(chunk_shape=[1, 2])),
            chunk_key_encoding=dict(name='default', configuration=dict(separator='/')),
            fill_value='NaN', codecs=[dict(name='bytes', configuration=dict(endian='little'))], attributes={})
        (path/'zarr.json').write_text(json.dumps(metadata))
        (path/'c/0').mkdir(parents=True)
        (path/'c/0/0').write_bytes(np.array([1., 2.], dtype='<f4').tobytes())
        url = self.url+'/nan'
        for offline in (False, False, True):
            with ZarrReader.open(url, cache_dir=self.cache, offline=offline) as reader:
                np.testing.assert_array_equal(reader[:], [[1., 2.], [np.nan, np.nan]])
        # A cache without the provenance record still compares decoded fills.
        from vc_reader.reader.disk_cache import decoded_cache_path
        attrs = Path(decoded_cache_path(self.cache, url))/'.zattrs'
        saved = attrs.read_bytes()
        attrs.unlink()
        with ZarrReader.open(url, cache_dir=self.cache) as reader:
            np.testing.assert_array_equal(reader[:], [[1., 2.], [np.nan, np.nan]])
        attrs.write_bytes(saved)
        for change in (dict(fill_value=0), dict(shape=[3, 2]), dict(attributes={'changed': True})):
            (path/'zarr.json').write_text(json.dumps(dict(metadata, **change)))
            with self.assertRaisesRegex(ValueError, 'holds a different array'):
                ZarrReader.open(url, cache_dir=self.cache)

    def test_cache_from_another_tool_serves_any_readable_codec(self):
        data = np.arange(8, dtype='u1').reshape(2, 2, 2)
        path = self.root/'foreign'
        path.mkdir()
        zarray = dict(zarr_format=2, shape=[2, 2, 4], chunks=[2, 2, 2], dtype='|u1', fill_value=0,
            compressor=None, filters=None, order='C', dimension_separator='/')
        (path/'.zarray').write_text(json.dumps(zarray))
        for key, chunk in (('0/0/0', data), ('0/0/1', data+100)):
            (path/key).parent.mkdir(parents=True, exist_ok=True)
            (path/key).write_bytes(chunk.tobytes())
        from vc_reader.reader.disk_cache import decoded_cache_path
        url = self.url+'/foreign'
        cache = Path(decoded_cache_path(self.cache, url))
        (cache/'0/0').mkdir(parents=True)
        (cache/'.zarray').write_text(json.dumps(dict(zarray, compressor=dict(id='zlib', level=1))))
        (cache/'.zattrs').write_text(json.dumps(dict(remote_url=url, level=0)))
        (cache/'0/0/0').write_bytes(zlib.compress((data+50).tobytes()))
        with ZarrReader.open(url, cache_dir=self.cache) as reader:
            np.testing.assert_array_equal(reader[:, :, :2], data+50)  # the cache served it
            np.testing.assert_array_equal(reader[:, :, 2:], data+100)  # fetched, not cached as raw bytes
        self.assertFalse((cache/'0/0/1').exists())
        # A codec vc_reader cannot read is still refused.
        (cache/'.zarray').write_text(json.dumps(dict(zarray, compressor=dict(id='no-such-codec'))))
        with self.assertRaisesRegex(ValueError, 'holds a different array'):
            ZarrReader.open(url, cache_dir=self.cache)

    def test_rejected_ranges_and_changed_shards(self):
        write_v3(self.root/'v3')
        for prefix, message in (('ignore-range', 'ignored HTTP Range'), ('bad-range', 'Content-Range')):
            with ZarrReader.open(self.url+f'/{prefix}/v3') as reader:
                with self.assertRaisesRegex(RuntimeError, message):
                    reader.read_region((0, 0), (1, 1))
        with ZarrReader.open(self.url+'/v3') as reader:
            reader.read_region((0, 0), (1, 1))
            shard = self.root/'v3/c/0/0'
            data = bytearray(shard.read_bytes())
            data[1] ^= 1
            shard.write_bytes(data)
            with self.assertRaisesRegex(RuntimeError, '412'):
                reader.read_region((4, 0), (1, 1))


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

    def test_revisiting_replaced_shard_refreshes_cached_index(self):
        for missing in (None, (0, 0)):
            for batched in (False, True):
                with self.subTest(missing=missing, batched=batched):
                    name = f'replaced-{missing is not None}-{batched}'
                    root = self.root/name
                    write_v3(root, missing=missing)
                    expected, _ = write_v3(self.root/(name+'-new'), missing=None)
                    replacement = bytearray((self.root/(name+'-new')/'c'/'0'/'0').read_bytes())
                    if missing is None:
                        replacement[0] ^= 1
                        expected[0, 0] = int(expected[0, 0]) ^ 1
                    with ZarrReader.open(self.url+'/'+name, threads=2) as reader:
                        reader[0, 5]
                        reader[0, 10]
                        (root/'c'/'0'/'0').write_bytes(replacement)
                        selection = np.s_[:3, :10] if batched else np.s_[:1, :1]
                        np.testing.assert_array_equal(reader[selection], expected[selection])

    def test_rejects_signed_and_overflowing_content_range(self):
        name = 'malformed-range'
        write_v3(self.root/name, missing=None)
        shard = (self.root/name/'c'/'0'/'0').read_bytes()
        for header in (f'bytes +0-+119/+{len(shard)}', f'bytes 0-119/{2**64}',
                       f'bytes 0-119/{len(shard)} trailing'):
            with self.subTest(header=header):
                def answer(handler):
                    if handler.path != f'/{name}/c/0/0' or handler.headers.get('Range') != 'bytes=0-119':
                        return False
                    handler.send_response(206)
                    handler.send_header('Content-Range', header)
                    handler.send_header('Content-Length', '120')
                    handler.end_headers()
                    handler.wfile.write(shard[:120])
                    return True
                self.server.request_hook = answer
                with ZarrReader.open(self.url+'/'+name, threads=2) as reader:
                    with self.assertRaisesRegex(RuntimeError, 'Content-Range'):
                        reader[:1, :1]

    def test_batch_only_fetches_missing_chunks_and_reuses_each_index(self):
        expected, _ = write_v3(self.root/'batch', missing=None)
        ranges = []

        def capture(handler):
            if handler.headers.get('Range'):
                ranges.append((handler.path, handler.headers['Range']))
            return False

        self.server.request_hook = capture
        uri = self.url + '/batch'
        with ZarrReader.open(uri, cache_dir=self.cache, threads=4, memory_cache=4096) as reader:
            np.testing.assert_array_equal(reader[:3, :5], expected[:3, :5])
            selections = [np.s_[0:6, 10:20], np.s_[0:6, 0:10], np.s_[:3, :5], np.s_[:0, :5]]
            for actual, selection in zip(reader.read_many(selections), selections):
                np.testing.assert_array_equal(actual, expected[selection])
            self.assertCountEqual(ranges, [
                ('/batch/c/0/0', 'bytes=-68'), ('/batch/c/0/0', 'bytes=0-119'),
                ('/batch/c/0/1', 'bytes=-68'), ('/batch/c/0/0', 'bytes=120-479'),
                ('/batch/c/0/1', 'bytes=0-479'),
            ])
        ranges.clear()
        with ZarrReader.open(uri, cache_dir=self.cache, offline=True) as reader:
            np.testing.assert_array_equal(reader[:6, :], expected[:6, :])
        self.assertEqual(ranges, [])

        with ZarrReader.open(uri, threads=2) as reader:
            for selection in (np.s_[:3, :5], np.s_[:3, 10:15], np.s_[3:6, :5]):
                np.testing.assert_array_equal(reader[selection], expected[selection])
        self.assertEqual(sum(value == 'bytes=-68' for _, value in ranges), 2)

    def test_batch_sorts_and_reconstructs_overlapping_payloads_without_gap_bytes(self):
        path = self.root/'overlap'
        expected, _ = write_v3(path, missing=None)
        shard = path/'c/0/0'
        body = shard.read_bytes()[:-68]
        positions = [112, 0, 240, 240]
        index = np.array([(offset, 120) for offset in positions], dtype='<u8').tobytes()
        shard.write_bytes(body + index + crc32c(index).to_bytes(4, 'little'))
        for (y, x), offset in zip(np.ndindex(2, 2), positions):
            expected[3*y:3*y+3, 5*x:5*x+5] = np.frombuffer(body[offset:offset+120], '<u8').reshape(3, 5)
        ranges = []

        def capture(handler):
            if handler.path == '/overlap/c/0/0':
                ranges.append(handler.headers['Range'])
            return False

        self.server.request_hook = capture
        with ZarrReader.open(self.url + '/overlap', threads=4) as reader:
            np.testing.assert_array_equal(reader[:6, :10], expected[:6, :10])
            self.assertEqual(reader.stats()['remote_bytes'], 68 + 232 + 120)
        self.assertCountEqual(ranges, ['bytes=-68', 'bytes=0-231', 'bytes=240-359'])

    def test_overlapping_callers_share_batch_producers(self):
        expected, _ = write_v3(self.root/'shared', missing=None)
        index_requested = threading.Event()
        release_index = threading.Event()
        ranges = []

        def hold_index(handler):
            if handler.path != '/shared/c/0/0':
                return False
            value = handler.headers['Range']
            ranges.append(value)
            if value == 'bytes=-68':
                index_requested.set()
                release_index.wait(5)
            return False

        self.server.request_hook = hold_index
        uri = self.url + '/shared'
        storage = MemoryCache(4096)
        with ZarrReader.open(uri, threads=2, memory_cache=storage) as one, \
                ZarrReader.open(uri, threads=2, memory_cache=storage) as two, \
                ThreadPoolExecutor(2) as pool:
            a = pool.submit(lambda: one[:6, :10])
            try:
                self.assertTrue(index_requested.wait(5))
                b = pool.submit(lambda: two[:6, :10])
            finally:
                release_index.set()
            np.testing.assert_array_equal(a.result(timeout=10), expected[:6, :10])
            np.testing.assert_array_equal(b.result(timeout=10), expected[:6, :10])
        self.assertCountEqual(ranges, ['bytes=-68', 'bytes=0-479'])

    def test_batch_caps_merged_range_bytes(self):
        data = np.arange(3 * 512 * 512, dtype=np.uint32).astype(np.uint8).reshape(3, 512, 512)
        writer = ZarrWriter.create(self.root/'bounded', shape=data.shape, chunks=(1, 512, 512),
                                   dtype='u1', version=3, shards=data.shape, compressor=None, separator='/')
        writer[:] = data
        ranges = []

        def capture(handler):
            if handler.path == '/bounded/c/0/0/0' and handler.headers['Range']:
                ranges.append(handler.headers['Range'])
            return False

        self.server.request_hook = capture
        with ZarrReader.open(self.url + '/bounded', threads=4) as reader:
            np.testing.assert_array_equal(reader[:], data)
            self.assertEqual(reader.stats()['remote_fetches'], 3)
        self.assertCountEqual(ranges, ['bytes=-52', 'bytes=0-524287', 'bytes=524288-786431'])

    def test_batch_retry_rejection_and_invalid_range(self):
        expected, _ = write_v3(self.root/'reject', missing=None)
        shard = (self.root/'reject/c/0/0').read_bytes()
        for mode in ('retry', 'fallback', 'invalid', 'truncated'):
            with self.subTest(mode=mode):
                attempts = []

                def respond(handler):
                    if handler.path != '/reject/c/0/0' or handler.headers.get('Range') != 'bytes=0-479':
                        return False
                    attempts.append(mode)
                    if mode == 'retry':
                        if len(attempts) > 1:
                            return False
                        handler.send_error(503)
                    elif mode == 'fallback':
                        handler.send_error(416)
                    else:
                        handler.send_response(206)
                        trailer = ' trailing' if mode == 'invalid' else ''
                        size = 480 if mode == 'invalid' else 100
                        handler.send_header('Content-Range', f'bytes 0-479/{len(shard)}{trailer}')
                        handler.send_header('Content-Length', str(size))
                        handler.end_headers()
                        handler.wfile.write(shard[:size])
                    return True

                self.server.request_hook = respond
                with ZarrReader.open(self.url + '/reject', threads=4) as reader:
                    if mode in ('invalid', 'truncated'):
                        message = 'Content-Range' if mode == 'invalid' else 'requested HTTP byte range'
                        with self.assertRaisesRegex(RuntimeError, message):
                            reader[:6, :10]
                    else:
                        np.testing.assert_array_equal(reader[:6, :10], expected[:6, :10])
                        self.assertEqual(reader.stats()['remote_fetches'], 6 if mode == 'fallback' else 2)
                self.assertEqual(len(attempts), 2 if mode == 'retry' else 1)

    def test_shard_visit_safety_pending_late_invalid_and_offline(self):
        # 1a. Unconfirmed retained index must not trust stale missing entries or persist .missing.
        with self.subTest(case='unconfirmed_retained_missing_to_present'):
            name = 'visit-unconfirmed'
            root = self.root/name
            write_v3(root, missing=(0, 0))
            expected_new, _ = write_v3(self.root/(name+'-new'), missing=None)
            replacement = bytearray((self.root/(name+'-new')/'c/0/0').read_bytes())
            replacement[0] ^= 1
            expected_new[0, 0] = int(expected_new[0, 0]) ^ 1

            cond_started = threading.Event()
            cond_release = threading.Event()
            hold = False

            def hook(handler):
                if hold and handler.path == f'/{name}/c/0/0' and handler.headers.get('If-Match'):
                    cond_started.set()
                    cond_release.wait(5)
                return False

            self.server.request_hook = hook
            uri = self.url + '/' + name
            cache = self.cache/'visit-1a'
            with ZarrReader.open(uri, cache_dir=cache, threads=2, memory_cache=None) as reader:
                _ = reader[0, 5]   # Visit A: chunk (0,0) is missing in A's index.
                _ = reader[0, 10]  # Visit B.
                (root/'c/0/0').write_bytes(replacement)
                hold = True
                with ThreadPoolExecutor(2) as pool:
                    fut = pool.submit(lambda: reader[3:6, 0:5])
                    self.assertTrue(cond_started.wait(5))
                    try:
                        actual_new = reader[0:3, 0:5]
                    finally:
                        cond_release.set()
                    np.testing.assert_array_equal(fut.result(timeout=10), expected_new[3:6, 0:5])
                np.testing.assert_array_equal(actual_new, expected_new[0:3, 0:5])
            with ZarrReader.open(uri, cache_dir=cache, offline=True, memory_cache=None) as offline:
                np.testing.assert_array_equal(offline[0:3, 0:5], expected_new[0:3, 0:5])

        # 1b. A pending visit to B must immediately demote previously trusted shard A.
        with self.subTest(case='pending_visit_demotes_previous_shard'):
            name = 'visit-pending'
            root = self.root/name
            write_v3(root, missing=(0, 0))
            expected_new, _ = write_v3(self.root/(name+'-new'), missing=None)
            replacement = bytearray((self.root/(name+'-new')/'c/0/0').read_bytes())
            replacement[0] ^= 2
            expected_new[0, 0] = int(expected_new[0, 0]) ^ 2

            b_started = threading.Event()
            b_release = threading.Event()
            hold_b = False

            def hook(handler):
                if hold_b and handler.path == f'/{name}/c/0/1' and handler.headers.get('If-Match'):
                    b_started.set()
                    b_release.wait(5)
                return False

            self.server.request_hook = hook
            uri = self.url + '/' + name
            cache = self.cache/'visit-1b'
            with ZarrReader.open(uri, cache_dir=cache, threads=2, memory_cache=None) as reader:
                _ = reader[0, 10]  # Visit B (retains B).
                _ = reader[0, 5]   # Visit A (A is confirmed active; (0,0) is missing).
                (root/'c/0/0').write_bytes(replacement)
                hold_b = True
                with ThreadPoolExecutor(2) as pool:
                    fut_b = pool.submit(lambda: reader[3:6, 10:15])
                    self.assertTrue(b_started.wait(5))
                    try:
                        actual_a = reader[0:3, 0:5]
                    finally:
                        b_release.set()
                    fut_b.result(timeout=10)
                np.testing.assert_array_equal(actual_a, expected_new[0:3, 0:5])
            with ZarrReader.open(uri, cache_dir=cache, offline=True, memory_cache=None) as offline:
                np.testing.assert_array_equal(offline[0:3, 0:5], expected_new[0:3, 0:5])

        # 1c. Late out-of-order response after A -> B -> A(old) -> B -> A(new pending) must not restore trust.
        for follow_with_pending in (False, True):
            with self.subTest(case='late_response_obsolete_trust', follow_with_pending=follow_with_pending):
                name = f'visit-late-{follow_with_pending}'
                root = self.root/name
                write_v3(root, missing=(0, 0))
                expected_new, _ = write_v3(self.root/(name+'-new'), missing=None)
                replacement = bytearray((self.root/(name+'-new')/'c/0/0').read_bytes())
                replacement[0] ^= 4
                expected_new[0, 0] = int(expected_new[0, 0]) ^ 4

                old_bytes = (root/'c/0/0').read_bytes()
                old_tag = '"' + hashlib.sha256(old_bytes).hexdigest() + '"'
                old_started = threading.Event()
                old_release = threading.Event()
                pending_started = threading.Event()
                pending_release = threading.Event()
                intercept_old = False
                intercept_pending = False

                def hook(handler):
                    nonlocal intercept_old, intercept_pending
                    if (intercept_old and handler.path == f'/{name}/c/0/0' and
                            handler.headers.get('If-Match') == old_tag and
                            handler.headers.get('Range') == 'bytes=120-239'):
                        intercept_old = False
                        old_started.set()
                        old_release.wait(5)
                        handler.send_response(206)
                        handler.send_header('ETag', old_tag)
                        handler.send_header('Content-Range', f'bytes 120-239/{len(old_bytes)}')
                        handler.send_header('Content-Length', '120')
                        handler.end_headers()
                        handler.wfile.write(old_bytes[120:240])
                        return True
                    if (intercept_pending and handler.path == f'/{name}/c/0/0' and
                            handler.headers.get('If-Match') == old_tag and
                            handler.headers.get('Range') == 'bytes=240-359'):
                        intercept_pending = False
                        pending_started.set()
                        pending_release.wait(5)
                        return False
                    return False

                self.server.request_hook = hook
                uri = self.url + '/' + name
                cache = self.cache/f'visit-1c-{follow_with_pending}'
                with ZarrReader.open(uri, cache_dir=cache, threads=2, memory_cache=None) as reader:
                    _ = reader[0, 5]   # Visit A (chunk (0,1), bytes 0..119).
                    _ = reader[0, 10]  # Visit B.
                    intercept_old = True
                    with ThreadPoolExecutor(3) as pool:
                        f_old = pool.submit(lambda: reader[3:6, 0:5])
                        self.assertTrue(old_started.wait(5))
                        _ = reader[3:6, 10:15]  # Newer visit to B.
                        (root/'c/0/0').write_bytes(replacement)
                        f_pending = None
                        try:
                            if follow_with_pending:
                                intercept_pending = True
                                f_pending = pool.submit(lambda: reader[3:6, 5:10])
                                self.assertTrue(pending_started.wait(5))
                            old_release.set()
                            f_old.result(timeout=10)
                            actual = reader[0:3, 0:5]
                        finally:
                            old_release.set()
                            pending_release.set()
                        if f_pending is not None:
                            np.testing.assert_array_equal(f_pending.result(timeout=10), expected_new[3:6, 5:10])
                    np.testing.assert_array_equal(actual, expected_new[0:3, 0:5])
                with ZarrReader.open(uri, cache_dir=cache, offline=True, memory_cache=None) as offline:
                    np.testing.assert_array_equal(offline[0:3, 0:5], expected_new[0:3, 0:5])

        # 1e. Retained-index conditional 206 (without intervening B visit) still must not promote A for missing entries,
        # and concurrent revisited readers on a replaced shard both recover cleanly without 412.
        with self.subTest(case='retained_conditional_never_trusts_missing_and_concurrent_revisit'):
            name = 'visit-r3-f1b'
            root = self.root/name
            write_v3(root, missing=(0, 0))
            expected_new, _ = write_v3(self.root/(name+'-new'), missing=None)
            replacement = bytearray((self.root/(name+'-new')/'c/0/0').read_bytes())
            replacement[0] ^= 8
            expected_new[0, 0] = int(expected_new[0, 0]) ^ 8
            uri = self.url + '/' + name
            cache = self.cache/'visit-1e'
            self.server.request_hook = None
            with ZarrReader.open(uri, cache_dir=cache, threads=2, memory_cache=None) as reader:
                _ = reader[0, 5]   # Visit A (chunk (0,0) missing).
                _ = reader[0, 10]  # Visit B.
                _ = reader[3:6, 0:5]  # Revisit A via retained index (conditional 206 succeeds on old A).
                (root/'c/0/0').write_bytes(replacement)  # Replace A without another B visit.
                # Missing entry (0,0) must still refresh A's index rather than trusting retained A!
                np.testing.assert_array_equal(reader[0:3, 0:5], expected_new[0:3, 0:5])
            with ZarrReader.open(uri, cache_dir=cache, offline=True, memory_cache=None) as offline:
                np.testing.assert_array_equal(offline[0:3, 0:5], expected_new[0:3, 0:5])

            # Concurrent revisited readers on a replaced shard (Defect F1-B).
            name2 = 'visit-f1b'
            root2 = self.root/name2
            write_v3(root2, missing=None)
            expected2, _ = write_v3(self.root/(name2+'-new'), missing=None)
            repl2 = bytearray((self.root/(name2+'-new')/'c/0/0').read_bytes())
            repl2[120] ^= 0x11
            repl2[240] ^= 0x22
            expected2[0, 5] = int(expected2[0, 5]) ^ 0x11
            expected2[3, 0] = int(expected2[3, 0]) ^ 0x22
            barrier = threading.Barrier(2, timeout=5)

            def sync_412(handler):
                if handler.path == f'/{name2}/c/0/0' and handler.headers.get('If-Match') and \
                        handler.headers.get('Range') in ('bytes=120-239', 'bytes=240-359'):
                    try:
                        barrier.wait()
                    except threading.BrokenBarrierError:
                        pass
                return False

            with ZarrReader.open(self.url + '/' + name2, threads=2, memory_cache=None) as reader:
                _ = reader[0, 0]   # Visit A.
                _ = reader[0, 10]  # Visit B.
                (root2/'c/0/0').write_bytes(bytes(repl2))
                self.server.request_hook = sync_412
                with ThreadPoolExecutor(2) as pool:
                    fa = pool.submit(lambda: reader[0:3, 5:10])
                    fb = pool.submit(lambda: reader[3:6, 0:5])
                    np.testing.assert_array_equal(fa.result(timeout=10), expected2[0:3, 5:10])
                    np.testing.assert_array_equal(fb.result(timeout=10), expected2[3:6, 0:5])

        # 1d. Invalid offset in a retained index must fall through to fetch a repaired shard.
        for batched in (False, True):
            with self.subTest(case='invalid_retained_offset_repaired', batched=batched):
                name = f'visit-invalid-{batched}'
                root = self.root/name
                write_v3(root, missing=None)
                expected_repaired, _ = write_v3(self.root/(name+'-repaired'), missing=None)
                repaired = bytearray((self.root/(name+'-repaired')/'c/0/0').read_bytes())
                repaired[240] ^= 1
                expected_repaired[3, 0] = int(expected_repaired[3, 0]) ^ 1

                shard_path = root/'c/0/0'
                broken = bytearray(shard_path.read_bytes())
                idx_start = len(broken) - 68
                broken[idx_start + 32 : idx_start + 40] = (2**63).to_bytes(8, 'little')
                broken[-4:] = crc32c(broken[idx_start:-4]).to_bytes(4, 'little')
                shard_path.write_bytes(bytes(broken))

                self.server.request_hook = None
                with ZarrReader.open(self.url + '/' + name, threads=2, memory_cache=None) as reader:
                    _ = reader[0, 0]   # Visit A (valid chunk (0,0)), caches A with broken entry (1,0).
                    _ = reader[0, 10]  # Visit B.
                    shard_path.write_bytes(bytes(repaired))
                    selection = np.s_[3:6, :10] if batched else np.s_[3:6, 0:5]
                    np.testing.assert_array_equal(reader[selection], expected_repaired[selection])

    def test_bounded_per_shard_recovery_and_termination(self):
        # 2a. Three independently replaced shards in one batch with threads=1 recover cleanly,
        # while an untouched fourth shard keeps its retained index without re-fetching.
        name = 'recover-three'
        root = self.root/name
        expected_old, _ = write_v3(root, missing=None)
        expected_new, _ = write_v3(self.root/(name+'-new'), missing=None)
        for rel, coord in (('c/0/0', (3, 5)), ('c/0/1', (3, 15)), ('c/1/0', (9, 5))):
            raw = bytearray((self.root/(name+'-new')/rel).read_bytes())
            raw[360] ^= 0x55
            (self.root/(name+'-new')/rel).write_bytes(bytes(raw))
            expected_new[coord] = int(expected_new[coord]) ^ 0x55

        index_gets = {}

        def capture(handler):
            if handler.headers.get('Range') == 'bytes=-68':
                index_gets[handler.path] = index_gets.get(handler.path, 0) + 1
            return False

        self.server.request_hook = capture
        with ZarrReader.open(self.url + '/' + name, threads=1, memory_cache=None) as reader:
            for coord in ((0, 0), (0, 10), (6, 0), (6, 10)):
                _ = reader[coord]
            for rel in ('c/0/0', 'c/0/1', 'c/1/0'):
                (root/rel).write_bytes((self.root/(name+'-new')/rel).read_bytes())
            selections = [np.s_[3:6, 5:10], np.s_[3:6, 15:20], np.s_[9:12, 5:10], np.s_[9:12, 15:20]]
            results = reader.read_many(selections)
            for actual, sel in zip(results[:3], selections[:3]):
                np.testing.assert_array_equal(actual, expected_new[sel])
            np.testing.assert_array_equal(results[3], expected_old[selections[3]])
        self.assertEqual(index_gets[f'/{name}/c/0/0'], 2)
        self.assertEqual(index_gets[f'/{name}/c/0/1'], 2)
        self.assertEqual(index_gets[f'/{name}/c/1/0'], 2)
        self.assertEqual(index_gets[f'/{name}/c/1/1'], 1)

        # 2b. A shard that repeatedly returns 412 terminates after one refresh while healthy shards succeed and cache.
        name = 'recover-bound'
        root = self.root/name
        expected_old, _ = write_v3(root, missing=None)
        expected_new, _ = write_v3(self.root/(name+'-new'), missing=None)
        raw01 = bytearray((self.root/(name+'-new')/'c/0/1').read_bytes())
        raw01[120] ^= 0x33
        expected_new[0, 15] = int(expected_new[0, 15]) ^ 0x33
        (self.root/(name+'-new')/'c/0/1').write_bytes(bytes(raw01))

        churn_index_gets = 0
        churn_payload_gets = 0
        force_412 = False

        def churn_hook(handler):
            nonlocal churn_index_gets, churn_payload_gets
            if handler.path == f'/{name}/c/0/0':
                if handler.headers.get('Range') == 'bytes=-68':
                    churn_index_gets += 1
                    return False
                if force_412 and handler.headers.get('If-Match'):
                    churn_payload_gets += 1
                    handler.send_error(412)
                    return True
            return False

        self.server.request_hook = churn_hook
        storage = MemoryCache(4096)
        with ZarrReader.open(self.url + '/' + name, threads=1, memory_cache=storage) as reader:
            _ = reader[0, 0]
            _ = reader[0, 10]
            (root/'c/0/1').write_bytes((self.root/(name+'-new')/'c/0/1').read_bytes())
            force_412 = True
            with self.assertRaisesRegex(RuntimeError, '412'):
                reader.read_many([np.s_[3:6, 0:5], np.s_[0:3, 15:20]])
            # Bounded: initial prime (1) + at most 1 refresh (1) = 2 index fetches, 2 failed payload attempts.
            self.assertEqual(churn_index_gets, 2)
            self.assertEqual(churn_payload_gets, 2)
            fetches_before = reader.stats()['remote_fetches']
            np.testing.assert_array_equal(reader[0:3, 15:20], expected_new[0:3, 15:20])
            self.assertEqual(reader.stats()['remote_fetches'], fetches_before)

    def test_per_chunk_error_isolation_and_singleflight(self):
        # 3a. Held invalid Content-Range on shard 0/0 must not block or fail concurrent SingleFlight reader of shard 0/1.
        with self.subTest(case='held_invalid_content_range_concurrent_singleflight'):
            name = 'iso-content-range'
            expected, _ = write_v3(self.root/name, missing=None)
            shard0 = (self.root/name/'c/0/0').read_bytes()
            bad_held = threading.Event()
            bad_release = threading.Event()

            def hook(handler):
                if handler.path == f'/{name}/c/0/0' and handler.headers.get('Range') == 'bytes=0-119':
                    bad_held.set()
                    bad_release.wait(5)
                    handler.send_response(206)
                    handler.send_header('Content-Range', f'bytes 1-119/{len(shard0)}')
                    handler.send_header('Content-Length', '120')
                    handler.end_headers()
                    handler.wfile.write(shard0[:120])
                    return True
                return False

            self.server.request_hook = hook
            uri = self.url + '/' + name
            storage = MemoryCache(4096)
            with ZarrReader.open(uri, threads=2, memory_cache=storage) as r1, \
                    ZarrReader.open(uri, threads=2, memory_cache=storage) as r2, \
                    ThreadPoolExecutor(2) as pool:
                f1 = pool.submit(lambda: r1.read_many([np.s_[0:3, 0:5], np.s_[0:3, 10:15]]))
                self.assertTrue(bad_held.wait(5))
                try:
                    f2 = pool.submit(lambda: r2[0:3, 10:15])
                    # Must complete BEFORE bad_release is set.
                    np.testing.assert_array_equal(f2.result(timeout=5), expected[0:3, 10:15])
                finally:
                    bad_release.set()
                with self.assertRaisesRegex(RuntimeError, 'Content-Range'):
                    f1.result(timeout=10)
                fetches_before = r2.stats()['remote_fetches']
                np.testing.assert_array_equal(r2[0:3, 10:15], expected[0:3, 10:15])
                self.assertEqual(r2.stats()['remote_fetches'], fetches_before)

        # 3b. Bad index offset in shard 0/0 must isolate to that chunk, leaving sibling chunk and shard 0/1 healthy & cached.
        with self.subTest(case='invalid_index_offset_isolated'):
            name = 'iso-bad-offset'
            expected, _ = write_v3(self.root/name, missing=None)
            shard0_path = self.root/name/'c/0/0'
            broken = bytearray(shard0_path.read_bytes())
            idx_start = len(broken) - 68
            broken[idx_start : idx_start + 8] = (2**63).to_bytes(8, 'little')
            broken[-4:] = crc32c(broken[idx_start:-4]).to_bytes(4, 'little')
            shard0_path.write_bytes(bytes(broken))

            self.server.request_hook = None
            storage = MemoryCache(4096)
            with ZarrReader.open(self.url + '/' + name, threads=2, memory_cache=storage) as reader:
                with self.assertRaisesRegex(RuntimeError, 'offset/length'):
                    reader.read_many([np.s_[0:3, 0:5], np.s_[0:3, 5:10], np.s_[0:3, 10:15]])
                fetches_before = reader.stats()['remote_fetches']
                np.testing.assert_array_equal(reader[0:3, 5:10], expected[0:3, 5:10])
                np.testing.assert_array_equal(reader[0:3, 10:15], expected[0:3, 10:15])
                self.assertEqual(reader.stats()['remote_fetches'], fetches_before)

        # 3c. HTTP 416 merged-range fallback isolates bad individual chunk from healthy sibling and concurrent reader.
        with self.subTest(case='http_416_fallback_isolated'):
            name = 'iso-416'
            expected, _ = write_v3(self.root/name, missing=None)
            shard0 = (self.root/name/'c/0/0').read_bytes()
            bad_held = threading.Event()
            bad_release = threading.Event()

            def hook(handler):
                if handler.path != f'/{name}/c/0/0':
                    return False
                rng = handler.headers.get('Range')
                if rng == 'bytes=0-239':
                    handler.send_error(416)
                    return True
                if rng == 'bytes=0-119':
                    bad_held.set()
                    bad_release.wait(5)
                    handler.send_response(206)
                    handler.send_header('Content-Range', f'bytes 1-119/{len(shard0)}')
                    handler.send_header('Content-Length', '120')
                    handler.end_headers()
                    handler.wfile.write(shard0[:120])
                    return True
                return False

            self.server.request_hook = hook
            uri = self.url + '/' + name
            storage = MemoryCache(4096)
            with ZarrReader.open(uri, threads=2, memory_cache=storage) as r1, \
                    ZarrReader.open(uri, threads=2, memory_cache=storage) as r2, \
                    ThreadPoolExecutor(2) as pool:
                f1 = pool.submit(lambda: r1.read_many([np.s_[0:3, 0:5], np.s_[0:3, 5:10]]))
                self.assertTrue(bad_held.wait(5))
                try:
                    f2 = pool.submit(lambda: r2[0:3, 5:10])
                    np.testing.assert_array_equal(f2.result(timeout=5), expected[0:3, 5:10])
                finally:
                    bad_release.set()
                with self.assertRaisesRegex(RuntimeError, 'Content-Range'):
                    f1.result(timeout=10)
                fetches_before = r2.stats()['remote_fetches']
                np.testing.assert_array_equal(r2[0:3, 5:10], expected[0:3, 5:10])
                self.assertEqual(r2.stats()['remote_fetches'], fetches_before)

        # 3d. Individual HTTP 400 failure on one shard does not poison healthy shard.
        with self.subTest(case='individual_request_failure_isolated'):
            name = 'iso-http-400'
            expected, _ = write_v3(self.root/name, missing=None)

            def hook(handler):
                if handler.path == f'/{name}/c/0/0' and handler.headers.get('Range') == 'bytes=0-119':
                    handler.send_error(400)
                    return True
                return False

            self.server.request_hook = hook
            storage = MemoryCache(4096)
            with ZarrReader.open(self.url + '/' + name, threads=2, memory_cache=storage) as reader:
                with self.assertRaisesRegex(RuntimeError, 'HTTP 400'):
                    reader.read_many([np.s_[0:3, 0:5], np.s_[0:3, 10:15]])
                fetches_before = reader.stats()['remote_fetches']
                np.testing.assert_array_equal(reader[0:3, 10:15], expected[0:3, 10:15])
                self.assertEqual(reader.stats()['remote_fetches'], fetches_before)

    def test_partial_raw_cache_reads_slab_and_mmap(self):
        name = 'partial-raw'
        data = np.arange(64 * 64 * 128, dtype=np.uint32).astype(np.uint8).reshape(64, 64, 128)
        writer = ZarrWriter.create(self.root/name, shape=data.shape, chunks=(64, 64, 64),
                                   dtype='u1', version=3, shards=data.shape, compressor=None, separator='/')
        writer[:] = data
        uri = self.url + '/' + name
        cache = self.cache/'partial-raw-cache'

        # Warm the raw disk cache with both 256 KiB chunks.
        with ZarrReader.open(uri, cache_dir=cache, threads=2, memory_cache=None) as reader:
            warm = reader.read_many([np.s_[0:1, 0:1, 0:1], np.s_[0:1, 0:1, 64:65]])
            self.assertEqual(warm[0][0, 0, 0], data[0, 0, 0])
            self.assertEqual(warm[1][0, 0, 0], data[0, 0, 64])

        sels = [np.s_[10:11, 20:21, 30:31], np.s_[10:11, 20:21, 94:95]]
        proc_io = Path('/proc/self/io')

        def rchar():
            for line in proc_io.read_text().splitlines():
                if line.startswith('rchar:'):
                    return int(line.split(':')[1].strip())
            return None

        for mode in ('slab', 'mmap'):
            for offline in (False, True):
                with self.subTest(mode=mode, offline=offline):
                    with ZarrReader.open(uri, cache_dir=cache, threads=1, memory_cache=None,
                                         read_mode=mode, offline=offline) as reader:
                        # First call warms mmap cache / scratch buffer.
                        out = reader.read_many(sels)
                        self.assertEqual(out[0][0, 0, 0], data[10, 20, 30])
                        self.assertEqual(out[1][0, 0, 0], data[10, 20, 94])
                        self.assertEqual(reader.stats()['remote_fetches'], 0)
                        if proc_io.exists():
                            # Measure 1 iteration vs 21 iterations to cancel /proc/self/io read overhead.
                            c0 = rchar()
                            reader.read_many(sels)
                            c1 = rchar()
                            for _ in range(21):
                                reader.read_many(sels)
                            c2 = rchar()
                            delta_20 = (c2 - c1) - (c1 - c0)
                            bytes_per_call = delta_20 / 20.0
                            if mode == 'slab':
                                self.assertLessEqual(bytes_per_call, 16.0)
                            else:
                                self.assertLessEqual(bytes_per_call, 16.0)

        # Mixed warm + cold batch preserves partial raw hit and still batches cold remote misses.
        mixed_name = 'partial-mixed'
        mixed_data = np.arange(64 * 64 * 192, dtype=np.uint32).astype(np.uint8).reshape(64, 64, 192)
        writer = ZarrWriter.create(self.root/mixed_name, shape=mixed_data.shape, chunks=(64, 64, 64),
                                   dtype='u1', version=3, shards=mixed_data.shape, compressor=None, separator='/')
        writer[:] = mixed_data
        mixed_uri = self.url + '/' + mixed_name
        mixed_cache = self.cache/'partial-mixed-cache'
        with ZarrReader.open(mixed_uri, cache_dir=mixed_cache, threads=2, memory_cache=None) as reader:
            # Warm only chunk 0.
            self.assertEqual(reader[0, 0, 0], mixed_data[0, 0, 0])
        ranges = []

        def capture_mixed(handler):
            if handler.path == f'/{mixed_name}/c/0/0/0' and handler.headers.get('Range'):
                ranges.append(handler.headers['Range'])
            return False

        self.server.request_hook = capture_mixed
        with ZarrReader.open(mixed_uri, cache_dir=mixed_cache, threads=2, memory_cache=None, read_mode='slab') as reader:
            out = reader.read_many([np.s_[1:2, 2:3, 3:4], np.s_[1:2, 2:3, 67:68], np.s_[1:2, 2:3, 131:132]])
            self.assertEqual(out[0][0, 0, 0], mixed_data[1, 2, 3])
            self.assertEqual(out[1][0, 0, 0], mixed_data[1, 2, 67])
            self.assertEqual(out[2][0, 0, 0], mixed_data[1, 2, 131])
        # Chunk 0 was warm on disk; chunks 1 and 2 coalesced into one 512 KiB range!
        self.assertEqual(len(ranges), 2)
        self.assertIn('bytes=262144-786431', ranges)

    def test_full_read_revision_consistency_across_windows_and_ranges(self):
        # 1. Cross-plan (2 windows) two-shard interleaved replacement fails fast instead of mixing revisions.
        with self.subTest(case='cross_plan_two_shard_interleaved'):
            name = 'rev-cross-plan-interleaved'
            v1_data = np.arange(20 * 2 * 4, dtype=np.uint8).reshape(20, 2, 4)
            v2_data = (v1_data.astype(np.uint16) + 100).astype(np.uint8)
            write_custom_shard_array(self.root / name, v1_data, (1, 1, 4), (20, 1, 4))
            write_custom_shard_array(self.root / (name + '-v2'), v2_data, (1, 1, 4), (20, 1, 4))
            v1_s0 = (self.root / name / 'c/0/0/0').read_bytes()
            v1_s0_etag = '"' + hashlib.sha256(v1_s0).hexdigest() + '"'
            replaced = False
            c0_done = threading.Event()

            def hook(handler):
                nonlocal replaced
                if handler.path == f'/{name}/c/0/0/0' and handler.headers.get('Range') == 'bytes=0-3':
                    handler.send_response(206)
                    handler.send_header('Content-Range', f'bytes 0-3/{len(v1_s0)}')
                    handler.send_header('Content-Length', '4')
                    handler.send_header('ETag', v1_s0_etag)
                    handler.end_headers()
                    handler.wfile.write(v1_s0[:4])
                    c0_done.set()
                    return True
                if not replaced and handler.path == f'/{name}/c/0/1/0' and handler.headers.get('Range') == 'bytes=0-59':
                    c0_done.wait(5)
                    replaced = True
                    (self.root / name / 'c/0/0/0').write_bytes((self.root / (name + '-v2') / 'c/0/0/0').read_bytes())
                return False

            self.server.request_hook = hook
            sels = [np.s_[0:1, 0:1, 0:4]] + [np.s_[y:y + 1, 1:2, 0:4] for y in range(15)] + [np.s_[16:17, 0:1, 0:4]]
            with ZarrReader.open(self.url + '/' + name, threads=1, memory_cache=None) as reader:
                with self.assertRaisesRegex(RuntimeError, '412|changed'):
                    reader.read_many(sels)
            self.assertTrue(replaced)

        # 2. Cross-plan single-shard retained replacement must fail Window 2 after Window 1 commits.
        with self.subTest(case='cross_plan_single_shard_retained'):
            name = 'rev-cross-plan-retained'
            v1_data = np.arange(20 * 4, dtype=np.uint8).reshape(20, 4)
            v2_data = (v1_data.astype(np.uint16) + 77).astype(np.uint8)
            write_custom_shard_array(self.root / name, v1_data, (1, 4), (20, 4))
            write_custom_shard_array(self.root / (name + '-v2'), v2_data, (1, 4), (20, 4))
            replaced = False
            suffix_gets = 0

            def hook(handler):
                nonlocal replaced, suffix_gets
                if handler.path == f'/{name}/c/0/0':
                    rng = handler.headers.get('Range')
                    if rng == 'bytes=-324':
                        suffix_gets += 1
                    elif not replaced and rng == 'bytes=4-67':
                        replaced = True
                        # Replace on disk right after Window 1's merged range starts; serve v1 for Window 1.
                        v1_bytes = (self.root / name / 'c/0/0').read_bytes()
                        (self.root / name / 'c/0/0').write_bytes((self.root / (name + '-v2') / 'c/0/0').read_bytes())
                        etag = '"' + hashlib.sha256(v1_bytes).hexdigest() + '"'
                        handler.send_response(206)
                        handler.send_header('Content-Range', f'bytes 4-67/{len(v1_bytes)}')
                        handler.send_header('Content-Length', '64')
                        handler.send_header('ETag', etag)
                        handler.end_headers()
                        handler.wfile.write(v1_bytes[4:68])
                        return True
                return False

            self.server.request_hook = hook
            with ZarrReader.open(self.url + '/' + name, threads=1, memory_cache=None) as reader:
                np.testing.assert_array_equal(reader[0:1, 0:4], v1_data[0:1, 0:4])
                sels = [np.s_[y:y + 1, 0:4] for y in range(1, 18)]
                with self.assertRaisesRegex(RuntimeError, '412|changed'):
                    reader.read_many(sels)
            self.assertTrue(replaced)
            self.assertEqual(suffix_gets, 1)

        # 3. Intra-plan commit-once two-range split prevents mixed (v1, v2) and contradictory offline .missing.
        with self.subTest(case='intra_plan_commit_once_two_ranges'):
            # Build 3-chunk shard (chunk 2 primes retained index, chunks 0 & 1 are uncached and >512 KiB combined).
            name = 'rev-intra-plan-3chunk'
            v1_data = np.full((64, 64, 240), 11, dtype=np.uint8)
            v2_data = np.full((64, 64, 240), 22, dtype=np.uint8)
            write_custom_shard_array(self.root / name, v1_data, (64, 64, 80), (64, 64, 240))
            write_custom_shard_array(self.root / (name + '-v2'), v2_data, (64, 64, 80), (64, 64, 240),
                                     missing_chunks=[(0, 0, 1)])
            v1_bytes = (self.root / name / 'c/0/0/0').read_bytes()
            v1_etag = '"' + hashlib.sha256(v1_bytes).hexdigest() + '"'
            v2_bytes = (self.root / (name + '-v2') / 'c/0/0/0').read_bytes()
            cache_dir = self.cache / 'rev-intra-plan-3chunk-cache'
            suffix_gets = 0
            replaced = False
            r0_swapped = threading.Event()

            def hook3(handler):
                nonlocal suffix_gets, replaced
                if handler.path == f'/{name}/c/0/0/0':
                    rng = handler.headers.get('Range')
                    if rng == 'bytes=-52':
                        suffix_gets += 1
                    elif rng == 'bytes=0-327679' and not replaced:
                        replaced = True
                        (self.root / name / 'c/0/0/0').write_bytes(v2_bytes)
                        r0_swapped.set()
                        handler.send_response(206)
                        handler.send_header('Content-Range', f'bytes 0-327679/{len(v1_bytes)}')
                        handler.send_header('Content-Length', '327680')
                        handler.send_header('ETag', v1_etag)
                        handler.end_headers()
                        handler.wfile.write(v1_bytes[:327680])
                        return True
                    elif rng == 'bytes=327680-655359':
                        r0_swapped.wait(5)
                return False

            self.server.request_hook = hook3
            uri = self.url + '/' + name
            with ZarrReader.open(uri, cache_dir=cache_dir, threads=1, memory_cache=None) as reader:
                self.assertEqual(reader[0, 0, 160], 11)
                with self.assertRaisesRegex(RuntimeError, '412|changed'):
                    reader.read_many([np.s_[0:1, 0:1, 0:1], np.s_[0:1, 0:1, 80:81]])
            self.assertTrue(replaced)
            self.assertEqual(suffix_gets, 1)
            # Offline reopen: chunk 0 is v1, chunk 1 has NO contradictory .missing marker!
            self.server.request_hook = None
            with ZarrReader.open(uri, cache_dir=cache_dir, threads=1, memory_cache=None,
                                 offline=True) as offline_reader:
                self.assertEqual(offline_reader[0, 0, 0], 11)
                with self.assertRaises(CacheMissError):
                    _ = offline_reader[0, 0, 80]

        # 3b. HTTP 416 fallback: once member 0 commits on v1, member 1 receiving 412 MUST fail fast (no mixed v1/v2).
        with self.subTest(case='http_416_fallback_commit_once_no_mixed_revisions'):
            name = 'rev-416-commit-once'
            v1_data = np.full((64, 64, 128), 41, dtype=np.uint8)
            v2_data = np.full((64, 64, 128), 82, dtype=np.uint8)
            write_custom_shard_array(self.root / name, v1_data, (64, 64, 64), (64, 64, 128))
            write_custom_shard_array(self.root / (name + '-v2'), v2_data, (64, 64, 64), (64, 64, 128))
            v1_bytes = (self.root / name / 'c/0/0/0').read_bytes()
            v1_etag = '"' + hashlib.sha256(v1_bytes).hexdigest() + '"'
            v2_bytes = (self.root / (name + '-v2') / 'c/0/0/0').read_bytes()
            replaced = False
            m0_swapped = threading.Event()

            def hook416(handler):
                nonlocal replaced
                if handler.path != f'/{name}/c/0/0/0':
                    return False
                rng = handler.headers.get('Range')
                if rng == 'bytes=0-524287':
                    handler.send_error(416)
                    return True
                if rng == 'bytes=0-262143' and not replaced:
                    replaced = True
                    (self.root / name / 'c/0/0/0').write_bytes(v2_bytes)
                    m0_swapped.set()
                    handler.send_response(206)
                    handler.send_header('Content-Range', f'bytes 0-262143/{len(v1_bytes)}')
                    handler.send_header('Content-Length', '262144')
                    handler.send_header('ETag', v1_etag)
                    handler.end_headers()
                    handler.wfile.write(v1_bytes[:262144])
                    return True
                if rng == 'bytes=262144-524287':
                    m0_swapped.wait(5)
                return False

            self.server.request_hook = hook416
            storage = MemoryCache(1 << 20)
            with ZarrReader.open(self.url + '/' + name, threads=1, memory_cache=storage) as reader:
                with self.assertRaisesRegex(RuntimeError, '412|changed'):
                    reader.read_many([np.s_[0:1, 0:1, 0:1], np.s_[0:1, 0:1, 64:65]])
                # Member 0 (v1) is already cached cleanly without double-delivery or abandoned claim.
                fetches_before = reader.stats()['remote_fetches']
                self.assertEqual(reader[0, 0, 0], 41)
                self.assertEqual(reader.stats()['remote_fetches'], fetches_before)
            self.assertTrue(replaced)

        # 4. Cross-plan offline missing-to-present contradiction prevented.
        with self.subTest(case='cross_plan_offline_missing_contradiction'):
            name = 'rev-cross-plan-missing'
            v1_data = np.full((20, 2, 4), 33, dtype=np.uint8)
            v2_data = np.full((20, 2, 4), 44, dtype=np.uint8)
            write_custom_shard_array(self.root / name, v1_data, (1, 1, 4), (20, 1, 4),
                                     missing_chunks=[(16, 0, 0)])
            write_custom_shard_array(self.root / (name + '-v2'), v2_data, (1, 1, 4), (20, 1, 4))
            v1_s0 = (self.root / name / 'c/0/0/0').read_bytes()
            v1_s0_etag = '"' + hashlib.sha256(v1_s0).hexdigest() + '"'
            cache_dir = self.cache / 'rev-cross-plan-missing-cache'
            replaced = False
            c0_done = threading.Event()

            def hook(handler):
                nonlocal replaced
                if handler.path == f'/{name}/c/0/0/0' and handler.headers.get('Range') == 'bytes=0-3':
                    handler.send_response(206)
                    handler.send_header('Content-Range', f'bytes 0-3/{len(v1_s0)}')
                    handler.send_header('Content-Length', '4')
                    handler.send_header('ETag', v1_s0_etag)
                    handler.end_headers()
                    handler.wfile.write(v1_s0[:4])
                    c0_done.set()
                    return True
                if not replaced and handler.path == f'/{name}/c/0/1/0' and handler.headers.get('Range') == 'bytes=0-59':
                    c0_done.wait(5)
                    replaced = True
                    (self.root / name / 'c/0/0/0').write_bytes((self.root / (name + '-v2') / 'c/0/0/0').read_bytes())
                return False

            self.server.request_hook = hook
            uri = self.url + '/' + name
            sels = [np.s_[0:1, 0:1, 0:4]] + [np.s_[y:y + 1, 1:2, 0:4] for y in range(15)] + [np.s_[16:17, 0:1, 0:4]]
            with ZarrReader.open(uri, cache_dir=cache_dir, threads=1, memory_cache=None) as reader:
                with self.assertRaisesRegex(RuntimeError, '412|changed'):
                    reader.read_many(sels)
            self.assertTrue(replaced)
            self.server.request_hook = None
            with ZarrReader.open(uri, cache_dir=cache_dir, threads=1, memory_cache=None,
                                 offline=True) as offline_reader:
                np.testing.assert_array_equal(offline_reader[0:1, 0:1, 0:4], np.full((1, 1, 4), 33, dtype=np.uint8))
                with self.assertRaises(CacheMissError):
                    _ = offline_reader[16:17, 0:1, 0:4]

        # 5. Foreign SingleFlight claim with different shard revision fails Reader A without blocking Reader B.
        with self.subTest(case='foreign_singleflight_revision_conflict'):
            name = 'rev-foreign-sf'
            v1_data = np.full((2, 4), 51, dtype=np.uint8)
            v2_data = np.full((2, 4), 62, dtype=np.uint8)
            write_custom_shard_array(self.root / name, v1_data, (1, 4), (2, 4))
            write_custom_shard_array(self.root / (name + '-v2'), v2_data, (1, 4), (2, 4))
            v1_bytes = (self.root / name / 'c/0/0').read_bytes()
            v1_etag = '"' + hashlib.sha256(v1_bytes).hexdigest() + '"'
            v2_bytes = (self.root / (name + '-v2') / 'c/0/0').read_bytes()
            b_holding = threading.Event()
            release_b = threading.Event()

            def hook(handler):
                if handler.path == f'/{name}/c/0/0':
                    rng = handler.headers.get('Range')
                    if rng == 'bytes=0-3':
                        # Reader B fetched v1 index and is now reading chunk 0; swap disk to v2 before waking Reader A.
                        (self.root / name / 'c/0/0').write_bytes(v2_bytes)
                        b_holding.set()
                        release_b.wait(5)
                        handler.send_response(206)
                        handler.send_header('Content-Range', f'bytes 0-3/{len(v1_bytes)}')
                        handler.send_header('Content-Length', '4')
                        handler.send_header('ETag', v1_etag)
                        handler.end_headers()
                        handler.wfile.write(v1_bytes[0:4])
                        return True
                    if rng == 'bytes=4-7':
                        # Reader A is fetching chunk 1 from v2; let Reader B finish chunk 0 on v1.
                        release_b.set()
                return False

            self.server.request_hook = hook
            uri = self.url + '/' + name
            with ZarrReader.open(uri, threads=2, memory_cache=None) as rb, \
                    ZarrReader.open(uri, threads=2, memory_cache=None) as ra, \
                    ThreadPoolExecutor(2) as pool:
                fb = pool.submit(lambda: rb[0:1, 0:4])
                self.assertTrue(b_holding.wait(5))
                try:
                    fa = pool.submit(lambda: ra.read_many([np.s_[0:1, 0:4], np.s_[1:2, 0:4]]))
                    with self.assertRaisesRegex(RuntimeError, 'changed|412'):
                        fa.result(timeout=10)
                    np.testing.assert_array_equal(fb.result(timeout=5), v1_data[0:1, 0:4])
                finally:
                    release_b.set()

        # 6. Weak and absent ETags succeed across multiple windows when unchanged, and fail when mutated.
        for tag_mode in ('weak', 'absent'):
            with self.subTest(case=f'etag_{tag_mode}_multi_window'):
                name = f'rev-etag-{tag_mode}'
                v1_data = np.arange(20 * 2 * 4, dtype=np.uint8).reshape(20, 2, 4)
                v2_data = (v1_data.astype(np.uint16) + 90).astype(np.uint8)
                write_custom_shard_array(self.root / name, v1_data, (1, 1, 4), (20, 1, 4))
                # Make v2 have a missing entry in unused chunk (19, 0, 0) so index bytes mutate even if total size differs or matches.
                write_custom_shard_array(self.root / (name + '-v2'), v2_data, (1, 1, 4), (20, 1, 4),
                                         missing_chunks=[(19, 0, 0)])
                mutate = False
                mutated = False
                c0_served_in_mutate = threading.Event()

                def serve_etag(handler):
                    nonlocal mutated
                    if not handler.path.startswith(f'/{name}/c/'):
                        return False
                    rel = handler.path[len(f'/{name}/'):]
                    fpath = self.root / name / rel
                    rng = handler.headers.get('Range')
                    if mutate and rel == 'c/0/1/0' and rng == 'bytes=56-59' and not mutated:
                        c0_served_in_mutate.wait(5)
                        mutated = True
                        (self.root / name / 'c/0/0/0').write_bytes((self.root / (name + '-v2') / 'c/0/0/0').read_bytes())
                    raw = fpath.read_bytes()
                    if rng and rng.startswith('bytes='):
                        spec = rng[6:]
                        if spec.startswith('-'):
                            n = int(spec[1:])
                            start, end = len(raw) - n, len(raw) - 1
                        else:
                            a, b = spec.split('-')
                            start, end = int(a), int(b)
                        chunk = raw[start:end + 1]
                        handler.send_response(206)
                        handler.send_header('Content-Range', f'bytes {start}-{end}/{len(raw)}')
                        handler.send_header('Content-Length', str(len(chunk)))
                        if tag_mode == 'weak':
                            handler.send_header('ETag', 'W/"' + hashlib.sha256(raw).hexdigest()[:16] + '"')
                        handler.end_headers()
                        handler.wfile.write(chunk)
                        if mutate and rel == 'c/0/0/0' and rng == 'bytes=0-3':
                            c0_served_in_mutate.set()
                        return True
                    return False

                self.server.request_hook = serve_etag
                sels = [np.s_[0:1, 0:1, 0:4]] + [np.s_[y:y + 1, 1:2, 0:4] for y in range(15)] + [np.s_[16:17, 0:1, 0:4]]
                with ZarrReader.open(self.url + '/' + name, threads=1, memory_cache=None) as reader:
                    out = reader.read_many(sels)
                    np.testing.assert_array_equal(out[0], v1_data[0:1, 0:1, 0:4])
                    np.testing.assert_array_equal(out[-1], v1_data[16:17, 0:1, 0:4])
                    mutate = True
                    with self.assertRaisesRegex(RuntimeError, 'changed|412'):
                        reader.read_many(sels)

        # 7. Non-batched sharded path (chunk_bytes > 512 KiB) enforces full-read revision pinning.
        with self.subTest(case='non_batched_large_chunk_sharded_consistency'):
            name = 'rev-non-batched-large'
            v1_data = np.full((64, 64, 260), 17, dtype=np.uint8)
            v2_data = np.full((64, 64, 260), 29, dtype=np.uint8)
            write_custom_shard_array(self.root / name, v1_data, (64, 64, 130), (64, 64, 260))
            write_custom_shard_array(self.root / (name + '-v2'), v2_data, (64, 64, 130), (64, 64, 260))
            v1_bytes = (self.root / name / 'c/0/0/0').read_bytes()
            v1_etag = '"' + hashlib.sha256(v1_bytes).hexdigest() + '"'
            v2_bytes = (self.root / (name + '-v2') / 'c/0/0/0').read_bytes()
            replaced = False
            c0_swapped = threading.Event()

            def hook(handler):
                nonlocal replaced
                if handler.path == f'/{name}/c/0/0/0':
                    rng = handler.headers.get('Range')
                    if rng == 'bytes=0-532479' and not replaced:
                        replaced = True
                        (self.root / name / 'c/0/0/0').write_bytes(v2_bytes)
                        c0_swapped.set()
                        handler.send_response(206)
                        handler.send_header('Content-Range', f'bytes 0-532479/{len(v1_bytes)}')
                        handler.send_header('Content-Length', '532480')
                        handler.send_header('ETag', v1_etag)
                        handler.end_headers()
                        handler.wfile.write(v1_bytes[:532480])
                        return True
                    if rng == 'bytes=532480-1064959':
                        c0_swapped.wait(5)
                return False

            self.server.request_hook = hook
            with ZarrReader.open(self.url + '/' + name, threads=1, memory_cache=None) as reader:
                with self.assertRaisesRegex(RuntimeError, '412|changed'):
                    reader.read_many([np.s_[0:1, 0:1, 0:1], np.s_[0:1, 0:1, 130:131]])
            self.assertTrue(replaced)

        # 8. Multi-shard pre-commit recovery across windows succeeds when all shards changed before read_many.
        with self.subTest(case='multi_shard_pre_commit_recovery_across_windows'):
            name = 'rev-multi-shard-precommit'
            v1_data = np.arange(10 * 3 * 4, dtype=np.uint8).reshape(10, 3, 4)
            v2_data = (v1_data.astype(np.uint16) + 123).astype(np.uint8)
            write_custom_shard_array(self.root / name, v1_data, (1, 1, 4), (10, 1, 4))
            write_custom_shard_array(self.root / (name + '-v2'), v2_data, (1, 1, 4), (10, 1, 4))
            self.server.request_hook = None
            with ZarrReader.open(self.url + '/' + name, threads=1, memory_cache=None) as reader:
                # Prime all 3 shards into retained LRU.
                for sx in range(3):
                    np.testing.assert_array_equal(reader[0:1, sx:sx + 1, 0:4], v1_data[0:1, sx:sx + 1, 0:4])
                # Replace all 3 shards BEFORE read_many begins.
                for sx in range(3):
                    (self.root / name / f'c/0/{sx}/0').write_bytes(
                        (self.root / (name + '-v2') / f'c/0/{sx}/0').read_bytes())
                # Request 21 chunks across all 3 shards (2 batching windows).
                sels = [np.s_[y:y + 1, sx:sx + 1, 0:4] for y in range(1, 8) for sx in range(3)]
                out = reader.read_many(sels)
                for idx_s, sel in enumerate(sels):
                    np.testing.assert_array_equal(out[idx_s], v2_data[sel])

        self.server.request_hook = None

    def test_concurrent_non_batched_pre_read_refresh_and_termination(self):
        # 1. Large chunks (>512 KiB, non-batched path) with 4 workers hitting stale retained v1
        #    concurrently must share the single pre-commit refresh and all return v2.
        with self.subTest(case='large_chunks_multi_worker_shared_refresh'):
            name = 'nb-large-shared-refresh'
            v1_data = np.full((128, 64, 520), 11, dtype=np.uint8)
            v2_data = np.full((128, 64, 520), 22, dtype=np.uint8)
            write_custom_shard_array(self.root / name, v1_data, (64, 64, 130), (64, 64, 520))
            write_custom_shard_array(self.root / (name + '-v2'), v2_data, (64, 64, 130), (64, 64, 520))
            v1_etag = '"' + hashlib.sha256((self.root / name / 'c/0/0/0').read_bytes()).hexdigest() + '"'
            v2_bytes = (self.root / (name + '-v2') / 'c/0/0/0').read_bytes()

            barrier = threading.Barrier(3, timeout=5)
            v1_hits = 0
            v2_suffix_gets = 0
            hook_lock = threading.Lock()

            def hook(handler):
                nonlocal v1_hits, v2_suffix_gets
                if handler.path == f'/{name}/c/0/0/0':
                    rng = handler.headers.get('Range') or ''
                    if rng.startswith('bytes=-'):
                        with hook_lock:
                            v2_suffix_gets += 1
                    elif handler.headers.get('If-Match') == v1_etag:
                        with hook_lock:
                            v1_hits += 1
                        try:
                            barrier.wait()
                        except threading.BrokenBarrierError:
                            pass
                return False

            with ZarrReader.open(self.url + '/' + name, threads=4, memory_cache=None) as reader:
                self.assertEqual(int(reader[0, 0, 0]), 11)
                self.assertEqual(int(reader[64, 0, 0]), 11)
                (self.root / name / 'c/0/0/0').write_bytes(v2_bytes)
                self.server.request_hook = hook
                try:
                    out = reader.read_many([
                        np.s_[0:1, 0:1, 130:131],
                        np.s_[0:1, 0:1, 260:261],
                        np.s_[0:1, 0:1, 390:391],
                    ])
                finally:
                    self.server.request_hook = None
                self.assertEqual(v1_hits, 3)
                self.assertEqual(v2_suffix_gets, 1)
                self.assertEqual([int(x[0, 0, 0]) for x in out], [22, 22, 22])

        # 2. Cancellable prefill (non-batched path) shares refresh cleanly and respects cancellation.
        with self.subTest(case='cancellable_prefill_concurrent_refresh_and_cancel'):
            name = 'nb-prefill-refresh'
            v1_data = np.full((2, 4, 4), 31, dtype=np.uint8)
            v2_data = np.full((2, 4, 4), 62, dtype=np.uint8)
            write_custom_shard_array(self.root / name, v1_data, (1, 1, 4), (1, 4, 4))
            write_custom_shard_array(self.root / (name + '-v2'), v2_data, (1, 1, 4), (1, 4, 4))
            v1_etag = '"' + hashlib.sha256((self.root / name / 'c/0/0/0').read_bytes()).hexdigest() + '"'
            v2_bytes = (self.root / (name + '-v2') / 'c/0/0/0').read_bytes()
            cache_dir = self.cache / 'nb-prefill-cache'

            barrier = threading.Barrier(2, timeout=5)
            v1_hits = 0
            hook_lock = threading.Lock()

            def hook_prefill(handler):
                nonlocal v1_hits
                if handler.path == f'/{name}/c/0/0/0' and handler.headers.get('If-Match') == v1_etag:
                    with hook_lock:
                        v1_hits += 1
                    try:
                        barrier.wait()
                    except threading.BrokenBarrierError:
                        pass
                return False

            from vc_reader import _reader
            cancel = _reader.CacheCancellation()
            with ZarrReader.open(self.url + '/' + name, cache_dir=cache_dir, threads=2, memory_cache=None) as reader:
                self.assertEqual(int(reader[0, 0, 0]), 31)
                self.assertEqual(int(reader[1, 0, 0]), 31)
                (self.root / name / 'c/0/0/0').write_bytes(v2_bytes)
                self.server.request_hook = hook_prefill
                res = reader.prefill_chunks([(0, 1, 0), (0, 2, 0)], cancel=cancel)
                self.assertEqual(res.status, _reader.PrefillStatus.Completed)
                self.assertEqual(res.error_chunks, 0)
                self.assertEqual(v1_hits, 2)
            self.server.request_hook = None
            with ZarrReader.open(self.url + '/' + name, cache_dir=cache_dir, offline=True,
                                 threads=1, memory_cache=None) as off_r:
                self.assertEqual(int(off_r[0, 1, 0]), 62)
                self.assertEqual(int(off_r[0, 2, 0]), 62)

            # Active cancellation must return Cancelled cleanly without hanging.
            cancel.cancel()
            with ZarrReader.open(self.url + '/' + name, cache_dir=self.cache / 'nb-cancel-cache',
                                 threads=2, memory_cache=None) as reader:
                res_cancel = reader.prefill_chunks([(0, 0, 0), (0, 1, 0), (0, 2, 0)], cancel=cancel)
                self.assertEqual(res_cancel.status, _reader.PrefillStatus.Cancelled)

        # 3. Repeatedly mutating shard on non-batched concurrent path terminates with bounded refresh (<= 1).
        with self.subTest(case='repeatedly_changing_non_batched_terminates'):
            name = 'nb-repeat-change'
            v1_data = np.full((128, 64, 390), 10, dtype=np.uint8)
            v2_data = np.full((128, 64, 390), 20, dtype=np.uint8)
            v3_data = np.full((128, 64, 390), 30, dtype=np.uint8)
            write_custom_shard_array(self.root / name, v1_data, (64, 64, 130), (64, 64, 390))
            write_custom_shard_array(self.root / (name + '-v2'), v2_data, (64, 64, 130), (64, 64, 390))
            write_custom_shard_array(self.root / (name + '-v3'), v3_data, (64, 64, 130), (64, 64, 390))
            v2_bytes = (self.root / (name + '-v2') / 'c/0/0/0').read_bytes()
            v3_bytes = (self.root / (name + '-v3') / 'c/0/0/0').read_bytes()
            v2_etag = '"' + hashlib.sha256(v2_bytes).hexdigest() + '"'
            suffix_gets = 0
            hook_lock = threading.Lock()

            def hook_repeat(handler):
                nonlocal suffix_gets
                if handler.path == f'/{name}/c/0/0/0':
                    rng = handler.headers.get('Range') or ''
                    if rng.startswith('bytes=-'):
                        with hook_lock:
                            suffix_gets += 1
                    elif handler.headers.get('If-Match') == v2_etag:
                        (self.root / name / 'c/0/0/0').write_bytes(v3_bytes)
                return False

            with ZarrReader.open(self.url + '/' + name, threads=2, memory_cache=None) as reader:
                self.assertEqual(int(reader[0, 0, 0]), 10)
                self.assertEqual(int(reader[64, 0, 0]), 10)
                (self.root / name / 'c/0/0/0').write_bytes(v2_bytes)
                self.server.request_hook = hook_repeat
                try:
                    with self.assertRaisesRegex(RuntimeError, '412|changed'):
                        reader.read_many([np.s_[0:1, 0:1, 130:131], np.s_[0:1, 0:1, 260:261]])
                finally:
                    self.server.request_hook = None
                self.assertEqual(suffix_gets, 1)

        # 4. Post-commit 412 on non-batched path still fails fast (commit-once preserved).
        with self.subTest(case='post_commit_412_non_batched_fails_fast'):
            name = 'nb-post-commit-fail'
            v1_data = np.full((128, 64, 390), 41, dtype=np.uint8)
            v2_data = np.full((128, 64, 390), 82, dtype=np.uint8)
            write_custom_shard_array(self.root / name, v1_data, (64, 64, 130), (64, 64, 390))
            write_custom_shard_array(self.root / (name + '-v2'), v2_data, (64, 64, 130), (64, 64, 390))
            v1_bytes = (self.root / name / 'c/0/0/0').read_bytes()
            v1_etag = '"' + hashlib.sha256(v1_bytes).hexdigest() + '"'
            v2_bytes = (self.root / (name + '-v2') / 'c/0/0/0').read_bytes()
            c1_served = threading.Event()
            suffix_after_swap = 0
            swapped = False

            def hook_post(handler):
                nonlocal suffix_after_swap, swapped
                if handler.path == f'/{name}/c/0/0/0':
                    rng = handler.headers.get('Range') or ''
                    if rng == 'bytes=532480-1064959':
                        part = v1_bytes[532480:1064960]
                        handler.send_response(206)
                        handler.send_header('Content-Range', f'bytes 532480-1064959/{len(v1_bytes)}')
                        handler.send_header('Content-Length', str(len(part)))
                        handler.send_header('ETag', v1_etag)
                        handler.end_headers()
                        handler.wfile.write(part)
                        handler.wfile.flush()
                        c1_served.set()
                        return True
                    if rng == 'bytes=1064960-1597439':
                        c1_served.wait(5)
                        threading.Event().wait(0.05)
                        if not swapped:
                            swapped = True
                            (self.root / name / 'c/0/0/0').write_bytes(v2_bytes)
                        return False
                    if swapped and rng.startswith('bytes=-'):
                        suffix_after_swap += 1
                return False

            with ZarrReader.open(self.url + '/' + name, threads=1, memory_cache=None) as reader:
                self.assertEqual(int(reader[0, 0, 0]), 41)
                self.assertEqual(int(reader[64, 0, 0]), 41)
                self.server.request_hook = hook_post
                try:
                    with self.assertRaisesRegex(RuntimeError, '412|changed'):
                        reader.read_many([np.s_[0:1, 0:1, 130:131], np.s_[0:1, 0:1, 260:261]])
                finally:
                    self.server.request_hook = None
                self.assertTrue(swapped)
                self.assertEqual(suffix_after_swap, 0)

        self.server.request_hook = None

    def test_weak_absent_etag_last_modified_and_commit_once_controls(self):
        # 1. Absent ETag with changing Last-Modified (same size + identical index bytes) must fail
        #    both across 16-chunk batch windows and within a single window.
        for scope in ('across_windows', 'within_single_window'):
            with self.subTest(case=f'absent_etag_last_modified_{scope}'):
                name = f'lm-absent-{scope}'
                v1_data = np.full((20, 2, 4), 11, dtype=np.uint8)
                v2_data = np.full((20, 2, 4), 22, dtype=np.uint8)
                write_custom_shard_array(self.root / name, v1_data, (1, 1, 4), (20, 1, 4))
                write_custom_shard_array(self.root / (name + '-v2'), v2_data, (1, 1, 4), (20, 1, 4))
                s0_v1 = (self.root / name / 'c/0/0/0').read_bytes()
                s0_v2 = (self.root / (name + '-v2') / 'c/0/0/0').read_bytes()
                s1_v1 = (self.root / name / 'c/0/1/0').read_bytes()
                self.assertEqual(len(s0_v1), len(s0_v2))
                self.assertEqual(s0_v1[-324:], s0_v2[-324:])

                state = {'ver': 1}
                seen_if_match = []

                def serve_lm(handler):
                    if handler.path not in (f'/{name}/c/0/0/0', f'/{name}/c/0/1/0'):
                        return False
                    if handler.headers.get('If-Match'):
                        seen_if_match.append(handler.headers.get('If-Match'))
                    rng = handler.headers.get('Range') or ''
                    if scope == 'across_windows' and handler.path == f'/{name}/c/0/1/0' and rng == 'bytes=56-59':
                        state['ver'] = 2
                    elif scope == 'within_single_window' and handler.path == f'/{name}/c/0/0/0' and rng == 'bytes=4-7':
                        state['ver'] = 2
                    blob = s1_v1 if handler.path.endswith('c/0/1/0') else (s0_v1 if state['ver'] == 1 else s0_v2)
                    spec = rng.removeprefix('bytes=')
                    if spec.startswith('-'):
                        n = int(spec[1:])
                        start, end = len(blob) - n, len(blob) - 1
                    else:
                        a, b = spec.split('-')
                        start, end = int(a), int(b)
                    chunk = blob[start:end + 1]
                    handler.send_response(206)
                    handler.send_header('Content-Range', f'bytes {start}-{end}/{len(blob)}')
                    handler.send_header('Content-Length', str(len(chunk)))
                    handler.send_header(
                        'Last-Modified',
                        'Thu, 26 Nov 2026 00:00:01 GMT' if state['ver'] == 1 else 'Thu, 26 Nov 2026 00:00:02 GMT',
                    )
                    handler.end_headers()
                    handler.wfile.write(chunk)
                    return True

                self.server.request_hook = serve_lm
                sels = (
                    [np.s_[0:1, 0:1, 0:4]]
                    + [np.s_[y:y + 1, 1:2, 0:4] for y in range(15)]
                    + [np.s_[16:17, 0:1, 0:4]]
                    if scope == 'across_windows'
                    else [np.s_[0:1, 0:1, 0:4], np.s_[1:2, 0:1, 0:4]]
                )
                with ZarrReader.open(self.url + '/' + name, threads=1, memory_cache=None) as reader:
                    with self.assertRaisesRegex(RuntimeError, 'changed|412'):
                        reader.read_many(sels)
                self.assertEqual(seen_if_match, [])

        # 2. Weak ETag negative control: no If-Match, no merged ranges, detects Last-Modified change,
        #    and succeeds cleanly when unchanged.
        with self.subTest(case='weak_etag_conservative_and_last_modified'):
            name = 'lm-weak-control'
            v1_data = np.full((4, 1, 4), 19, dtype=np.uint8)
            v2_data = np.full((4, 1, 4), 38, dtype=np.uint8)
            write_custom_shard_array(self.root / name, v1_data, (1, 1, 4), (4, 1, 4))
            write_custom_shard_array(self.root / (name + '-v2'), v2_data, (1, 1, 4), (4, 1, 4))
            s0_v1 = (self.root / name / 'c/0/0/0').read_bytes()
            s0_v2 = (self.root / (name + '-v2') / 'c/0/0/0').read_bytes()
            state = {'ver': 1, 'mutate': False}
            ranges_seen = []
            if_matches = []

            def serve_weak(handler):
                if handler.path != f'/{name}/c/0/0/0':
                    return False
                rng = handler.headers.get('Range') or ''
                ranges_seen.append(rng)
                if handler.headers.get('If-Match'):
                    if_matches.append(handler.headers.get('If-Match'))
                if state['mutate'] and rng == 'bytes=4-7':
                    state['ver'] = 2
                blob = s0_v1 if state['ver'] == 1 else s0_v2
                spec = rng.removeprefix('bytes=')
                if spec.startswith('-'):
                    n = int(spec[1:])
                    start, end = len(blob) - n, len(blob) - 1
                else:
                    a, b = spec.split('-')
                    start, end = int(a), int(b)
                chunk = blob[start:end + 1]
                handler.send_response(206)
                handler.send_header('Content-Range', f'bytes {start}-{end}/{len(blob)}')
                handler.send_header('Content-Length', str(len(chunk)))
                handler.send_header('ETag', 'W/"sem-v1"')
                handler.send_header(
                    'Last-Modified',
                    'Thu, 26 Nov 2026 00:00:01 GMT' if state['ver'] == 1 else 'Thu, 26 Nov 2026 00:00:02 GMT',
                )
                handler.end_headers()
                handler.wfile.write(chunk)
                return True

            self.server.request_hook = serve_weak
            with ZarrReader.open(self.url + '/' + name, threads=1, memory_cache=None) as reader:
                out = reader.read_many([np.s_[0:1, 0:1, 0:4], np.s_[1:2, 0:1, 0:4]])
                self.assertEqual([int(x[0, 0, 0]) for x in out], [19, 19])
                self.assertEqual(if_matches, [])
                self.assertIn('bytes=0-3', ranges_seen)
                self.assertIn('bytes=4-7', ranges_seen)
                self.assertNotIn('bytes=0-7', ranges_seen)
                state['mutate'] = True
                with self.assertRaisesRegex(RuntimeError, 'changed|412'):
                    reader.read_many([np.s_[0:1, 0:1, 0:4], np.s_[1:2, 0:1, 0:4]])

        # 3. Deterministic WriteBehind post-commit 412 (zero contradictory .missing markers on disk)
        #    and 416 fallback commit-once semantics (replacement for superseded mixed-revision test).
        with self.subTest(case='writebehind_post_commit_and_416_commit_once'):
            name = 'wb-commit-once'
            v1_data = np.full((20, 2, 4), 55, dtype=np.uint8)
            v2_data = np.full((20, 2, 4), 66, dtype=np.uint8)
            write_custom_shard_array(self.root / name, v1_data, (1, 1, 4), (20, 1, 4))
            write_custom_shard_array(
                self.root / (name + '-v2'), v2_data, (1, 1, 4), (20, 1, 4),
                missing_chunks=[(16, 0, 0)],
            )
            v2_bytes = (self.root / (name + '-v2') / 'c/0/0/0').read_bytes()
            cache_dir = self.cache / 'wb-commit-once-cache'
            c0_done = threading.Event()
            swapped = False

            def hook_wb(handler):
                nonlocal swapped
                if handler.path == f'/{name}/c/0/0/0' and handler.headers.get('Range') == 'bytes=0-3':
                    c0_done.set()
                elif handler.path == f'/{name}/c/0/1/0' and not swapped:
                    c0_done.wait(5)
                    swapped = True
                    (self.root / name / 'c/0/0/0').write_bytes(v2_bytes)
                return False

            sels = (
                [np.s_[0:1, 0:1, 0:4]]
                + [np.s_[y:y + 1, 1:2, 0:4] for y in range(15)]
                + [np.s_[16:17, 0:1, 0:4]]
            )
            self.server.request_hook = hook_wb
            with ZarrReader.open(self.url + '/' + name, cache_dir=cache_dir, threads=1, memory_cache=None) as reader:
                with self.assertRaisesRegex(RuntimeError, '412|changed'):
                    reader.read_many(sels)
            self.server.request_hook = None
            self.assertTrue(swapped)
            self.assertEqual(list(cache_dir.rglob('*.missing')), [])

            # 416 fallback commit-once: member 0 succeeds on v1, shard mutates before member 1 -> raises 412,
            # zero v2 suffix GETs, member 0 cleanly cached.
            name2 = 'fallback-416-commit-once'
            write_custom_shard_array(self.root / name2, v1_data, (1, 1, 4), (20, 1, 4))
            write_custom_shard_array(self.root / (name2 + '-v2'), v2_data, (1, 1, 4), (20, 1, 4))
            v2_bytes2 = (self.root / (name2 + '-v2') / 'c/0/0/0').read_bytes()
            swapped2 = False
            v2_suffix_gets = 0

            def hook_416(handler):
                nonlocal swapped2, v2_suffix_gets
                if handler.path != f'/{name2}/c/0/0/0':
                    return False
                rng = handler.headers.get('Range') or ''
                if rng == 'bytes=4-11':
                    handler.send_error(416)
                    return True
                if rng == 'bytes=8-11' and not swapped2:
                    swapped2 = True
                    (self.root / name2 / 'c/0/0/0').write_bytes(v2_bytes2)
                    return False
                if swapped2 and rng.startswith('bytes=-'):
                    v2_suffix_gets += 1
                return False

            mc = MemoryCache(1 << 20)
            with ZarrReader.open(self.url + '/' + name2, threads=1, memory_cache=mc) as reader:
                self.assertEqual(int(reader[0, 0, 0]), 55)
                self.assertEqual(int(reader[0, 1, 0]), 55)
                self.server.request_hook = hook_416
                with self.assertRaisesRegex(RuntimeError, '412|changed'):
                    reader.read_many([np.s_[1:2, 0:1, 0:4], np.s_[2:3, 0:1, 0:4]])
                self.assertTrue(swapped2)
                self.assertEqual(v2_suffix_gets, 0)
                fetches_before = reader.stats()['remote_fetches']
                self.assertEqual(int(reader[1, 0, 0]), 55)
                self.assertEqual(reader.stats()['remote_fetches'], fetches_before)

        self.server.request_hook = None

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
