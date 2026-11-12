"""Selective channel transport uses the existing range/ETag fixtures."""
import numpy as np
import pickle
import multiprocessing
import unittest

from test_source_storage import RemoteFixtureCase
from vc_reader.reader import MemoryCache, ZarrReader
from vc_reader.reader.writer import ZarrWriter


def fork_channel_registration(reader, connection):
    try:
        connection.send((reader.cache_channels, int(reader[5, 0, 0, 0])))
    except BaseException as error:
        connection.send(error)
    finally:
        connection.close()


class ChannelStorageHttpTests(RemoteFixtureCase):
    def make_volume(self, name, version, sharded=False):
        shape = (7, 64, 64, 64)
        data = (np.arange(np.prod(shape), dtype=np.uint32).reshape(shape) // 17 % 251).astype(np.uint8)
        writer = ZarrWriter.create(self.root/name, shape=shape, chunks=shape,
                                  dtype='u1', compressor='blosc', cname='zstd', level=3,
                                  version=version, separator='/', shards=shape if sharded else None)
        writer[:] = data
        return data

    def test_partial_ranges_and_shared_channel_policy(self):
        for version, sharded in [(2, False), (3, False), (3, True)]:
            with self.subTest(version=version, sharded=sharded):
                name = f'v{version}-{sharded}'
                data = self.make_volume(name, version, sharded)
                cache = MemoryCache(8 << 20)
                with ZarrReader.open(self.url+'/'+name, memory_cache=cache, threads=1) as reader:
                    reader.request_channel_cache([4, 5])
                    regions = [(4, slice(None), slice(None), slice(None)),
                               (5, slice(None), slice(None), slice(None))]
                    got = reader.read_many(regions)
                    for channel, value in zip([4, 5], got):
                        np.testing.assert_array_equal(value, data[channel])
                    floats = reader.read_many(regions, dtype=np.float32)
                    for channel, value in zip([4, 5], floats):
                        np.testing.assert_array_equal(value, data[channel].astype(np.float32))
                    first = reader.stats()
                    # An independent user sees the same volume policy and extends it.
                    with ZarrReader.open(self.url+'/'+name, memory_cache=cache, threads=1) as other:
                        self.assertEqual(other.cache_channels, [4, 5])
                        other.request_channel_cache([6])
                        self.assertEqual(reader.cache_channels, [4, 5, 6])
                        restored = pickle.loads(pickle.dumps(reader))
                        self.assertEqual(restored.cache_channels, [4, 5, 6])
                        restored.close()
                        np.testing.assert_array_equal(other[4], data[4])
                        self.assertEqual(other.stats()['remote_fetches'], 0)
                    np.testing.assert_array_equal(reader[5], data[5])
                    self.assertEqual(reader.stats()['remote_fetches'], first['remote_fetches'])
                    self.assertLessEqual(cache.nbytes, 3*64**3)
                    # Reading the whole chunk remains value-exact after partial caching.
                    np.testing.assert_array_equal(reader[:], data)
                    self.assertLess(first['remote_bytes'], reader.stats()['remote_bytes'])

    def test_channel_ranges_preserve_revision_check(self):
        self.make_volume('changing', 3)
        payload = next(p for p in (self.root/'changing'/'c').rglob('*') if p.is_file())
        calls = 0

        def change_after_header(handler):
            nonlocal calls
            if '/changing/c/' in handler.path and handler.headers.get('Range'):
                calls += 1
                if calls == 2:
                    contents = bytearray(payload.read_bytes())
                    contents[-1] ^= 1
                    payload.write_bytes(contents)
            return False

        self.server.request_hook = change_after_header
        with ZarrReader.open(self.url+'/changing', memory_cache=8 << 20, threads=1) as reader:
            reader.request_channel_cache([4])
            with self.assertRaisesRegex(RuntimeError, '412|changed'):
                reader[4]

    @unittest.skipUnless('fork' in multiprocessing.get_all_start_methods(), 'fork is unavailable')
    def test_fork_preserves_shared_and_later_channel_requests(self):
        data = self.make_volume('fork-policy', 3)
        url = self.url+'/fork-policy'
        with ZarrReader.open(url, threads=1) as first, \
             ZarrReader.open(url, threads=1) as stale, \
             ZarrReader.open(url, threads=1) as passive:
            stale.request_channel_cache([4])
            first.request_channel_cache([5])
            # passive never requested channels; stale's own snapshot is only [4].
            for reader in (passive, stale):
                context = multiprocessing.get_context('fork')
                parent, child = context.Pipe()
                process = context.Process(target=fork_channel_registration, args=(reader, child))
                process.start()
                child.close()
                try:
                    self.assertTrue(parent.poll(20), 'forked channel reader did not return')
                    result = parent.recv()
                    if isinstance(result, BaseException):
                        raise result
                    self.assertEqual(result, ([4, 5], int(data[5, 0, 0, 0])))
                finally:
                    parent.close()
                    process.join(5)
                    if process.is_alive():
                        process.terminate()
                        process.join()
                self.assertEqual(process.exitcode, 0)
