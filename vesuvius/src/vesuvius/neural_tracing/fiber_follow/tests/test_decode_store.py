"""In-place decoding keeps every value, is resumable, and reads through memory maps."""
import json
import os

import numcodecs
import numpy as np

from vesuvius.neural_tracing.fiber_follow.decode_store import RAW_SUFFIX, chunk_file, chunk_keys, decode_in_place
from vesuvius.neural_tracing.fiber_follow.volume import ChunkedArray


def write_array(path, data, chunks, sep):
    codec = numcodecs.Blosc(cname='zstd', clevel=1)
    path.mkdir()
    (path/'.zarray').write_text(json.dumps(dict(shape=list(data.shape), chunks=list(chunks), dtype='|u1',
        fill_value=0, order='C', filters=None, dimension_separator=sep, compressor=codec.get_config(), zarr_format=2)))
    grid = [range(-(-n//c)) for n, c in zip(data.shape, chunks)]
    for z in grid[0]:
        for y in grid[1]:
            for x in grid[2]:
                if (z+y+x) % 5 == 4:
                    continue  # a missing chunk reads as fill
                block = np.zeros(chunks, np.uint8)
                part = data[z*chunks[0]:(z+1)*chunks[0], y*chunks[1]:(y+1)*chunks[1], x*chunks[2]:(x+1)*chunks[2]]
                block[:part.shape[0], :part.shape[1], :part.shape[2]] = part
                file = path/sep.join(map(str, (z, y, x)))
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(codec.encode(block))
    return path


def test_decode_in_place_is_identical_resumable_and_memory_mapped(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, (13, 9, 11), dtype=np.uint8)
    for sep in ('.', '/'):
        source = write_array(tmp_path/f'array{sep == "/"}', data, (4, 4, 4), sep)
        compressed = ChunkedArray(source)
        starts = [(-2, -1, -3), (0, 0, 0), (5, 3, 2), (11, 7, 9)]
        expected = [compressed.read(start, (7, 6, 5)) for start in starts]
        nearest = compressed.sample_nearest(np.array([[1., 2., 3.], [12.4, 8.2, 10.]]))
        keys = chunk_keys(source, sep)
        # Leftovers of an interrupted run: one finished sibling, one partial write.
        with open(chunk_file(source, sep, keys[0])+RAW_SUFFIX, 'wb') as fh:
            fh.write(compressed.chunk(keys[0]).tobytes())
        with open(chunk_file(source, sep, keys[1])+RAW_SUFFIX+'.partial.1', 'wb') as fh:
            fh.write(b'junk')
        report = decode_in_place(source, workers=2, verify=4)
        assert report['chunks'] == len(keys) and report['swapped'] == len(keys) and report['verified'] == 4
        assert report['decoded_bytes'] == 64*(len(keys)-1)
        assert json.loads((source/'.zarray').read_text())['compressor'] is None
        assert chunk_keys(source, sep) == keys
        assert not any(RAW_SUFFIX in name or '.partial.' in name for _, _, files in os.walk(source) for name in files)
        decoded = ChunkedArray(source)
        assert decoded.codec is None
        chunk = decoded.chunk(keys[0])
        assert isinstance(chunk, np.memmap) and not chunk.flags.writeable
        for start, value in zip(starts, expected):
            assert np.array_equal(decoded.read(start, (7, 6, 5)), value)
        assert np.array_equal(decoded.sample_nearest(np.array([[1., 2., 3.], [12.4, 8.2, 10.]])), nearest)
        assert decoded.chunk((0, 1, 3)) is None  # still missing, still fill
        # Running again is a no-op.
        again = decode_in_place(source, workers=2, verify=4)
        assert again['decoded_bytes'] == 0 and again['swapped'] == 0 and again['verified'] == 0
        assert np.array_equal(ChunkedArray(source).read(starts[0], (7, 6, 5)), expected[0])
