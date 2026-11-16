from functools import partial
import multiprocessing as mp
from pathlib import Path
import pickle

import numpy as np
import pytest

from vc_reader import images
from vc_reader.reader import ZarrReader, ZarrWriter


def normalized_chunk(source, missing, index):
    if missing and index == (0, 1, 1):
        return None
    start = [int(at) * chunk for at, chunk in zip(index, source.chunks)]
    return source.read_region(start, source.chunks).astype(np.float32) / np.float32(255)


def read_in_worker(reader, result):
    with reader:
        result.put(reader[:])


class CtDataset:
    def __init__(self, reader):
        self.reader = reader

    def __len__(self):
        return 1

    def __getitem__(self, index):
        import torch
        return torch.from_numpy(self.reader[:])


@pytest.fixture
def ct_source(tmp_path):
    fixture = Path(__file__).parent / 'data/PHerc0800'
    ct = np.stack([images.read(path) for path in sorted(fixture.glob('*.tif'))])
    path = tmp_path / 'ct.zarr'
    ZarrWriter.create(path, shape=ct.shape, chunks=(3, 16, 16), dtype=ct.dtype)[:] = ct
    with ZarrReader.open(path, threads=1) as source:
        yield source, ct.astype(np.float32) / np.float32(255)


def computed(source, **options):
    missing = options.pop('missing', False)
    return ZarrReader.computed(partial(normalized_chunk, source, missing), shape=source.shape,
                               chunks=source.chunks, dtype='<f4', fill_value=-1,
                               identity='PHerc0800-normalized', threads=1, **options)


@pytest.mark.parametrize('cache', [False, True])
@pytest.mark.parametrize('memory', [None, 1 << 20])
def test_computed_ct_pickle_roundtrip(ct_source, tmp_path, cache, memory):
    source, expected = ct_source
    with computed(source, cache_dir=tmp_path / 'cache' if cache else None, memory_cache=memory) as reader:
        np.testing.assert_array_equal(reader[:], expected)
        with pickle.loads(pickle.dumps(reader)) as restored:
            np.testing.assert_array_equal(restored[:], expected)


def test_computed_ct_missing_chunk_survives_pickle(ct_source):
    source, expected = ct_source
    expected[:, 16:, 16:] = -1
    with computed(source, missing=True) as reader:
        with pickle.loads(pickle.dumps(reader)) as restored:
            np.testing.assert_array_equal(restored[:], expected)


def test_computed_ct_spawn_worker(ct_source):
    source, expected = ct_source
    context = mp.get_context('spawn')
    result = context.Queue()
    with computed(source) as reader:
        worker = context.Process(target=read_in_worker, args=(reader, result))
        try:
            worker.start()
            np.testing.assert_array_equal(result.get(timeout=30), expected)
            worker.join(timeout=30)
            assert worker.exitcode == 0
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=10)
            result.close()
            result.join_thread()


def test_computed_ct_spawn_dataloader(ct_source):
    torch = pytest.importorskip('torch')
    source, expected = ct_source
    with computed(source) as reader:
        batches = list(torch.utils.data.DataLoader(CtDataset(reader), batch_size=1, num_workers=1,
                                                   multiprocessing_context='spawn', timeout=30))
        assert len(batches) == 1
        np.testing.assert_array_equal(batches[0].numpy()[0], expected)


def test_computed_still_requires_picklable_callback(ct_source):
    source, _ = ct_source
    with ZarrReader.computed(lambda index: source[:], shape=source.shape, chunks=source.shape) as reader:
        with pytest.raises((AttributeError, pickle.PicklingError)):
            pickle.dumps(reader)
