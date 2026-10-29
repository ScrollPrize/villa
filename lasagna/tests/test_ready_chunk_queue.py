"""Whole-volume accumulation must parallelize aligned chunks without races."""
import numpy as np
import ast
import re
import torch

from lasagna.tiled_predict3d import _ReadyChunkQueue


def test_real_pipeline_balances_aligned_chunks_and_matches_serial(tmp_path, monkeypatch, capsys):
    from lasagna import tiled_predict3d as tiled
    from lasagna.tests.predict3d_spawn_helpers import SpawnIdentityAdapter

    class Output:
        def __init__(self):
            self.chunks = {}

        def product_chunk_complete(self, product, *, chunk_origin_zyx):
            return False

        def write_product_chunk(self, product, *, chunk_origin_zyx, data):
            self.chunks[chunk_origin_zyx] = data['value'].copy()

    product = tiled.OutputProductSpec(name='identity', channels=('value',),
        level=0, scaledown=1, inference_scaledown=1, chunk_size=64)
    volume = np.random.default_rng(8).integers(1, 255, (128, 128, 128), dtype=np.uint8)
    monkeypatch.setattr(tiled, '_input_has_chunks', lambda *args, **kwargs: True)
    outputs = []
    for workers in (0, 4):
        output = Output()
        root = tmp_path / str(workers)
        root.mkdir()
        adapter = SpawnIdentityAdapter(product)
        tiled.run_tiled_inference_3d(
            None, volume, crop_slices=(0, 128, 0, 128, 0, 128), device=torch.device('cpu'),
            devices=(torch.device('cpu'), torch.device('cpu')),
            model_adapter=adapter, output_adapter=output, products=(product,),
            output_regions_zyx={'identity': (0, 0, 0, 128, 128, 128)},
            full_output_shapes_zyx={'identity': (128, 128, 128)},
            tile_size=96, overlap=32, border=0, tmp_dir=str(root),
            input_reader='python-zarr', accumulator_workers=workers, flush_workers=0,
            product_accumulator_dtype='float16',
        )
        outputs.append(output.chunks)
    assert outputs[0].keys() == outputs[1].keys()
    for key in outputs[0]:
        np.testing.assert_array_equal(outputs[0][key], outputs[1][key])
    counts = ast.literal_eval(re.findall(r'worker_tasks=(\[[^\]]*\])', capsys.readouterr().out)[-1])
    assert len(counts) == 4 and min(counts) > 0


def test_aligned_chunks_use_all_32_workers():
    queue = _ReadyChunkQueue()
    keys = [(4, (0, 0, 64 * x)) for x in range(32)]
    for sequence in range(3):
        for key in keys:
            queue.put(key, sequence)
    for sequence in range(3):
        assigned = [queue.take() for _ in range(32)]
        assert {key for key, _ in assigned} == set(keys)
        assert all(value == sequence for _, value in assigned)
        assert queue.take() is None  # Other updates wait for their chunk, not a worker.
        for key, _ in reversed(assigned):
            queue.complete(key)
    assert not queue.pending and not queue.active and not queue.ready


def test_blocked_chunk_does_not_block_independent_work():
    queue = _ReadyChunkQueue()
    queue.put('slow', 0)
    assert queue.take() == ('slow', 0)
    queue.put('slow', 1)
    queue.put('fast', 2)
    assert queue.take() == ('fast', 2)
    queue.complete('fast')
    queue.put('new', 3)
    assert queue.take() == ('new', 3)
    queue.complete('new')
    assert queue.take() is None
    queue.complete('slow')
    assert queue.take() == ('slow', 1)
    queue.complete('slow')
    assert not queue.pending


def test_out_of_order_workers_preserve_exact_half_accumulation():
    rng = np.random.default_rng(7)
    updates = rng.normal(size=(80, 40, 16)).astype(np.float16)
    expected = np.zeros((40, 16), np.float16)
    actual = expected.copy()
    queue = _ReadyChunkQueue()
    for sequence, update in enumerate(updates):
        expected += update
        for chunk in range(40):
            queue.put((4, (0, 0, chunk * 64)), (sequence, chunk))
    active = []
    while queue.pending:
        while len(active) < 32:
            item = queue.take()
            if item is None:
                break
            active.append(item)
        key, (sequence, chunk) = active.pop(int(rng.integers(len(active))))
        actual[chunk] += updates[sequence, chunk]
        queue.complete(key)
    np.testing.assert_array_equal(actual, expected)
