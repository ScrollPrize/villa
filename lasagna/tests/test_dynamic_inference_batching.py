"""Shared worker contract tests; tiny CPU fixtures, not CPU training."""
import queue

import numpy as np
import pytest
import torch

from lasagna import tiled_predict3d as tiled
from lasagna.demand_predict3d import DemandWorkerPool


def test_budget_batch_split_preserves_values_and_learns_cap():
    class Adapter:
        def run_tile_inference(self, model, tile, *, device):
            if len(tile) > 2:
                raise torch.cuda.OutOfMemoryError('synthetic budget')
            return {'value': tile * 2}
    tile = torch.arange(7.).reshape(7,1)
    cap = [8]
    output = tiled._resident_tile_inference(Adapter(), None, tile, torch.device('cpu'), cap)
    assert torch.equal(output['value'], tile * 2)
    assert 1 <= cap[0] <= 2


@pytest.mark.parametrize("cap", [0, -1, True, 1.5, "4"])
def test_invalid_batch_cap_before_allocating(cap):
    with pytest.raises(ValueError, match="max_batch_size"):
        DemandWorkerPool({}, {}, [], max_batch_size=cap)


def test_ready_batch_never_waits_and_consumes_only_one_sentinel():
    work = queue.Queue()
    assert tiled._ready_inference_batch(work, "a", 128) == (["a"], False)
    for item in ("b", "c", None, None):
        work.put(item)
    assert tiled._ready_inference_batch(work, "a", 2) == (["a", "b"], False)
    assert tiled._ready_inference_batch(work, work.get_nowait(), 128) == (["c"], True)
    assert work.get_nowait() is None


class BatchIdentityAdapter:
    def __init__(self, bad_batch=False, native_stride=1):
        self.sizes = []
        self.bad_batch = bad_batch
        self.native_output_stride = native_stride

    def load_model(self, *, device):
        return torch.nn.Identity()

    def preprocess_tile(self, tile, valid_mask):
        assert tile.shape == valid_mask.shape
        # Independent sample statistics, deliberately different input ranges.
        return tile - tile.mean(dim=(2, 3, 4), keepdim=True)

    def run_tile_inference(self, model, tile, *, device):
        self.sizes.append(tile.shape[0])
        stride = self.native_output_stride
        output = model(tile)[:, :, ::stride, ::stride, ::stride]
        return output[:1] if self.bad_batch else output

    def product_tensors_from_output(self, output):
        return {"identity": output, "double": output * 2}


@pytest.mark.parametrize("cap", [1, 3, 128])
@pytest.mark.parametrize("profile", [False, True])
def test_batch_distinct_mixed_dtype_tiles_products_and_partial_tail(cap, profile):
    _run_worker_fixture(cap, profile)


@pytest.mark.parametrize("cap", [1, 3])
@pytest.mark.parametrize("total_scale", [4, 8])
@pytest.mark.parametrize("profile", [False, True])
def test_native_stride4_batch_weight_lattice_and_remaining_downsample(cap, total_scale, profile):
    _run_worker_fixture(cap, profile, native_stride=4, total_scale=total_scale)


def test_bad_output_batch_aborts_without_result_publication():
    _run_worker_fixture(3, False, bad_batch=True)


def _run_worker_fixture(cap, profile, bad_batch=False, native_stride=1, total_scale=2):
    count, size = 5, 2 * total_scale
    output_size = size // total_scale
    ramp = np.ones(size, np.float32) if native_stride == 1 else np.linspace(0.25, 1, size, dtype=np.float32)
    weights = torch.from_numpy(ramp[:, None, None] * ramp[None, :, None] * ramp[None, None, :])
    native_weights = tiled._pyrdown3d(weights.unsqueeze(0), factor=native_stride).squeeze(0) if native_stride > 1 else weights
    remaining = total_scale // native_stride
    expected_weights = tiled._pyrdown3d(native_weights.unsqueeze(0), factor=remaining).squeeze(0) if remaining > 1 else native_weights
    input_layouts, input_bytes = tiled._packed_layouts((("input", (size,) * 3, np.uint16),))
    result_layouts, result_bytes = tiled._packed_layouts((
        (f"weight:{total_scale}", (output_size,) * 3, np.float32),
        ("product:identity", (1, output_size, output_size, output_size), np.float32),
        ("product:double", (1, output_size, output_size, output_size), np.float32)))
    inputs = tiled._create_shared_slots(count, input_bytes)
    results = tiled._create_shared_slots(count, result_bytes)
    adapter, work, events = BatchIdentityAdapter(bad_batch, native_stride), queue.Queue(), queue.Queue()
    expected = []
    try:
        for index in range(count):
            dtype = np.uint8 if index % 2 else np.uint16
            raw = (np.arange(size ** 3).reshape((size,) * 3) * (index + 1) + 11).astype(dtype)
            from dataclasses import replace
            layout = replace(input_layouts[0], dtype=np.dtype(dtype).str)
            tiled._slot_array(inputs[index], layout)[:] = raw
            for out in result_layouts:
                tiled._slot_array(results[index], out)[:] = -999
            name = "identity" if index % 2 else "double"
            work.put((index, (index, 0, 0), index, count - 1 - index, (name,), np.dtype(dtype).str))
            converted = tiled._input_tile_to_device(raw, torch.device("cpu"))
            normalized = converted - converted.mean(dim=(2, 3, 4), keepdim=True)
            product = normalized[0, :, ::native_stride, ::native_stride, ::native_stride] * (1 if name == "identity" else 2)
            weighted = product * native_weights
            expected.append((name, (tiled._pyrdown3d(weighted, factor=remaining) if remaining > 1 else weighted).numpy()))
        work.put(None)
        input_specs = tuple(tiled._SharedSlotSpec(x.name, input_bytes, input_layouts) for x in inputs)
        result_specs = tuple(tiled._SharedSlotSpec(x.name, result_bytes, result_layouts) for x in results)
        tiled._multi_gpu_worker_main(0, "cpu", adapter, None, size, ramp,
            {"identity": total_scale, "double": total_scale}, native_stride, input_specs, result_specs, work, events,
            profile, cap, True)
        messages = []
        while not events.empty():
            messages.append(events.get_nowait())
        if bad_batch:
            assert [x for x in messages if x[0] == "error"]
            assert not [x for x in messages if x[0] in ("result", "batch")]
            return
        assert not [x for x in messages if x[0] == "error"]
        assert adapter.sizes == ([1] * 5 if cap == 1 else [3, 2] if cap == 3 else [5])
        released = [x for x in messages if x[0] == "input_released"]
        returned = [x for x in messages if x[0] == "result"]
        batches = [x for x in messages if x[0] == "batch"]
        assert [x[1] for x in released] == list(range(count))
        assert [x[1] for x in returned] == list(range(count))
        assert sum(x[2]["size"] for x in batches) == count
        assert sum(x[2]["seconds"] for x in batches) == pytest.approx(sum(x[6] for x in returned))
        assert all(len(x) == (9 if profile else 7) for x in returned)
        for index, (name, values) in enumerate(expected):
            slot = results[count - 1 - index]
            assert returned[index][3] == count - 1 - index
            for layout in result_layouts:
                output = tiled._slot_array(slot, layout)
                if layout.name == f"weight:{total_scale}":
                    np.testing.assert_array_equal(output, expected_weights.numpy())
                elif layout.name == f"product:{name}":
                    np.testing.assert_array_equal(output, values)
                else:
                    np.testing.assert_array_equal(output, -999)
    finally:
        for shm in inputs + results:
            shm.close()
            shm.unlink()
