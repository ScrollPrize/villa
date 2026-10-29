"""CPU-only tests of model adaptation, standard Zarr, pooling and tile resume."""
import itertools
import json

import numpy as np
import pytest
import torch
import zarr

from lasagna.qualzz_inference import (
    QualzzModelAdapter, ProbabilityZarrOutputAdapter, _NormalizedModel, _pool2,
    _quantize, run_inference,
)
from lasagna.compact_zarr import chunk_file, _open_group, _create_array
from lasagna.tiled_predict3d import run_tiled_inference_3d


def output(tmp_path, shape=(9, 8, 7), chunk=4):
    adapter = QualzzModelAdapter(tmp_path, chunk)
    product, = adapter.output_products
    writer = ProbabilityZarrOutputAdapter(tmp_path / "prediction.zarr", product, shape,
                                          voxel_size_um=9.6, provenance={"test": True})
    writer.prepare()
    return adapter, product, writer


def test_channel_selection_is_four_class_softmax(tmp_path):
    adapter = QualzzModelAdapter(tmp_path)
    logits = torch.arange(8.).reshape(2, 4, 1, 1, 1)
    actual = adapter.product_tensors_from_output({"seg": logits})["semantic"]
    torch.testing.assert_close(actual, logits.softmax(1)[:, 1:3], rtol=0, atol=0)
    assert actual.shape == (2, 2, 1, 1, 1)
    with pytest.raises(ValueError, match="segmentation logits"):
        adapter.product_tensors_from_output({"seg": logits[:, :2]})


def test_model_uses_supplied_normalization_on_uint8_scale():
    seen = []
    def normalize(value):
        seen.append(value)
        return value - 42
    model = _NormalizedModel(torch.nn.Identity(), normalize)
    ct = torch.tensor([0., 1., 128., 255.]).reshape(1, 1, 1, 1, 4)
    torch.testing.assert_close(model(ct / 255), ct - 42)
    torch.testing.assert_close(seen[0], ct, rtol=0, atol=0)


def test_standard_arrays_and_pool_before_quantization(tmp_path):
    adapter, product, writer = output(tmp_path)
    shape = writer.shape
    values = np.random.default_rng(7).random((2, *shape), dtype=np.float32)
    for origin in itertools.product(*(range(0, n, 4) for n in shape)):
        block = values[(slice(None), *(slice(o, o+4) for o in origin))]
        writer.write_product_chunk(product, chunk_origin_zyx=origin,
                                   data=adapter.finalize_product_slab(product, block))
        assert writer.product_chunk_complete(product, chunk_origin_zyx=origin)
    group = zarr.open_group(writer.root, mode="r")
    assert set(group.array_keys()) == {"semantic", "tracing"}
    assert "lasagna" not in group.attrs
    np.testing.assert_array_equal(group["semantic"][:], _quantize(values))
    np.testing.assert_array_equal(group["tracing"][:], _quantize(_pool2(values)))
    assert group["semantic"].dtype == np.uint8
    assert group["semantic"].chunks == (2, 4, 4, 4)
    meta = json.loads((tmp_path / "prediction.zarr/semantic/.zarray").read_text())
    assert meta["zarr_format"] == 2
    assert meta["compressor"]["cname"] == "zstd"
    # Show that this isn't pooling already-quantized predictions.
    assert np.any(_quantize(_pool2(values)) != _quantize(_pool2(_quantize(values)/255)))


def test_completion_requires_both_scales_and_resume_validates(tmp_path):
    adapter, product, writer = output(tmp_path)
    values = np.zeros((2, 4, 4, 4), np.float32)
    writer.write_product_chunk(product, chunk_origin_zyx=(0, 0, 0),
                               data=adapter.finalize_product_slab(product, values))
    writer.prepare()
    chunk_file(tmp_path / "prediction.zarr/tracing", (0, 0, 0)).unlink()
    assert not writer.product_chunk_complete(product, chunk_origin_zyx=(0, 0, 0))
    writer.provenance = {"test": False}
    with pytest.raises(ValueError, match="provenance"):
        writer.prepare()


def test_nonfinite_prediction_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        _quantize(np.array([np.nan]))


class TinyModel(torch.nn.Module):
    def forward(self, tile):
        return {"seg": torch.cat((tile, tile*2, tile*3, tile*4), dim=1)}


@pytest.mark.parametrize("flush_workers", [0, 2])
def test_tiled_engine_and_resume(tmp_path, flush_workers):
    shape = (12, 10, 8)
    adapter, product, writer = output(tmp_path, shape)
    group = _open_group(tmp_path / "ct.zarr")
    _create_array(group, "0", shape, (4, 4, 4), np.uint8, None,
                  array_path=tmp_path / "ct.zarr/0")
    source = group["0"]
    source[:] = np.random.default_rng(2).integers(1, 256, size=shape, dtype=np.uint8)
    progress = {}
    model = TinyModel()
    def run():
        run_tiled_inference_3d(model, source, crop_slices=(0,12,0,10,0,8),
            device=torch.device("cpu"), devices=(torch.device("cpu"),),
            model_adapter=adapter, output_adapter=writer, products=(product,),
            output_regions_zyx={"semantic": (0,0,0,*shape)},
            full_output_shapes_zyx={"semantic": shape}, tile_size=8, overlap=4, border=2,
            input_zarr_path=str(tmp_path / "ct.zarr/0"),
            tmp_dir=str(tmp_path), progress=progress, product_accumulator_dtype="float32",
            flush_workers=flush_workers)
    run()
    predicted = zarr.open_group(writer.root, mode="r")
    expected = adapter.product_tensors_from_output(model(torch.from_numpy(source[:].copy())
                    .float()[None,None]/255))["semantic"][0].numpy()
    np.testing.assert_array_equal(predicted["semantic"][:], _quantize(expected))
    np.testing.assert_array_equal(predicted["tracing"][:], _quantize(_pool2(expected)))
    model.forward = lambda _: pytest.fail("complete output must not infer again")
    run()


def test_entry_point_rejects_wrong_resolution_before_loading(tmp_path):
    with pytest.raises(ValueError, match="8.64 and 9.6"):
        run_inference(model_dir=tmp_path, input_path=tmp_path, output=tmp_path,
                      voxel_size_um=2.4, devices=(torch.device("cpu"),))


def test_local_model_loader_and_full_cpu_entry_point(tmp_path):
    # Exercise the supplied-code contract without downloading weights or
    # requiring the optional nnU-Net dependency in the base regression suite.
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "model.py").write_text('''
import torch
class Model(torch.nn.Module):
    def forward(self, x):
        return {"seg": torch.cat((x*0, x*0+1, x*0+2, x*0+3), dim=1)}
def load_model(folder, device):
    return Model().to(device).eval()
def normalize_ct(x):
    return (x - x.mean()) / x.std().clamp_min(10)
''')
    (folder / "plans.json").write_text("{}")
    (folder / "checkpoint_final.pth").write_bytes(b"fixture")
    group = _open_group(tmp_path / "ct.zarr")
    _create_array(group, "0", (8,8,8), (8,8,8), np.uint8, None,
                  array_path=tmp_path / "ct.zarr/0")
    group["0"][:] = 128
    path = run_inference(model_dir=folder, input_path=tmp_path / "ct.zarr/0",
                         output=tmp_path / "out.zarr", voxel_size_um=9.6,
                         devices=(torch.device("cpu"),), chunk_size=4,
                         flush_workers=0, accumulator_workers=0)
    result = zarr.open_group(str(path), mode="r")
    p = torch.arange(4.).softmax(0)[1:3].numpy()
    np.testing.assert_array_equal(result["semantic"][:],
                                  np.broadcast_to(_quantize(p)[:,None,None,None], (2,8,8,8)))
    np.testing.assert_array_equal(result["tracing"][:],
                                  np.broadcast_to(_quantize(p)[:,None,None,None], (2,4,4,4)))


def test_cli_always_enables_cooperative_gpu_handling(monkeypatch):
    from lasagna import qualzz_inference as module
    monkeypatch.delenv("LASAGNA_COOPERATIVE", raising=False)
    monkeypatch.setattr(module, "resolve_inference_devices", lambda **kw: (torch.device("cuda:0"),))
    seen = []
    monkeypatch.setattr(module, "run_inference", lambda **kw: seen.append(kw))
    module.main(["--model-dir", "model", "--input", "ct", "--output", "out", "--voxel-size-um", "9.6"])
    import os
    assert os.environ["LASAGNA_COOPERATIVE"] == "1"
    assert seen[0]["devices"] == (torch.device("cuda:0"),)
