from pathlib import Path
import ast
import json

import numpy as np
import pytest
import zarr

from lasagna.compact_zarr import CompactOutputAdapter, resolve_output, chunk_file
from lasagna.lasagna_volume import LasagnaVolume
from lasagna.tiled_predict3d import OutputProductSpec


def product(name, channels, level=0, dtype=np.uint8):
    return OutputProductSpec(name=name, channels=channels, level=level, scaledown=2**level,
                             chunk_size=4, dtype=dtype, inference_scaledown=2**level)


def test_touched_python_parses():
    root = Path(__file__).resolve().parents[2]
    for name in ["lasagna/compact_zarr.py", "lasagna/lasagna_volume.py", "lasagna/preprocess_cos_omezarr.py",
                 "lasagna/tiled_predict3d.py", "lasagna/demand_predict3d.py", "lasagna/inference_provenance.py",
                 "vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/infer.py",
                 "vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/live_evaluate.py"]:
        ast.parse((root / name).read_text(), filename=name)


def test_packed_atomic_blank_resume_pyramid(tmp_path):
    p = product("fiber", ["presence", "nx", "ny"])
    width = product("width", ["width"])
    output = CompactOutputAdapter(root=tmp_path / "prediction.zarr", products=(p, width),
                                  n_levels=2, validity_channel="presence", blank_invalid=True)
    output.prepare(base_shape_zyx=(5, 5, 5), ome_chunk=4)
    values = np.full((4, 4, 4), 255, np.uint8)
    presence = values.copy()
    presence[0, 0, 0] = 0
    data = [(p, dict(presence=presence, nx=values, ny=values)), (width, dict(width=values))]
    output.write_products_chunk(data, chunk_origin_zyx=(0, 0, 0))
    array = zarr.open_array(str(tmp_path / "prediction.zarr/sd0/0"), mode="r")
    assert array.chunks == (4, 4, 4, 4)
    assert np.array_equal(array[:, 0, 0, 0], [0, 0, 0, 0])
    assert np.array_equal(array[:, 1, 1, 1], [255]*4)
    assert output.validate_chunk((0, 0, 0))
    manifest = LasagnaVolume.load(tmp_path / "prediction.zarr")
    assert manifest.validity_channel == "presence"
    assert manifest.groups["sd0"].channels == ["presence", "nx", "ny", "width"]
    output.prepare(base_shape_zyx=(5, 5, 5), ome_chunk=4)
    with pytest.raises(ValueError, match="partial"):
        output.write_products_chunk(data[:1], chunk_origin_zyx=(0, 0, 0))
    output.build_pyramids(workers=2)
    assert chunk_file(tmp_path / "prediction.zarr/sd0/1", (0, 0, 0)).exists()
    output.write_products_chunk(data, chunk_origin_zyx=(0, 0, 0))
    assert not chunk_file(tmp_path / "prediction.zarr/sd0/1", (0, 0, 0)).exists()


def test_scales_validity_and_zero_chunks(tmp_path):
    cos = product("cos", ["cos"])
    normal = product("normal", ["grad_mag", "nx", "ny"], level=1)
    output = CompactOutputAdapter(root=tmp_path / "normal.zarr", products=(cos, normal),
                                  n_levels=2, validity_channel="grad_mag")
    output.prepare(base_shape_zyx=(8,8,8), ome_chunk=4)
    output.write_products_chunk([(normal, {c: np.zeros((4,4,4), np.uint8) for c in normal.channel_names})],
                                chunk_origin_zyx=(0,0,0))
    assert output.product_chunk_complete(normal, chunk_origin_zyx=(0,0,0))
    manifest = LasagnaVolume.load(tmp_path / "normal.zarr")
    assert set(manifest.groups) == {"sd0", "sd1"}
    attrs = json.loads((tmp_path / "normal.zarr/.zattrs").read_text())
    del attrs["lasagna"]["validity_channel"]
    with pytest.raises(ValueError, match="validity"):
        LasagnaVolume.validate_compact(attrs["lasagna"])


def test_output_resolution(tmp_path):
    assert resolve_output(tmp_path / "new.lasagna.json")[0].name == "new.zarr"
    old = tmp_path / "old.lasagna.json"
    old.write_text("{}")
    assert resolve_output(old)[1] == "legacy"


def test_compact_options_and_resume_encoding(tmp_path):
    p = product("fiber", ["presence", "nx", "ny"])
    width = product("width", ["width"], dtype=np.float32)
    output = CompactOutputAdapter(root=tmp_path / "p.zarr", products=(p,width),
                                  n_levels=1, validity_channel="presence", blank_invalid=False)
    kwargs = dict(base_shape_zyx=(4,4,4), ome_chunk=4,
                  extra_fields={"width_encoding": "float32-input-voxels"})
    output.prepare(**kwargs)
    data = np.full((4,4,4), 128, np.uint8)
    output.write_products_chunk([(p, dict(presence=np.zeros_like(data), nx=data, ny=data)),
                                 (width, dict(width=np.full(data.shape, 12.5, np.float32)))],
                                chunk_origin_zyx=(0,0,0))
    packed = zarr.open_array(str(tmp_path / "p.zarr/sd0/0"), mode="r")
    assert packed.dtype == np.float32
    assert np.array_equal(packed[:,0,0,0], [0,128,128,12.5])
    output.prepare(**kwargs)
    with pytest.raises(ValueError, match="width_encoding"):
        output.prepare(**{**kwargs, "extra_fields": {"width_encoding": "uint8-base-voxels"}})
    path = tmp_path / "p.zarr/sd0/0/.zarray"
    metadata = json.loads(path.read_text())
    metadata["order"] = "F"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="encoding"):
        output.prepare(**kwargs)


def test_packed_provenance(tmp_path):
    from lasagna.inference_provenance import structural_inventory
    p = product("fiber", ["presence", "nx", "ny"])
    output = CompactOutputAdapter(root=tmp_path / "p.zarr", products=(p,), n_levels=1, validity_channel="presence")
    output.prepare(base_shape_zyx=(4,4,4), ome_chunk=4)
    inventory = structural_inventory(tmp_path / "p.zarr")
    assert inventory[0]["path"] == "p.zarr/.zattrs"
    assert inventory[1]["levels"][0]["shape_czyx"] == [3,4,4,4]


def test_width_export():
    from vesuvius.neural_tracing.fiber_trace_3d.inference_adapter import FiberTrace3DPredictAdapter
    adapter = object.__new__(FiberTrace3DPredictAdapter)
    adapter.emit_normals = False
    adapter.emit_width = True
    adapter.product_prefix = "fiber"
    adapter.width_uint8_base = True
    adapter.input_to_base = 2
    p = product("fiber_width", ["width"])
    raw = np.array([[[[0., 12.5, 200.]]]], np.float32)
    values = adapter.finalize_product_slab(p, raw)["width"]
    assert values.dtype == np.uint8
    assert values.tolist() == [[[0,25,255]]]


def test_compact_demand_complete_and_warm_resume(tmp_path):
    from lasagna.demand_predict3d import DemandStore, DemandGeometry
    geometry = DemandGeometry((4,4,4), 4, 0, 0, 1, 4)
    p = product("fiber", ["presence", "nx", "ny"])
    class Model:
        output_products = (p,)
        def finalize_product_slab(self, product, raw):
            return {c: np.rint(raw[i]*255).astype(np.uint8) for i,c in enumerate(product.channel_names)}
    # Store ownership validates these logical channel locators, even though
    # packed publication uses output.array_path rather than those old locators.
    from dataclasses import replace
    from lasagna.tiled_predict3d import OutputChannelSpec
    p = replace(p, channels=tuple(OutputChannelSpec(c, str(tmp_path / c)) for c in p.channel_names))
    Model.output_products = (p,)
    model = Model()
    output = CompactOutputAdapter(root=tmp_path / "prediction.zarr", products=(p,), n_levels=1,
                                  validity_channel="presence", blank_invalid=True)
    def prepare():
        output.prepare(base_shape_zyx=(4,4,4), ome_chunk=4)
    with DemandStore(tmp_path, {"format": 3}, geometry, model, output, prepare_output=prepare) as store:
        store.request((0,0,0))
        tile = store.next_tile()
        assert tile is not None
        arrays = {"weight": np.ones((4,4,4), np.float32), "fiber": np.ones((3,4,4,4), np.float32)}
        store.complete(tile, arrays)
        assert store._cached_chunk_complete((0,0,0))
        assert len(tuple(store._chunk_files((0,0,0)))) == 1
    with DemandStore(tmp_path, {"format": 3}, geometry, model, output, prepare_output=prepare) as store:
        store.request((0,0,0))
        assert store.stats["warm_chunks"] == 1


@pytest.mark.parametrize("with_distance", [False, True])
def test_compact_lasagna_full_volume_matches_legacy(tmp_path, monkeypatch, with_distance):
    import torch
    from lasagna import preprocess_cos_omezarr as module
    class Model:
        def __call__(self, x):
            return torch.full((x.shape[0], 8, *x.shape[-3:]), .2, device=x.device)
    monkeypatch.setattr(module.LasagnaCosPredict3DAdapter, "load_model", lambda self, device: Model())
    def distance(**kwargs):
        kwargs["output_level_arr"][:] = 91
    monkeypatch.setattr(module, "_compute_pred_dt_slab", distance)
    source = tmp_path / "input.zarr"
    try:
        array = zarr.open_array(str(source), mode="w", shape=(8,8,8), chunks=(4,4,4), dtype="uint8", zarr_format=2)
    except TypeError:
        array = zarr.open_array(str(source), mode="w", shape=(8,8,8), chunks=(4,4,4), dtype="uint8")
    array[:] = 50
    for fmt in ("legacy", "compact"):
        dest = tmp_path / fmt / "output.lasagna.json"
        module.run_preprocess_3d(input_path=str(source), output_path=str(dest), output_format=fmt,
            unet3d_checkpoint="unused.pt", device="cpu", crop_xyzwhd=None, tile_size=8, overlap=0, border=0,
            cos_scaledown=1, scaledown=2, base_ref=str(source), base_scale=0, n_levels=3, ome_chunk=4,
            flush_workers=0, accumulator_workers=0, pred_dt_path=str(source) if with_distance else None)
    packed_root = tmp_path / "compact/output.zarr"
    manifest = LasagnaVolume.load(packed_root)
    assert manifest.validity_channel == "grad_mag"
    assert set(manifest.groups) == {"sd0", "sd1"}
    assert not (tmp_path / "compact/.output_pred_dt.staging.zarr").exists()
    assert not (tmp_path / "compact/output_pred_dt.ome.zarr").exists()
    for group in manifest.groups.values():
        for level in range(group.scaledown, 3):
            packed = zarr.open_array(str(packed_root / f"sd{group.scaledown}/{level}"), mode="r")
            for i, channel in enumerate(group.channels):
                legacy = zarr.open_array(str(tmp_path / f"legacy/output_{channel}.ome.zarr/{level}"), mode="r")
                assert np.array_equal(packed[i], legacy[:])


def test_compact_publication_benchmark(tmp_path):
    import time
    from lasagna.tiled_predict3d import create_product_omezarr_groups, OmeZarrOutputAdapter, OutputChannelSpec
    p = OutputProductSpec(name="fiber", channels=tuple(OutputChannelSpec(c, str(tmp_path / c))
        for c in ("presence", "nx", "ny")), level=0, scaledown=1, inference_scaledown=1, chunk_size=64)
    create_product_omezarr_groups(products=(p,), base_shape_zyx=(64,64,64), n_levels=1, ome_chunk=64)
    legacy = OmeZarrOutputAdapter(products=(p,), n_levels=1)
    compact = CompactOutputAdapter(root=tmp_path / "compound.zarr", products=(p,), n_levels=1, validity_channel="presence")
    compact.prepare(base_shape_zyx=(64,64,64), ome_chunk=64)
    rng = np.random.default_rng(17)
    data = {c: rng.integers(0,256,(64,64,64),dtype=np.uint8) for c in p.channel_names}
    measured = {"legacy": [], "compact": []}
    for i in range(23):
        for name, output in (("legacy", legacy), ("compact", compact)):
            start = time.perf_counter()
            output.write_products_chunk([(p,data)], chunk_origin_zyx=(0,0,0))
            if i >= 3:
                measured[name].append((time.perf_counter()-start)*1000)
    print("publication benchmark: one CPU caller, 64^3, three random uint8 channels, blosc-zstd, 3 warmups + 20 measured writes")
    for name, times in measured.items():
        print(name, "mean/p50/p95 ms", np.mean(times), *np.percentile(times,[50,95]))
    assert np.array_equal(zarr.open_array(str(tmp_path / "compound.zarr/sd0/0"),mode="r")[:],np.stack(list(data.values())))


def test_compact_live_store_native_trace(tmp_path):
    native = pytest.importorskip("vc.fiber_trace")
    from lasagna.demand_predict3d import DemandGeometry
    from vesuvius.neural_tracing.fiber_trace_3d.live_evaluate import create_demand_store
    config = {"model_3d": {"input_channels": 1, "output_channels": 7, "direction_branch_count": 1,
                           "features_per_stage": [2,4], "strides": [[1,1,1],[2,2,2]], "normalization": "group"},
              "training": {"mixed_precision": "off"}}
    geometry = DemandGeometry((4,4,4),4,0,0,1,4)
    with create_demand_store(tmp_path, {"base_shape": [4,4,4]}, {"test": "compound"},
                             geometry, config, None, 0, {}) as store:
        store.request((0,0,0))
        tile = store.next_tile()
        p, = store.products
        raw = np.full((7,4,4,4), .5, np.float32)
        raw[6] = 1
        store.complete(tile, {"weight": np.ones((4,4,4), np.float32), p.name: raw})
        assert len(tuple(store._chunk_files((0,0,0)))) == 1
        field = native.open_prediction_field(str(tmp_path / "prediction.zarr"), scaledown_power=0)
        trace = native.TraceConfig(step_voxels=1., beam_width=1, beam_lookahead_steps=1, parallel_threads=1,
                                   smoothness_weight=0., smoothness_normal_weight=0., smoothness_tangent_weight=0.,
                                   cumulative_smoothness_tangent_weight=0.)
        result = native.trace_extrapolation(field, [1.,1.,1.], [1.,0.,0.], 1., trace)
        assert len(result.points) >= 1


def test_compact_tta_and_mixed_views(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.tta import selected_manifest
    from vesuvius.neural_tracing.fiber_trace_3d.mixed_inference import write_mixed_manifest
    native = pytest.importorskip("vc.fiber_trace")
    groups = []
    for variant in ("mean", "median"):
        groups.append(product(variant, [variant + "_" + c for c in ("presence", "nx", "ny")]))
    tta_root = tmp_path / "tta/prediction.zarr"
    tta = CompactOutputAdapter(root=tta_root, products=groups, n_levels=1, validity_channel="mean_presence")
    tta.prepare(base_shape_zyx=(4,4,4), ome_chunk=4)
    view = selected_manifest(tta_root, "mean")
    field = native.open_prediction_field(str(view), scaledown_power=0)
    assert field.option_count == 1
    roots = []
    for role, p in enumerate((product("fiber",["presence","nx","ny"]),
                              product("embedding",["presence","embedding_000","embedding_001"], dtype=np.float32))):
        root = tmp_path / str(role) / "prediction.zarr"
        out = CompactOutputAdapter(root=root, products=(p,), n_levels=1, validity_channel="presence")
        out.prepare(base_shape_zyx=(4,4,4), ome_chunk=4)
        (root.parent / "cache_identity.json").write_text(json.dumps({"identity": {"source": {"same": True}}}))
        roots.append(root)
    mixed = write_mixed_manifest(tmp_path / "mixed", *roots)
    field = native.open_prediction_field(str(mixed), scaledown_power=0)
    assert field.option_count == 1


def test_compact_float_native_trace(tmp_path):
    native = pytest.importorskip("vc.fiber_trace")
    p = product("fiber", ["presence", "nx", "ny"])
    width = product("width", ["width"], dtype=np.float32)
    root = tmp_path / "p.zarr"
    output = CompactOutputAdapter(root=root, products=(p,width), n_levels=1, validity_channel="presence")
    output.prepare(base_shape_zyx=(4,4,4), ome_chunk=4,
                   extra_fields={"width_encoding": "float32-input-voxels"})
    ones = np.ones((4,4,4), np.float32)
    output.write_products_chunk([(p, dict(presence=(255*ones).astype(np.uint8),
                                          nx=(255*ones).astype(np.uint8), ny=(128*ones).astype(np.uint8))),
                                 (width, dict(width=12.5*ones))], chunk_origin_zyx=(0,0,0))
    field = native.open_prediction_field(str(root), scaledown_power=0, enable_width=True)
    config = native.TraceConfig(step_voxels=1., beam_width=1, beam_lookahead_steps=1,
                                parallel_threads=1, width_relative_tolerance=.4,
                                smoothness_weight=0., smoothness_normal_weight=0., smoothness_tangent_weight=0.,
                                cumulative_smoothness_tangent_weight=0.)
    result = native.trace_extrapolation(field, [1.,1.,1.], [1.,0.,0.], 1., config)
    assert len(result.points) > 1
