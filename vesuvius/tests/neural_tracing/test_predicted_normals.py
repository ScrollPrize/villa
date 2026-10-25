import numpy as np
import json
import pytest
import torch
import zarr
from vesuvius.neural_tracing.fiber_trace_3d.inference_adapter import FiberTrace3DPredictAdapter
from vesuvius.neural_tracing.fiber_trace_3d.inference_cache import validate_cached_model
from vesuvius.neural_tracing.fiber_trace_3d.model import build_fiber_trace_3d_model
from lasagna.normal_encoding import encode_normal_nxny_u8


@pytest.mark.parametrize('normal', [False, True])
def test_volume_export_includes_available_normals(tmp_path, monkeypatch, normal):
    from vesuvius.neural_tracing.fiber_trace_3d import infer
    from lasagna.lasagna_volume import LasagnaVolume

    cfg = {'annotations': {'normal_weight': int(normal), 'width_weight': 1, 'hv_weight': 1},
           'model_3d': {'features_per_stage': [8, 16], 'normalization': 'group', 'output_stride': 2}}
    torch.manual_seed(7)
    model = build_fiber_trace_3d_model(cfg).eval()
    monkeypatch.setattr(FiberTrace3DPredictAdapter, 'load_model', lambda self, device: model)
    checkpoint = tmp_path / 'model.pt'
    torch.save({'config': cfg, 'step': 123}, checkpoint)
    source = tmp_path / 'input.zarr'
    kwargs = dict(mode='w', shape=(16, 16, 16), chunks=(8, 8, 8), dtype='uint8')
    try:
        array = zarr.open_array(str(source), zarr_format=2, **kwargs)
    except TypeError:  # Zarr 2 uses v2 by default.
        array = zarr.open_array(str(source), **kwargs)
    array[:] = 100
    for fmt in ('compact', 'legacy'):
        output = tmp_path / fmt / 'prediction.lasagna.json'
        infer.run_fiber_trace_3d_inference(
            input_path=str(source), output_path=output, checkpoint=checkpoint,
            output_format=fmt, device='cpu', no_download=True, base_ref=str(source), base_scale=0,
            tile_size=16, overlap=0, border=0, inference_scaledown_power=1,
            levels=3, ome_chunk=4, flush_workers=0, accumulator_workers=0,
            inference_precision='off',
        )
        manifest = LasagnaVolume.load(output.parent / 'prediction.zarr' if fmt == 'compact' else output)
        channels = {c for group in manifest.groups.values() for c in group.channels}
        expected = {'presence', 'nx', 'ny', 'width'}
        if normal:
            expected |= {'sheet_nx', 'sheet_ny', 'sheet_grad_mag'}
        assert channels == expected
        provenance = json.loads((output.parent / 'inference.json').read_text())
        schema = provenance['checkpoint']['output_schema']
        assert schema['annotations'] == cfg['annotations']
        assert 'annotation_products_exported' not in schema
    if normal:
        array = zarr.open_array(str(tmp_path / 'compact/prediction.zarr/sd1/1'), mode='r')
        assert array.shape[0] == array.chunks[0] == 7
        assert np.any(array[-1] == 255)
        values = array[:]
        assert np.all(values[:, values[0] == 0] == 0)


@pytest.mark.parametrize('width', [False, True])
def test_normals_are_explicit_and_preserve_other_outputs(width):
    torch.manual_seed(7)
    cfg = {'annotations': {'normal_weight':1,'width_weight':1,'hv_weight':1},
           'model_3d': {'features_per_stage':[8,16],'normalization':'group','output_stride':2}}
    model = build_fiber_trace_3d_model(cfg).eval()
    old = FiberTrace3DPredictAdapter(cfg, emit_width=width)
    new = FiberTrace3DPredictAdapter(cfg, emit_width=width, emit_normals=True)
    x = torch.randn(1,1,16,16,16)
    baseline = old.run_tile_inference(model,x,device=torch.device('cpu'))
    output = new.run_tile_inference(model,x,device=torch.device('cpu'))
    torch.testing.assert_close(output[:,:-6],baseline,rtol=0,atol=0)
    with torch.no_grad(): expected = model(x,return_annotations=True)['annotations'][:,:6].float().sigmoid()
    torch.testing.assert_close(output[:,-6:],expected,rtol=0,atol=0)
    products = new.product_tensors_from_output(output)
    old_products = old.product_tensors_from_output(baseline)
    for key in old_products: torch.testing.assert_close(products[key],old_products[key],rtol=0,atol=0)
    values = new.finalize_product_slab(new.output_products[-1],expected[0].numpy())
    nx,ny = encode_normal_nxny_u8(*expected[0].numpy())
    np.testing.assert_array_equal(values['sheet_nx'],nx)
    np.testing.assert_array_equal(values['sheet_ny'],ny)
    assert np.all(values['sheet_grad_mag']==255)
    with pytest.raises(ValueError,match='invalid normal'):
        new.finalize_product_slab(new.output_products[-1], np.full((6,2,2,2),np.nan))


def test_requested_head_and_cache_are_explicit():
    with pytest.raises(ValueError,match='normal-head'):
        FiberTrace3DPredictAdapter({},emit_normals=True)
    model={'checkpoint_sha256':'a','scale':1}
    with pytest.raises(ValueError,match='different sheet normal'):
        validate_cached_model(model,'a',{'emit_normals':True})
    assert validate_cached_model({**model,'sheet_normal_channels':'six-sigmoid-compact-axis-u8-v1'},'a',{})['emit_normals']


@pytest.mark.parametrize('axis,expected', [([1.,0.,0.],(255,128)),([0.,1.,0.],(128,255)),([0.,0.,1.],(128,128))])
def test_known_normal_axes(axis,expected):
    from vesuvius.neural_tracing.fiber_trace_3d.direction import encode_lasagna_direction_3x2
    adapter=FiberTrace3DPredictAdapter({'annotations':{'normal_weight':1}},emit_normals=True)
    encoded=encode_lasagna_direction_3x2(torch.tensor([axis])).numpy().reshape(6,1,1,1)
    values=adapter.finalize_product_slab(adapter.output_products[-1],encoded)
    # At nz=0 either antipodal sign is a valid hemisphere representative.
    assert abs(int(values['sheet_nx'].item())-128)==abs(expected[0]-128)
    assert abs(int(values['sheet_ny'].item())-128)==abs(expected[1]-128)
