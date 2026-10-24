import numpy as np
import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d.inference_adapter import FiberTrace3DPredictAdapter
from vesuvius.neural_tracing.fiber_trace_3d.inference_cache import validate_cached_model
from vesuvius.neural_tracing.fiber_trace_3d.model import build_fiber_trace_3d_model
from lasagna.normal_encoding import encode_normal_nxny_u8


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
