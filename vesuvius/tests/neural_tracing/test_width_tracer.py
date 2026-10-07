"""Width output is explicit, positive, unnormalized and cache-distinct."""
import numpy as np
import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d.inference_adapter import FiberTrace3DPredictAdapter
from vesuvius.neural_tracing.fiber_trace_3d.model import build_fiber_trace_3d_model
from vesuvius.neural_tracing.fiber_trace_3d.annotations import AnnotationConfig, predicted_width
from vesuvius.neural_tracing.fiber_trace_3d.inference_cache import validate_cached_model


@pytest.mark.parametrize('embedding',[False,True])
def test_width_same_forward_preserves_prediction_and_units(embedding):
    torch.manual_seed(7)
    cfg={'annotations':{'normal_weight':1,'width_weight':1,'hv_weight':1},
         'embedding':{'enabled':embedding},
         'model_3d':{'features_per_stage':[8,16],'normalization':'group','output_stride':2}}
    model=build_fiber_trace_3d_model(cfg).eval()
    old=FiberTrace3DPredictAdapter(cfg,emit_embeddings=embedding)
    new=FiberTrace3DPredictAdapter(cfg,emit_embeddings=embedding,emit_width=True)
    x=torch.randn(1,1,16,16,16)
    baseline=old.run_tile_inference(model,x,device=torch.device('cpu'))
    output=new.run_tile_inference(model,x,device=torch.device('cpu'))
    torch.testing.assert_close(output[:,:-1],baseline,rtol=0,atol=0)
    with torch.no_grad(): expected=predicted_width(model(x,return_annotations=True)['annotations'],AnnotationConfig.from_mapping(cfg))
    torch.testing.assert_close(output[:,-1:],expected,rtol=0,atol=0)
    product=new.output_products[-1]
    assert np.dtype(product.dtype)==np.float32 and product.value_range is None
    tensors=new.product_tensors_from_output(output)
    values=new.finalize_product_slab(product,tensors[product.name][0].numpy())
    np.testing.assert_array_equal(values['width'],expected[0,0].numpy())
    # No byte clipping, normalization or implicit stride scaling.
    assert np.all(new.finalize_product_slab(product,np.full((1,2,2,2),300.5))['width']==300.5)
    with pytest.raises(ValueError,match='invalid width'):
        new.finalize_product_slab(product,np.full((1,2,2,2),np.nan))


def test_width_requested_head_and_cache_are_explicit():
    with pytest.raises(ValueError,match='width-head'):
        FiberTrace3DPredictAdapter({},emit_width=True)
    model={'checkpoint_sha256':'a','scale':1}
    with pytest.raises(ValueError,match='different width'):
        validate_cached_model(model,'a',{'emit_width':True})
    assert validate_cached_model({**model,'width_channel':'float32-input-voxels-v1'},'a',{})['emit_width']
