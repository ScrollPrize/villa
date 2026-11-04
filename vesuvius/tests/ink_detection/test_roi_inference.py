"""Structural regression checks; efficacy is separately on real scroll CT."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
import vesuvius.ink_detection.inference.infer as infer


@pytest.mark.parametrize('shape,roi',[
    ((257,319),(3,84,197,319)), # irregular final patch boundary
    ((80,90),(0,80,0,90)), # parent smaller than model patch
    ((256,256),(0,1,255,256)), # one output pixel at a corner
    ((401,319),(320,401,200,319)), # two irregular boundary-aligned final origins
    ((256,256),(64,192,64,192)),
])
@pytest.mark.parametrize('stride',[32,64,96,128])
def test_roi_preserves_exact_parent_schedule(shape,roi,stride):
    full=infer.iter_blocks(shape,128,stride)
    wanted=[b for b in full if b.y0<roi[1] and b.y0+b.valid_h>roi[0] and b.x0<roi[3] and b.x0+b.valid_w>roi[2]]
    assert infer.iter_blocks(shape,128,stride,roi_bounds=roi)==wanted
    assert infer.iter_blocks(shape,128,stride,roi_bounds=(0,shape[0],0,shape[1]))==full


@pytest.mark.parametrize('roi',[(-1,80,0,80),(0,81,0,80),(0,0,0,80),(0,80,5,4),(False,80,0,80),(0.,80,0,80),(0,80,0)])
def test_roi_rejects_wrong_or_empty_coordinate_frame(roi):
    with pytest.raises(ValueError):infer.resolve_roi_bounds(roi,height=80,width=80)


class SavedLogitModel(nn.Module):
    """Exercise actual inference assembly using frozen real CT model logits."""
    def __init__(self,probability):
        super().__init__();p=torch.as_tensor(probability).clamp(1e-7,1-1e-7)
        self.register_buffer('logits',torch.logit(p)[None,None])
    def forward(self,image):return self.logits.expand(image.shape[0],-1,-1,-1)


def test_clipped_real_prediction_assembly_and_raw_zero_skip(tmp_path):
    fixture=Path(__file__).parent/'fixtures/ink_roi_case.npz'
    with np.load(fixture) as saved:
        ct=saved['ct'];p=saved['probability']
    probability=np.zeros((128,128),np.float32);probability[:80,:80]=p
    reader=infer.FlatPatchReader(input_path='fixture',resolution='0',depth_axis_first=True,height=80,width=80,
        layer_indices=np.arange(17),output_depth=17,preprocessing='tifxyz_robust');reader._array=ct
    roi=(3,66,5,73);blocks=infer.iter_blocks((80,80),128,64,roi_bounds=roi)
    dataset=infer.FlatBlockDataset(reader=reader,blocks=blocks,patch_size=128,preprocessing='tifxyz_robust')
    out=np.zeros((63,68),np.float32);weight=np.zeros_like(out);clipped=[infer.clip_block_to_roi(b,roi) for b in blocks]
    accumulator=infer.ChunkAccumulator(shape=out.shape,chunk_shape=(17,23),prob_sum_store=out,weight_sum_store=weight,
        contribution_counts=infer.compute_chunk_contribution_counts(clipped,chunk_shape=(17,23)))
    infer.run_block_inference(loader=DataLoader(dataset,batch_size=1),model=SavedLogitModel(probability),accumulator=accumulator,
        weight_map=infer.compute_importance_map_2d(patch_size=(128,128),mode='hann').numpy(),mask=None,
        device=torch.device('cpu'),amp_dtype=None,tta_axes=(),tta_batch_size=None,roi_bounds=roi)
    accumulator.flush_remaining();result=out/weight
    np.testing.assert_allclose(result,p[3:66,5:73],rtol=1e-6,atol=1e-7)
    assert not accumulator.buffers
    # Known empty raw input must not acquire artificial ink through normalization.
    reader._array=np.zeros_like(ct);out.fill(0);weight.fill(0)
    accumulator=infer.ChunkAccumulator(shape=out.shape,chunk_shape=(17,23),prob_sum_store=out,weight_sum_store=weight,
        contribution_counts=infer.compute_chunk_contribution_counts(clipped,chunk_shape=(17,23)))
    infer.run_block_inference(loader=DataLoader(dataset,batch_size=1),model=SavedLogitModel(probability),accumulator=accumulator,
        weight_map=np.ones((128,128),np.float32),mask=None,device=torch.device('cpu'),amp_dtype=None,tta_axes=(),tta_batch_size=None,roi_bounds=roi)
    accumulator.flush_remaining();assert not out.any() and not weight.any()


def test_roi_cli_does_not_rescale_coordinates():
    args=infer.parse_args(['input.zarr','model.pth','out.tif','--resolution','2','--roi','3','83','5','85'])
    assert args.roi==[3,83,5,85] and args.resolution=='2'
