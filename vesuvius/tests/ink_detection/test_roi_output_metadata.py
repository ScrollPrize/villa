"""Real-CT fixture, saved logits: no new checkpoint prediction or ink claim."""
import json
from pathlib import Path
import numpy as np
import pytest
import tifffile
import torch
from torch import nn
import zarr
import vesuvius.ink_detection.inference.infer as infer

class SavedFixtureModel(nn.Module):
    def __init__(self, probability):
        super().__init__()
        p = torch.as_tensor(probability).clamp(1e-7, 1-1e-7)
        self.register_buffer('logits', torch.logit(p)[None, None])

    def forward(self, image):
        return self.logits.expand(image.shape[0], -1, -1, -1)

@pytest.mark.parametrize('next_roi', [None, [2, 70, 4, 76]])
def test_reused_output_never_retains_old_roi_frame(tmp_path, next_roi):
    fixture = Path(__file__).parent / 'fixtures/ink_roi_case.npz'
    with np.load(fixture) as saved:
        ct, probability = saved['ct'], saved['probability']
    input_path = tmp_path / 'real_fixture.zarr'
    volume = zarr.open_array(str(input_path), mode='w', shape=ct.shape,
                             chunks=ct.shape, dtype=ct.dtype, zarr_format=2)
    volume[:] = ct
    tile = np.zeros((128, 128), np.float32)
    tile[:80, :80] = probability
    configured = infer.ConfiguredModel(model=SavedFixtureModel(tile), patch_size=128,
        input_depth=17, preprocessing='tifxyz_robust', amp_dtype=None)
    output = tmp_path / 'ink.tif'
    metadata = output.with_suffix('.tif.roi.json')
    args = infer.parse_args([str(input_path), 'unused-checkpoint.pth', str(output),
        '--num-workers', '0', '--no-compile', '--stride', '64', '--roi', '3', '66', '5', '73'])
    infer.infer_single_zarr(args=args, input_zarr=input_path, configured_model=configured,
                            device=torch.device('cpu'), output_tiff=output)
    assert tifffile.imread(output).shape == (63, 68)
    assert json.loads(metadata.read_text())['roi_row_start_stop_col_start_stop'] == [3, 66, 5, 73]
    args.roi = next_roi
    infer.infer_single_zarr(args=args, input_zarr=input_path, configured_model=configured,
                            device=torch.device('cpu'), output_tiff=output)
    if next_roi is None:
        assert tifffile.imread(output).shape == (80, 80)
        assert not metadata.exists(), 'Full image inherited the old cropped coordinate frame'
    else:
        assert tifffile.imread(output).shape == (68, 72)
        assert json.loads(metadata.read_text())['roi_row_start_stop_col_start_stop'] == next_roi
