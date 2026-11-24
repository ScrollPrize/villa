"""Batched sheet-strip inference (sheet_strip/batched.py) against ``StripPredictor.predict``, on a random checkpoint
with a random frame model and a synthetic planar CT: on the CPU the batched geometry and engine give predict's
heights, frames and positions exactly; on a GPU, crops sampled by vc_reader's CUDA reader (``crops='gpu'``) give the
strips of CPU crops within float rounding."""
import dataclasses
from argparse import Namespace

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.tracers.fibers.data.samples import SampleConfig
from vesuvius.neural_tracing.tracers.fibers.heading_model.model import HeadingConfig, HeadingNet, save_heading_model
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.batched import StripEngine, StripGeometry
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.infer import MIN_HEAD_ARC, StripPredictor
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.model import SheetStripConfig, SheetStripFollower
from vesuvius.neural_tracing.tracers.fibers.tests.model_fixtures import array_at
from vesuvius.neural_tracing.tracers.fibers.tests.test_sheet_strip import SMALL, STRIP
from vesuvius.neural_tracing.tracers.shared.annotations.traced import TracedFiber
from vesuvius.neural_tracing.tracers.shared.crop_sampling.normalization import ZSCORE_METHOD
from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import CTVolumeSpec
from vesuvius.neural_tracing.tracers.shared.geometry import arclength

BATCH = 2


def strip_run(tmp_path):
    """A random sheet-strip checkpoint (every weight perturbed, so no head starts at zero) with a random frame model,
    on a planar CT, its volume options, a horizontal fiber and the fiber's decision arcs."""
    z, y, x = np.indices((96, 96, 160))
    array_at(tmp_path/'ct'/'0', (x+2*y+z)//2)
    spec = CTVolumeSpec(str(tmp_path/'ct'), ct_level=0, ct_grid_scale=4., grid_scale=4.,
                        ct_normalization=dict(method=ZSCORE_METHOD))
    torch.manual_seed(0)
    save_heading_model(tmp_path/'frame_model.pt',
                       HeadingNet(HeadingConfig(width=1, predict_normals=True, predict_frames=True)))
    cfg = SheetStripConfig(**SMALL, **STRIP)
    model = SheetStripFollower(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(.05*torch.randn_like(p))
    torch.save(dict(model_type='sheet_strip', model_cfg=cfg.to_dict(), ema=model.state_dict(), step=0,
                    run_config=dict(data={}), vol_spec=spec.to_dict(),
                    sample_cfg=dataclasses.asdict(SampleConfig(n_history=cfg.n_history))),
               tmp_path/'strip.pt')
    # every decision's seed (its trace's start) needs the CT structure tensor's 77^3 context inside the volume
    points = np.c_[40.+np.arange(49.), np.full(49, 48.), np.full(49, 48.)]
    fiber = TracedFiber('fiber', points, arclength(points), 'H')
    volume = Namespace(frame_checkpoint=str(tmp_path/'frame_model.pt'))
    return tmp_path/'strip.pt', volume, fiber, np.arange(MIN_HEAD_ARC, fiber.length+1e-9, 8.)


def batched(predictor, fiber, arcs):
    """Heights, frames and positions of one fiber through StripGeometry and StripEngine, the frame model in predict's
    batches."""
    geometry = StripGeometry.of(predictor)
    engine = StripEngine(predictor, batch=BATCH, frame_chunk=BATCH)
    items, request = geometry.plan(fiber, arcs)
    (heading, normal), = engine.frame_outputs([request])
    inputs = geometry.finish(fiber, items, request, heading, normal)
    (heights, _), = engine.heights([inputs])
    return heights, inputs.frames, inputs.positions


def test_batched_engine_reproduces_predict(tmp_path):
    path, volume, fiber, arcs = strip_run(tmp_path)
    predictor = StripPredictor(str(path), 'cpu', None, BATCH, volume=volume)
    heights, _, frames, positions = predictor.predict(fiber, arcs)
    assert len(heights) == len(arcs) and np.ptp(heights) > 1e-3  # a strip that depends on its inputs
    got = batched(predictor, fiber, arcs)
    for name, expected, value in zip(('heights', 'frames', 'positions'), (heights, frames, positions), got):
        np.testing.assert_array_equal(value, expected, err_msg=name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA device')
def test_gpu_crops_match_cpu_crops(tmp_path):
    """``--crops gpu`` (CUDA crops and the frame model on the GPU), per fiber and batched, against ``--crops cpu``
    with the strip model on the same GPU."""
    path, volume, fiber, arcs = strip_run(tmp_path)
    heights, _, frames, positions = StripPredictor(str(path), 'cuda', None, BATCH, volume=volume).predict(fiber, arcs)
    assert np.ptp(heights) > 1e-3
    gpu = StripPredictor(str(path), 'cuda', None, BATCH, volume=volume, crops='gpu', gpu_chunks=4)
    per_fiber = gpu.predict(fiber, arcs)
    for got in ((per_fiber[0], per_fiber[2], per_fiber[3]), batched(gpu, fiber, arcs)):
        np.testing.assert_allclose(got[2], positions, rtol=0, atol=1e-9)
        np.testing.assert_allclose(got[1], frames, rtol=0, atol=1e-4)
        # the strip model runs under bf16 autocast, so a last-bit change in its float32 crops moves heights by a few
        # bf16 steps (1/64-1/32 at these heights; 0.0625 measured on an RTX 5090), not by a fixed small amount
        np.testing.assert_allclose(got[0], heights, rtol=0, atol=.125)
