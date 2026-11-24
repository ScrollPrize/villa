"""Batched sheet-strip inference (sheet_strip/batched.py) against ``StripPredictor.predict``, on a random checkpoint
with a random frame model and a synthetic planar CT: on the CPU the batched geometry and engine give predict's
heights, frames and positions exactly, and infer_all.py's batched mode (``--workers``) writes the strips and index
records of its per-fiber loop; on a GPU, crops sampled by vc_reader's CUDA reader (``crops='gpu'``) give the strips of
CPU crops within float rounding, and batched mode with CPU crops keeps the frame model on the CPU, as fiber by fiber."""
import dataclasses
from argparse import Namespace
import json

import numpy as np
import pytest
import tifffile
import torch

from vesuvius.neural_tracing.tracers.fibers.data.samples import SampleConfig
from vesuvius.neural_tracing.tracers.fibers.heading_model.model import HeadingConfig, HeadingNet, save_heading_model
from vesuvius.neural_tracing.tracers.fibers.sheet_strip import infer_all
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.batched import StripEngine, StripGeometry
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.infer import MIN_HEAD_ARC, StripPredictor
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.model import SheetStripConfig, SheetStripFollower
from vesuvius.neural_tracing.tracers.fibers.tests.model_fixtures import array_at
from vesuvius.neural_tracing.tracers.fibers.tests.test_sheet_strip import SMALL, STRIP
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import write_fibers
from vesuvius.neural_tracing.tracers.shared.annotations.traced import TracedFiber
from vesuvius.neural_tracing.tracers.shared.crop_sampling.normalization import ZSCORE_METHOD
from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import CTVolumeSpec
from vesuvius.neural_tracing.tracers.shared.geometry import arclength

BATCH = 2


def strip_run(tmp_path, **options):
    """A random sheet-strip checkpoint (every weight perturbed, so no head starts at zero; ``options``: further
    SheetStripConfig fields) with a random frame model, on a planar CT, its volume options, a horizontal fiber and the
    fiber's decision arcs."""
    z, y, x = np.indices((96, 96, 160))
    array_at(tmp_path/'ct'/'0', (x+2*y+z)//2)
    spec = CTVolumeSpec(str(tmp_path/'ct'), ct_level=0, ct_grid_scale=4., grid_scale=4.,
                        ct_normalization=dict(method=ZSCORE_METHOD))
    torch.manual_seed(0)
    save_heading_model(tmp_path/'frame_model.pt',
                       HeadingNet(HeadingConfig(width=1, predict_normals=True, predict_frames=True)))
    cfg = SheetStripConfig(**SMALL, **STRIP, **options)
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



@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA device')
def test_cuda_workers_with_cpu_crops_match_per_fiber(tmp_path):
    """``--device cuda`` with the default CPU crops, per fiber and batched (``--workers 2``): the frame model runs on
    the CPU in both, so frames and kept nodes are the same, and strips agree within the bf16 bar."""
    path, volume, fiber, arcs = strip_run(tmp_path)
    predictor = StripPredictor(str(path), 'cuda', None, BATCH, volume=volume)
    heights, _, frames, positions = predictor.predict(fiber, arcs)
    got = batched(predictor, fiber, arcs)
    np.testing.assert_array_equal(got[2], positions)
    np.testing.assert_array_equal(got[1], frames)
    np.testing.assert_allclose(got[0], heights, rtol=0, atol=.125)

    _, common = infer_all_run(tmp_path, path, volume, fiber, 'cuda')
    infer_all.main([*common, '--out', str(tmp_path/'per_fiber')])
    infer_all.main([*common, '--out', str(tmp_path/'batched'), '--workers', '2'])
    assert index_records(tmp_path/'batched') == index_records(tmp_path/'per_fiber')
    per_fiber, batched_files = strips_written(tmp_path/'per_fiber'), strips_written(tmp_path/'batched')
    assert sorted(per_fiber) == sorted(batched_files)
    for name, value in per_fiber.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(batched_files[name] == -1, value == -1, err_msg=name)
            np.testing.assert_allclose(batched_files[name], value, rtol=0, atol=.125, err_msg=name)
        elif name.endswith('trace.json'):
            assert batched_files[name] == value, name

def infer_all_run(tmp_path, path, volume, fiber, device):
    """A two-fiber store for ``strip_run``'s checkpoint, in base voxels (the CT's grid scale is 4): its fiber and one
    beside it traced the other way. Returns the fiber ids and infer_all.py's options for it on ``device``."""
    ids = ['00000000-0000-4000-8000-000000000001', '10000000-0000-4000-8000-000000000002']
    shifted = fiber.points[::-1]+(0., 4., 0.)
    write_fibers(tmp_path/'fibers.vcobj', [(dict(id=fid, name=fid, family='H'), 4.*points)
                                           for fid, points in zip(ids, (fiber.points, shifted))], volume='test')
    return ids, ['--checkpoint', str(path), '--frame-checkpoint', volume.frame_checkpoint, '--fibers',
                 str(tmp_path/'fibers.vcobj'), '--device', device, '--batch', str(BATCH), '--threads', '1',
                 '--min-length-mm', '0', '--trim', '8']


def index_records(out):
    """OUT's index records by fiber, without their seconds."""
    lines = (out/'index'/'shard_0.jsonl').read_text().splitlines()
    return sorted(({k: v for k, v in json.loads(line).items() if k != 'seconds'} for line in lines),
                  key=lambda r: r['fiber'])


def strips_written(out):
    """Every file under OUT/strips by path: TIFF arrays, and the JSON of the rest (meta.json without its fresh uuid)."""
    files = {}
    for path in sorted((out/'strips').rglob('*.*')):
        if path.suffix == '.tif':
            files[str(path.relative_to(out))] = tifffile.imread(path)
        else:
            files[str(path.relative_to(out))] = {k: v for k, v in json.loads(path.read_text()).items() if k != 'uuid'}
    return files


def test_infer_all_batched_writes_the_per_fiber_strips(tmp_path):
    """infer_all.py on a two-fiber store, fiber by fiber and with ``--workers 2`` (spawned CPU workers): the same
    strips, expected errors, trace.json and index records apart from seconds; a rerun resumes past both."""
    path, volume, fiber, arcs = strip_run(tmp_path, strip_uncertainty=True)
    ids, common = infer_all_run(tmp_path, path, volume, fiber, 'cpu')
    infer_all.main([*common, '--out', str(tmp_path/'per_fiber')])
    infer_all.main([*common, '--out', str(tmp_path/'batched'), '--workers', '2'])
    expected, got = index_records(tmp_path/'per_fiber'), index_records(tmp_path/'batched')
    assert [r['fiber'] for r in expected] == ids
    assert [(r['decisions'], r['family'], r['path']) for r in expected] == [
        (len(arcs), 'H', f'strips/{fid[:2]}/fiber_{fid}') for fid in ids]
    assert all(r['nodes'] > 0 and r['expected_error_vox'] > 0 for r in expected)
    assert got == expected
    per_fiber, batched_files = strips_written(tmp_path/'per_fiber'), strips_written(tmp_path/'batched')
    assert sorted(per_fiber) == sorted(batched_files)
    assert {name.rsplit('/', 1)[1] for name in per_fiber} >= {'x.tif', 'y.tif', 'z.tif', 'expected_error.tif',
                                                               'meta.json', 'trace.json'}
    for name, value in per_fiber.items():
        if isinstance(value, np.ndarray):
            assert np.ptp(value[np.isfinite(value) & (value != -1)]) > 0, name  # a strip, not a constant
            np.testing.assert_array_equal(batched_files[name], value, err_msg=name)
        else:
            assert batched_files[name] == value, name
    assert json.loads((tmp_path/'batched'/'run.json').read_text()) == json.loads(
        (tmp_path/'per_fiber'/'run.json').read_text())
    infer_all.main([*common, '--out', str(tmp_path/'batched'), '--workers', '2'])  # nothing left to do
    assert len((tmp_path/'batched'/'index'/'shard_0.jsonl').read_text().splitlines()) == 2


@pytest.mark.parametrize('options, message', [
    (['--across-fibers', '--compile', 'default'], '--compile needs --workers'),
    (['--workers', '2', '--compile', 'default'], '--compile needs --workers'),
    (['--workers', '2', '--across-fibers', '--compile', 'default', '--device', 'mps'], '--compile needs a CUDA')])
def test_infer_all_refuses_batched_options_without_their_mode(tmp_path, options, message):
    with pytest.raises(SystemExit, match=message):
        infer_all.main(['--out', str(tmp_path), *options])


def test_batched_engine_refuses_mps():
    """The engine has no MPS launch handling: on MPS, batched mode stops before loading anything."""
    with pytest.raises(ValueError, match='on MPS run without --workers'):
        StripEngine(Namespace(device='mps'))
