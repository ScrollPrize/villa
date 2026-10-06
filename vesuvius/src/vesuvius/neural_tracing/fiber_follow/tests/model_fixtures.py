"""Shared model, batch and run-configuration fixtures.

``config`` is a small regression model (models/crop_transformer.py), the crop/horizon/label contract most tests use;
``coordinate_config`` adds recurrent refinement. ``coordinate_batch``/``batch`` hold every input the crop transformer
reads (CT crop, seed, observed path and path-geometry tokens) with label targets. ``run_document``/``run_args`` give a
run configuration (train/run_config.py) and the trainer settings resolved from it.
"""
import json
from dataclasses import replace

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.models.crop_transformer import FlowConfig, RegressionConfig
from vesuvius.neural_tracing.fiber_follow.data.data import TracedFiber
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, arclength
from label_fixtures import state_labels

SMALL = dict(hidden=16, heads=2, layers=1, ffn=32, cnn_channels=(4, 8), cnn_blocks=(1, 1), n_future=4, n_history=32,
             gate_plane=None, recurrent_refinement_steps=0)


def small_crop(crop):
    """Round a crop up to the small models' token stride (4)."""
    return replace(crop, depth=4*((crop.depth+3)//4), width=4*((crop.width+3)//4))


def config(**kwargs):
    options = dict(SMALL, fine=CropSpec(depth=24, width=20, behind=8))
    options.update(kwargs)
    options['fine'] = small_crop(options['fine'])
    return RegressionConfig(**options)


def coordinate_config(**kwargs):
    options = dict(fine=CropSpec(depth=16, width=12, behind=7), layers=2, recurrent_refinement_steps=3)
    options.update(kwargs)
    return config(**options)


def flow_config(**kwargs):
    """A small flow model with fixed residual scales."""
    options = dict(SMALL, fine=CropSpec(depth=16, width=12, behind=7), layers=2, flow_steps=2, flow_draws=3,
                   flow_samples=0, flow_time_conditioning='input', flow_sigma_floor=1., flow_unknown_planes='padded',
                   flow_loss='mse')
    options.update(kwargs)
    options.pop('recurrent_refinement_steps', None)
    options['fine'] = small_crop(options['fine'])
    options.setdefault('flow_sigma', ((1., 1.),)*options['n_future'])
    return FlowConfig(**options)


def observation(path):
    path = np.asarray(path, dtype=float)
    return dict(pos=path[-1].copy(), frame=np.eye(3), hist_local=np.zeros((32, 3)), hmask=np.zeros(32),
                observed_path=path, seed_pos=path[0].copy(), seed_tangent=np.array([0., 0., 1.]),
                seed_age=float(arclength(path)[-1]), seed_valid=True)


def raw_batch(cfg, b=2):
    hist = torch.zeros(b, cfg.n_history, 3)
    hist[..., 2] = -torch.arange(1, cfg.n_history+1)
    q = 4*(cfg.n_future-1)+1
    x = dict(fine=torch.rand(b, cfg.input_channels, cfg.fine.depth, cfg.fine.width, cfg.fine.width),
             seed=torch.zeros(b, 1, 3), seed_mask=torch.ones(b, 1), seed_tangent=torch.tensor([0., 0., 1.]).expand(b, -1),
             seed_age=torch.zeros(b))
    return dict(x=x, hist=hist, hmask=torch.ones(b, cfg.n_history), dense_ab=torch.zeros(b, q, 2),
                dense_mask=torch.ones(b, q), **state_labels(b), endpoint_known=torch.zeros(b),
                end_local=torch.zeros(b, 3), source=torch.zeros(b),
                foreign=torch.zeros(b, cfg.fine.depth, cfg.fine.width, cfg.fine.width, dtype=torch.uint8))


def geometry_batch(c, count=2, length=100.):
    from vesuvius.neural_tracing.fiber_follow.models.path_geometry import path_geometry_inputs
    out = raw_batch(c, count)
    path = np.c_[np.zeros(int(length)+1), np.zeros(int(length)+1), np.arange(int(length)+1.)]
    out['x'].update(path_geometry_inputs([observation(path) for _ in range(count)]))
    return out


def coordinate_batch(c, b=2):
    """A batch with every input the crop transformer reads (CT crop, seed, observed path and geometry tokens)."""
    out = geometry_batch(c, b)
    out['x']['fine'] = out['x']['fine'][:, :c.input_channels].contiguous()
    return out


def forward(m, b):
    return m(b['x'], b['hist'], b['hmask'])


def proposal_output(curves, hazards, selected=-1):
    """Build the current all-proposal output contract for loss/policy fixtures."""
    from vesuvius.neural_tracing.fiber_follow.models.survival_confidence import survival_predictions
    logits, confidence = survival_predictions(hazards)
    return dict(points=curves[:, selected], initial_points=curves[:, 0],
                hazard_logits=hazards[:, selected], confidence_logits=logits[:, selected],
                confidence=confidence[:, selected], refinement_points=curves,
                refinement_hazard_logits=hazards, refinement_confidence_logits=logits,
                refinement_confidence=confidence,
                refinement_mask=torch.ones(curves.shape[:2], device=curves.device, dtype=torch.bool),
                selected_refinement=torch.full((len(curves),), selected % curves.shape[1], device=curves.device))


def line_fiber(length=600.):
    arc = np.arange(length)
    return TracedFiber('line', np.c_[arc*0+100, arc*0+100, arc+200], arc, '')


def array_at(path, values):
    path.mkdir(parents=True, exist_ok=True)
    (path/'.zarray').write_text(json.dumps(dict(shape=list(values.shape), chunks=list(values.shape),
                                                dtype='|u1', fill_value=0, order='C', filters=None, compressor=None,
                                                zarr_format=2)))
    (path/'0.0.0').write_bytes(values.astype(np.uint8).tobytes())


def batch(cfg, b=2):
    return coordinate_batch(cfg, b)


def ct_volume(root):
    """A 96^3 synthetic CT volume (x+2y+z) with its z-score normalization record under ``root``."""
    from vesuvius.neural_tracing.fiber_follow.data import ct_normalization as norm
    from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolume, FiberVolumeSpec
    z, y, x = np.indices((96, 96, 96))
    array_at(root/'ct'/'0', (x+2*y+z).clip(0, 255))
    spec = FiberVolumeSpec('', ct_zarr=str(root/'ct'), ct_level=0, ct_grid_scale=4., inputs='ct', load_presence=False)
    norm.prepare_normalization(root/'run', [spec], known=dict(method=norm.ZSCORE_METHOD, volumes={}))
    return FiberVolume(spec, cache_bytes=1 << 20)
