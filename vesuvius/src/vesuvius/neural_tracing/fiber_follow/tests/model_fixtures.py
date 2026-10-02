"""Shared model, batch and CLI fixtures.

``coordinate_config``/``coordinate_batch`` are a small version of the model the coordinate regression run trains
(scripts/train_coordinate_regression.sh): CT-only input, patch4 token-only encoder with a residual stem, fine
history encoder with path and path-geometry tokens, recurrent refinement. ``config``/``batch``
are the small generic model fixtures.
"""
import json
from dataclasses import replace

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.models.model import CoordinateRegressionConfig
from vesuvius.neural_tracing.fiber_follow.data.data import TracedFiber
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec
from label_fixtures import state_labels

REQUIRED = ['--name', 'test', '--fiber-zarrs', 'unused', '--fibers', 'unused',
            '--ct', 'unused', '--manifest', 'unused']


def config(**kwargs):
    options = dict(fine=CropSpec(depth=24, width=20, behind=8), hidden=16,
                   heads=2, layers=1, decoder_layers=1, n_future=4, n_history=32, activation_checkpointing=False,
                   recurrent_refinement_steps=0, stem_channels=4, encoder_ffn=32, decoder_ffn=32, scorer_layers=1)
    options.update(kwargs)
    crop = options['fine']
    options['fine'] = replace(crop, depth=4*((crop.depth+3)//4), width=4*((crop.width+3)//4))
    return CoordinateRegressionConfig(**options)


def coordinate_config(**kwargs):
    options = dict(fine=CropSpec(depth=16, width=12, behind=7), stem_channels=4, stem_blocks=2, layers=2, recurrent_refinement_steps=3)
    options.update(kwargs)
    return config(**options)


def slab_inputs(b):
    valid = torch.zeros(b, 8, dtype=torch.bool)
    valid[:, :2] = True
    return dict(history_slabs=torch.rand(b, 8, 2, 8, 65, 65), history_valid=valid,
                history_pose=torch.zeros(b, 8, 14), history_ages=torch.zeros(b, 8),
                history_overlap=torch.zeros(b, 8), history_load_seconds=torch.zeros(b))


def raw_batch(cfg, b=2):
    hist = torch.zeros(b, cfg.n_history, 3)
    hist[..., 2] = -torch.arange(1, cfg.n_history+1)
    q = 4*(cfg.n_future-1)+1
    x = dict(fine=torch.rand(b, cfg.input_channels, cfg.fine.depth, cfg.fine.width, cfg.fine.width),
             seed=torch.zeros(b, 1, 3), seed_mask=torch.ones(b, 1), seed_tangent=torch.tensor([0., 0., 1.]).expand(b, -1),
             seed_age=torch.zeros(b))
    x.update(slab_inputs(b))
    return dict(x=x, hist=hist, hmask=torch.ones(b, cfg.n_history), dense_ab=torch.zeros(b, q, 2),
                dense_mask=torch.ones(b, q), **state_labels(b), endpoint_known=torch.zeros(b),
                end_local=torch.zeros(b, 3), source=torch.zeros(b),
                foreign=torch.zeros(b, cfg.fine.depth, cfg.fine.width, cfg.fine.width, dtype=torch.uint8))


def slab_batch(c, b=2, step=0):
    return raw_batch(c, b)


def path_batch(c, count=2):
    out = slab_batch(c, count)
    points = torch.zeros(count, 8, 3, 3)
    points[..., 2] = torch.tensor([-1., 0., 1.])
    tangents = torch.zeros_like(points)
    tangents[..., 2] = 1
    out['x'].update(history_path_points=points, history_path_tangents=tangents,
                    history_path_valid=out['x']['history_valid'][..., None].expand(-1, -1, 3).clone())
    return out


def geometry_batch(c, count=2, length=100.):
    from vesuvius.neural_tracing.fiber_follow.models.path_geometry import path_geometry_inputs
    from test_history_slabs import observation
    out = path_batch(c, count)
    path = np.c_[np.zeros(int(length)+1), np.zeros(int(length)+1), np.arange(int(length)+1.)]
    out['x'].update(path_geometry_inputs([observation(path) for _ in range(count)]))
    return out


def coordinate_batch(c, b=2):
    """A batch with every input the coordinate regression model reads (CT-only image, slabs, path and geometry)."""
    out = geometry_batch(c, b)
    out['x']['fine'] = out['x']['fine'][:, :c.input_channels].contiguous()
    out['x']['history_pose'] = torch.rand_like(out['x']['history_pose'])
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
