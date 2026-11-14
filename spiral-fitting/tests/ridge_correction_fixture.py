"""A tiny synthetic fit and a CT whose ridges sit at a known offset from it.

The checkpoint is a real SpiralAndTransform (random gap logits and a small
random flow) saved the way fits are saved. The CT is written as an
uncompressed Zarr v2 array: each voxel's intensity is a bright ridge where the
fit's own winding coordinate W (from the model's forward transform) is within
a ridge width of an integer plus a planted offset d(theta), so the papyrus of
winding k sits at W = k + d. Chunks that hold only air are not written, as in
masked scroll volumes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

SHAPE = (96, 224, 224)
CHUNK = 32
Z_BEGIN, Z_END = 0, 96
CENTRE = 112.0
AMPLITUDE = 0.15  # planted offset d(theta) = AMPLITUDE * sin(2 theta), windings


def planted_offset(theta):
    return AMPLITUDE * np.sin(2 * theta)


def make_fit(directory: Path, seed=0):
    import flatten_spiral_checkpoint as fsc
    from checkpoint_io import model_state_sha256
    from config import Config
    from transforms import SpiralAndTransform

    directory.mkdir(parents=True, exist_ok=True)
    umbilicus = directory / "umbilicus.json"
    umbilicus.write_text(json.dumps({"control_points": [
        {"z": 0, "y": CENTRE, "x": CENTRE}, {"z": 96, "y": CENTRE, "x": CENTRE}]}))
    cfg = Config().as_dict()
    cfg.update(model_flow_bounds_radius=112, model_flow_bounds_z_margin=24, model_flow_voxel_resolution=16,
               model_gap_expander_logit_resolution=12, model_gap_expander_capacity_windings=8,
               model_gap_expander_num_windings=8, model_linear_z_resolution=24, model_initial_dr_per_winding=12.0,
               model_num_flow_stages=1, output_first_winding=1, shell_outer_winding_idx=6)
    head = {"cfg": cfg, "z_begin": Z_BEGIN, "z_end": Z_END, "spiral_outward_sense": "CW"}
    z = np.arange(Z_BEGIN, Z_END)
    centre = fsc.json_umbilicus_z_to_yx(str(umbilicus))
    umb = torch.from_numpy(np.concatenate([z[:, None], centre(z)], -1).astype(np.float32))
    model = SpiralAndTransform(
        flow_integration_steps=3, flow_integration_solver="rk4",
        flow_min_corner_zyx=torch.tensor([Z_BEGIN - 24, -112, -112]),
        flow_max_corner_zyx=torch.tensor([Z_END + 24, 112, 112]),
        umbilicus_zyx=umb, config=fsc._checkpoint_config(head), spiral_outward_sense="CW")
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        logits = model.gap_expander_params.logits
        logits.add_(0.0005 * torch.randn(logits.shape, generator=gen))
        for flow in model.flow_field.flows:
            flow.add_(0.02 * torch.randn(flow.shape, generator=gen))
    state = model.state_dict()
    checkpoint = dict(head, schema_version=2, completed_iterations=7, base_shape_zyx=list(SHAPE),
                      spiral_and_transform=state,
                      model_state_sha256=model_state_sha256(state, (), Z_BEGIN, Z_END))
    path = directory / "checkpoint_fitted.ckpt"
    torch.save(checkpoint, path)
    return path, umbilicus


def winding_and_angle(checkpoint_path, umbilicus, points_zyx):
    """W and the spiral angle of CT points, through the fit's own transform."""
    import flatten_spiral_checkpoint as fsc
    import sample_spiral
    from checkpoint_io import load_checkpoint_cpu

    ck = load_checkpoint_cpu(str(checkpoint_path))
    model = fsc._build_model(ck, fsc._checkpoint_config(ck), Path(umbilicus), torch.device("cpu"))
    tr = model.get_slice_to_spiral_transform()
    dr = float(model.get_dr_per_winding().detach())
    W, TH = [], []
    with torch.inference_mode():
        for i in range(0, len(points_zyx), 400_000):
            s = tr(torch.as_tensor(points_zyx[i:i + 400_000], dtype=torch.float32))
            theta, _ = sample_spiral.get_theta(s[:, 1:])
            radius = torch.linalg.norm(s[:, 1:], dim=1)
            W.append((radius / dr - theta / (2 * math.pi)).numpy())
            TH.append(theta.numpy())
    return np.concatenate(W), np.concatenate(TH)


def make_ct(directory: Path, checkpoint_path, umbilicus, *, offset=planted_offset, seed=1, width=0.08):
    """Write the synthetic CT as a Zarr v2 array; returns its path."""
    directory.mkdir(parents=True, exist_ok=True)
    zz, yy, xx = np.meshgrid(*(np.arange(s) for s in SHAPE), indexing="ij")
    pts = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], 1).astype(np.float32)
    W, theta = winding_and_angle(checkpoint_path, umbilicus, pts)
    r = W - offset(theta)
    dist = r - np.round(r)
    rng = np.random.default_rng(seed)
    vol = 40 + 160 * np.exp(-0.5 * (dist / width) ** 2) + rng.normal(0, 8, len(W))
    radius = np.hypot(pts[:, 1] - CENTRE, pts[:, 2] - CENTRE)
    vol[radius > 100] = 0  # air outside the scroll
    vol = np.clip(np.round(vol), 0, 255).astype(np.uint8).reshape(SHAPE)
    (directory / ".zarray").write_text(json.dumps({
        "zarr_format": 2, "shape": list(SHAPE), "chunks": [CHUNK] * 3, "dtype": "|u1", "compressor": None, "fill_value": 0,
        "order": "C", "filters": None, "dimension_separator": "/"}))
    for kz in range(SHAPE[0] // CHUNK):
        for ky in range(SHAPE[1] // CHUNK):
            for kx in range(SHAPE[2] // CHUNK):
                block = vol[kz * CHUNK:(kz + 1) * CHUNK, ky * CHUNK:(ky + 1) * CHUNK, kx * CHUNK:(kx + 1) * CHUNK]
                if not block.any():
                    continue
                path = directory / str(kz) / str(ky) / str(kx)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(np.ascontiguousarray(block).tobytes())
    return directory
