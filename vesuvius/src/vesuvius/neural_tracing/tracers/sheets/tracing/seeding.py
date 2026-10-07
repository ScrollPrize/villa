"""Seed rows from an episode's GT window, and the between-step perturbation used during training."""
from __future__ import annotations

import torch

from vesuvius.neural_tracing.tracers.sheets.geometry import grid_eval, project, sheet_distance, surface_frame
from vesuvius.neural_tracing.tracers.shared.crop_sampling.gpu import sample_block


def _sheet_noise(frame, sigma_n: float, sigma_t: float, generator):
    """Gaussian offsets along the normal and within the sheet, in the given ``(t, a, n)`` frames."""
    shape = frame.shape[:-2]
    noise = torch.randn(*shape, 3, device=frame.device, generator=generator)
    noise = torch.stack([noise[..., 0]*sigma_t, noise[..., 1]*sigma_t, noise[..., 2]*sigma_n], -1)
    return (frame@noise[..., None])[..., 0]


def seed_rows(gt, valid, cfg, spec, generator=None, noise: bool = True):
    """GT rows ``u = -history*step .. 0`` at the front points' ``v`` (B, history+1, N, 3), oldest first."""
    d = cfg.data
    B = gt.shape[0]
    u = -torch.arange(d.history, -1, -1, device=gt.device, dtype=torch.float32)*d.step
    v = torch.arange(d.front_points, device=gt.device)*d.point_spacing
    uu, vv = torch.meshgrid((u-spec.u_lo)/spec.res, (v-spec.v_lo)/spec.res, indexing='ij')
    uv = torch.stack([uu, vv], -1).reshape(1, -1, 2).expand(B, -1, -1)
    S, Su, Sv, _ = grid_eval(gt, valid, uv)
    if noise:
        S = S+_sheet_noise(surface_frame(Su, Sv), d.seed_noise_normal, d.seed_noise_tangent, generator)
    return S.reshape(B, d.history+1, d.front_points, 3)


def perturb(pos, gt, valid, cfg, snap: torch.Tensor, generator=None):
    """Training only: in episodes with ``snap`` (B,), snap points beyond ``reset_distance`` back onto their GT
    foot point; add noise everywhere. Free episodes let points drift onto other sheets (confidence negatives).
    """
    tc = cfg.train
    with torch.no_grad():
        _, S, frame, ok = project(gt, valid, pos)
        far = ok & snap[:, None] & (sheet_distance(pos, S, frame).abs() > tc.reset_distance)
        noise = _sheet_noise(frame, tc.state_noise_normal, tc.state_noise_tangent, generator)
        delta = torch.where(far[..., None], S-pos, torch.zeros_like(pos))
        delta = delta+torch.where(ok[..., None], noise, torch.zeros_like(noise))
    return pos+delta, far


def sheet_jump(pos, gt, valid, cfg, active: torch.Tensor, generator=None, block=None, origin=None):
    """Push a random contiguous run of front points onto a neighbouring sheet in ``active`` episodes.

    Each pushed point moves along the GT sheet normal (random sign per episode) to the brightest CT sample between
    ``jump_min`` and ``jump_max`` voxels, i.e. onto the next sheet rather than into the gap, so confidence learns
    from realistic sheet switches. Without a volume ``block`` the push is the middle of that range.
    """
    tc = cfg.train
    B, N = pos.shape[:2]
    with torch.no_grad():
        _, _, frame, ok = project(gt, valid, pos)
        length = torch.randint(1, N//2+1, (B, 1), device=pos.device, generator=generator)
        first = (torch.rand(B, 1, device=pos.device, generator=generator)*(N-length+1)).long()
        k = torch.arange(N, device=pos.device)[None]
        run = active[:, None] & ok & (k >= first) & (k < first+length)
        sign = torch.where(torch.rand(B, 1, device=pos.device, generator=generator) < .5, -1., 1.)
        normal = sign[..., None]*frame[..., 2]
        if block is None:
            distance = torch.full((B, N), (tc.jump_min+tc.jump_max)/2, device=pos.device)
        else:
            offsets = torch.arange(tc.jump_min, tc.jump_max+.5, 1., device=pos.device)
            probe = pos[:, :, None]+offsets[None, None, :, None]*normal[:, :, None]  # (B, N, K, 3)
            values = sample_block(block, origin, cfg.data.volume_level, probe[:, :, :, None])[:, 0, ..., 0]
            distance = offsets[values.argmax(-1)]
        delta = torch.where(run[..., None], distance[..., None]*normal, torch.zeros_like(pos))
    return pos+delta, run
