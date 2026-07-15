import os

import torch


def maybe_compile(fn):
    # torch.compile the hot pure-tensor helpers when FIT_SPIRAL_COMPILE=1.
    # Inductor fuses the elementwise chains (fwd and generated bwd) but uses
    # FMA contraction, so results shift at the last-ulp level; off by default.
    if os.environ.get('FIT_SPIRAL_COMPILE', '0') == '1':
        # The training loop backwards each loss family with retain_graph=True;
        # donated buffers assume single-use backward graphs and hard-error.
        import torch._functorch.config
        torch._functorch.config.donated_buffer = False
        return torch.compile(fn, dynamic=True)
    return fn


@maybe_compile
def expm_2x2(L):
    # Closed-form matrix exponential for (..., 2, 2) matrices:
    # exp(L) = e^m (cosh(s) I + sinh(s)/s (L - m I)), where
    # m = tr(L)/2 and s^2 = ((a - d)/2)^2 + bc.
    a, b = L[..., 0, 0], L[..., 0, 1]
    c, d = L[..., 1, 0], L[..., 1, 1]
    m = 0.5 * (a + d)
    s2 = (0.5 * (a - d)) ** 2 + b * c
    small = s2.abs() < 1e-8
    s = torch.where(small, torch.ones_like(s2), s2).abs().sqrt()
    pos = s2 >= 0
    cosh_term = torch.where(small, 1.0 + s2 / 2.0, torch.where(pos, torch.cosh(s), torch.cos(s)))
    sinc_term = torch.where(small, 1.0 + s2 / 6.0, torch.where(pos, torch.sinh(s), torch.sin(s)) / s)
    em = torch.exp(m)
    f_diag = em * cosh_term
    f_off = em * sinc_term
    out = torch.empty_like(L)
    out[..., 0, 0] = f_diag + f_off * (a - m)
    out[..., 0, 1] = f_off * b
    out[..., 1, 0] = f_off * c
    out[..., 1, 1] = f_diag + f_off * (d - m)
    return out


def interp1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor, dim: int=-1, extrapolate: str='const') -> torch.Tensor:
    # See https://github.com/pytorch/pytorch/issues/50334
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])
    # indices = torch.sum(x[None, ...] >= xp.view(-1, *[1] * x.ndim), dim=0) - 1
    indices = torch.searchsorted(xp.squeeze(-1), x) - 1
    indices = torch.clamp(indices, 0, len(m) - 1)
    return m[indices] * x[..., None] + b[indices]
