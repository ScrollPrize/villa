import torch

def interp1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor, dim: int=-1, extrapolate: str='const') -> torch.Tensor:
    # See https://github.com/pytorch/pytorch/issues/50334
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])
    # indices = torch.sum(x[None, ...] >= xp.view(-1, *[1] * x.ndim), dim=0) - 1
    indices = torch.searchsorted(xp.squeeze(-1), x) - 1
    indices = torch.clamp(indices, 0, len(m) - 1)
    return m[indices] * x[..., None] + b[indices]

