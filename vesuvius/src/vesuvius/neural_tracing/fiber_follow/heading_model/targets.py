"""The heading target: the crop axis through the head that keeps the upcoming fiber most centered in crop."""
import numpy as np
import torch


def in_crop_heading(future, init, steps=200, temperature=.05, lr=.02):
    """Max-margin crop axis: minimizes the largest lateral distance of the future fiber from the axis.

    future: (B, K, 3) annotated fiber points ahead of the head, relative to it; init: (B, 3) forward directions.
    A smooth maximum (``temperature`` in voxels) is minimized jointly for the batch on the unit sphere.
    """
    f = torch.as_tensor(np.asarray(future), dtype=torch.float64)
    h = torch.nn.Parameter(torch.as_tensor(np.asarray(init), dtype=torch.float64).clone())
    opt = torch.optim.Adam([h], lr=lr)
    for _ in range(steps):
        u = torch.nn.functional.normalize(h, dim=-1)
        lateral = (f-(f*u[:, None]).sum(-1, keepdim=True)*u[:, None]).norm(dim=-1)
        loss = (temperature*torch.logsumexp(lateral/temperature, dim=-1)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.nn.functional.normalize(h.detach(), dim=-1).numpy()


def lateral_extent(future, heading, forward=None):
    """Largest lateral distance of future fiber points (within ``forward`` along the heading) from the axis."""
    future = np.asarray(future)
    along = future @ heading
    lateral = np.linalg.norm(future-along[:, None]*heading, axis=-1)
    keep = along <= forward if forward is not None else np.ones(len(along), bool)
    return float(lateral[keep].max()) if keep.any() else 0.
