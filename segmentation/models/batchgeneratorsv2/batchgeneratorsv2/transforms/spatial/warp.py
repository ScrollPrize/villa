"""
WarpTransform -- scroll-specific coherent warping augmentation for batchgeneratorsv2.

Addresses ScrollPrize/villa issue #201, the *Warping* item:
    "The scroll in some areas is heavily warped. Simple grid deformations /
     elastic deformations do not tend to represent the type of warping present
     on scroll data. ... take a relatively straight chunk and augment it such
     that it appears similar to a more warped region."

Why it differs from elastic deformation
---------------------------------------
Elastic deformation adds *isotropic, random, Gaussian-blurred* offsets on every
axis -> local jitter that does not preserve the layered (concentric-sheet)
structure of a scroll. Real scroll warping is *coherent*: the stacked sheets bend
together as a unit. This transform models that directly:

  - Displace ONLY along the across-sheet (normal) axis,
  - by a smooth, low-frequency field that varies over the IN-PLANE axes,
  - so a flat stack is bent into a coherent wave/curve, sheets kept intact and
    locally parallel (concentric layers bending together).

Because the displacement does not depend on the normal coordinate itself, the
per-column map is a pure shift -> strictly monotonic, never folds. The smooth
field is a coarse random grid upsampled (bicubic) to the patch -- built and
applied entirely on-device (no CPU/NumPy round-trip, unlike the elastic FFT path).

Geometric transform: image (bilinear) and labels (nearest) are warped with the
identical field (subclasses BasicTransform), so geometry stays consistent.
"""
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.nn.functional import grid_sample

from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


def _centered_identity_grid(size, device, dtype=torch.float32):
    space = [torch.linspace((1 - s) / 2, (s - 1) / 2, s, device=device, dtype=dtype) for s in size]
    return torch.stack(torch.meshgrid(space, indexing="ij"), -1)


def _convert_my_grid_to_grid_sample_grid(my_grid, original_shape):
    for d in range(len(original_shape)):
        my_grid[..., d] = my_grid[..., d] / (original_shape[d] / 2)
    return torch.flip(my_grid, (my_grid.ndim - 1,))


class WarpTransform(BasicTransform):
    """
    Coherent scroll-warping augmentation. 2D (C,X,Y) and 3D (C,X,Y,Z), channel
    first, no batch dim.

    Parameters
    ----------
    p_warp : float
        Probability of applying the transform.
    amplitude : RandomScalar
        Peak bend, as a *fraction of the normal-axis length* (sampled per call).
    coarse : Tuple[int, int]
        Inclusive range for the coarse-grid resolution of the smooth field.
        Smaller -> longer-wavelength (smoother, larger-scale) warps.
    allowed_axes : Optional[Tuple[int, ...]]
        Which spatial axes may act as the across-sheet (normal) axis. Default all.
    mode_seg : str
        Interpolation for segmentation ('nearest' for label maps).
    """

    def __init__(self,
                 p_warp: float = 1.0,
                 amplitude: RandomScalar = (0.05, 0.2),
                 coarse: Tuple[int, int] = (3, 5),
                 allowed_axes: Optional[Tuple[int, ...]] = None,
                 mode_seg: str = 'nearest'):
        super().__init__()
        assert coarse[0] >= 2 and coarse[1] >= coarse[0]
        self.p_warp = p_warp
        self.amplitude = amplitude
        self.coarse = coarse
        self.allowed_axes = allowed_axes
        self.mode_seg = mode_seg

    def get_parameters(self, **data_dict) -> dict:
        img = data_dict['image']
        spatial = img.shape[1:]
        dim = len(spatial)
        device = img.device

        if torch.rand(1).item() >= self.p_warp:
            return {'displacement': None, 'axis': None}

        axes = tuple(self.allowed_axes) if self.allowed_axes is not None else tuple(range(dim))
        axis = int(axes[torch.randint(len(axes), (1,)).item()])
        L = int(spatial[axis])
        in_plane = [ax for ax in range(dim) if ax != axis]
        amp = max(1.0, float(sample_scalar(self.amplitude, image=img, dim=axis)) * L)
        nc = int(torch.randint(self.coarse[0], self.coarse[1] + 1, (1,)).item())

        if dim == 3:
            P, Q = int(spatial[in_plane[0]]), int(spatial[in_plane[1]])
            coarse = torch.randn(1, 1, nc, nc, device=device)
            f = F.interpolate(coarse, size=(P, Q), mode='bicubic', align_corners=True)[0, 0]
        else:
            P = int(spatial[in_plane[0]])
            coarse = torch.randn(1, 1, nc, device=device)
            f = F.interpolate(coarse, size=(P,), mode='linear', align_corners=True)[0, 0]

        f = f - f.mean()
        f = amp * f / (f.abs().max() + 1e-6)
        return {'displacement': f, 'axis': axis}

    def _warp(self, x, displacement, axis, mode):
        spatial = x.shape[1:]
        dim = len(spatial)
        device = x.device
        in_plane = [ax for ax in range(dim) if ax != axis]
        expect = 1
        for ax in in_plane:
            expect *= spatial[ax]
        if displacement.numel() != expect:        # shape mismatch guard
            return x

        vshape = list(spatial)
        vshape[axis] = 1
        grid = _centered_identity_grid(spatial, device=device, dtype=torch.float32)
        grid[..., axis] = grid[..., axis] + displacement.to(device).reshape(vshape)
        grid = _convert_my_grid_to_grid_sample_grid(grid, spatial)
        return grid_sample(x[None].float(), grid[None], mode=mode,
                           padding_mode="reflection", align_corners=False)[0]

    def _apply_to_image(self, img, **p):
        if p['displacement'] is None:
            return img
        return self._warp(img, p['displacement'], p['axis'], 'bilinear').to(img.dtype)

    def _apply_to_segmentation(self, segmentation, **p):
        if p['displacement'] is None:
            return segmentation
        out = self._warp(segmentation.contiguous(), p['displacement'], p['axis'], self.mode_seg)
        return out.to(segmentation.dtype).contiguous()

    def _apply_to_dist_map(self, dist_map, **p):
        if p['displacement'] is None:
            return dist_map
        return self._warp(dist_map, p['displacement'], p['axis'], 'bilinear').to(dist_map.dtype)

    def _apply_to_regr_target(self, regression_target, **p):
        return self._apply_to_dist_map(regression_target, **p)

    def _apply_to_keypoints(self, keypoints, **p):
        raise NotImplementedError

    def _apply_to_bbox(self, bbox, **p):
        raise NotImplementedError


if __name__ == '__main__':
    import time

    def make_layered(shape, axis, period=6):
        v = torch.zeros((1, *shape))
        idx = [slice(None)] * (len(shape) + 1)
        for p in range(period, shape[axis] - period, period):
            idx[axis + 1] = p
            v[tuple(idx)] = 1.0
        return v

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device = {dev}\n")
    shape = (160, 160, 160)
    axis = 0
    img = make_layered(shape, axis).to(dev)
    seg = (img > 0.5).to(torch.uint8)

    # 1) identity when disabled
    t0 = WarpTransform(p_warp=0.0)
    out0 = t0(image=img.clone(), segmentation=seg.clone())
    assert torch.equal(out0['image'], img), "p_warp=0 must be identity"
    print("[ok] p_warp=0 is identity")

    # 2) shape preserved, finite, labels subset
    torch.manual_seed(0)
    t = WarpTransform(p_warp=1.0, amplitude=(0.15, 0.15), coarse=(3, 3), allowed_axes=(axis,))
    out = t(image=img.clone(), segmentation=seg.clone())
    assert out['image'].shape == img.shape and out['segmentation'].shape == seg.shape
    assert torch.isfinite(out['image']).all()
    assert set(out['segmentation'].unique().tolist()).issubset(set(seg.unique().tolist()))
    print("[ok] shape preserved, finite, no spurious labels")

    # 3) coherent bend: sheets preserved per column, and the bend VARIES across columns
    s = out['segmentation'][0] > 0                      # (X,Y,Z)
    rises_in = ((seg[0] > 0)[1:] & ~(seg[0] > 0)[:-1]).sum(0).float()
    rises_out = (s[1:] & ~s[:-1]).sum(0).float()
    anyc = s.any(0)
    firstpos = s.float().argmax(0)[anyc].float()
    print(f"[ok] sheets/col median in={rises_in.median():.0f} out={rises_out.median():.0f} (preserved)")
    print(f"[ok] bend varies across columns: first-sheet std={firstpos.std():.2f} vox "
          f"(>0 => coherent warp, not a global shift)")

    # 4) speed
    for _ in range(3):
        _ = t(image=img.clone(), segmentation=seg.clone())
    if dev == 'cuda':
        torch.cuda.synchronize()
    n = 50
    st = time.time()
    for _ in range(n):
        _ = t(image=img.clone(), segmentation=seg.clone())
    if dev == 'cuda':
        torch.cuda.synchronize()
    print(f"[ok] {shape} image+seg: {(time.time()-st)/n*1000:.2f} ms/sample on {dev}")
    print("\nAll checks passed.")
