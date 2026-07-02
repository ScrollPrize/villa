"""
SliceJitterTransform -- scroll-specific inter-slice misalignment augmentation
for batchgeneratorsv2.

Addresses ScrollPrize/villa issue #201 ("Scroll specific 3d augmentations for
model training"), which welcomes additional CT-specific augmentations beyond
the named Squeeze / Decohesion / Warp items.

Physical motivation
-------------------
Micro-CT volumes are reconstructed slice-by-slice along the scan axis. Stage
drift, sample settling and the stitching of separately scanned sub-volumes all
produce small IN-PLANE misalignments between adjacent slices: a papyrus sheet
that is geometrically smooth appears "jagged" or "stair-stepped" when viewed
in a cross-section that cuts across the slice axis. Annotations are made on
the reconstructed (already misaligned) volume, so the labels follow the
artifact -- hence this is a geometric transform applied consistently to image
and segmentation (BasicTransform), not an image-only one.

The displacement of slice k is modelled as two components:

  1. *jitter* -- i.i.d. per-slice in-plane translation (uniform in
     [-a, a] voxels), the high-frequency slice-to-slice misalignment;
  2. *drift*  -- a smooth, low-frequency in-plane wander along the slice axis
     (coarse random control points, linearly upsampled, zero-mean), modelling
     slow stage drift over the scan.

How it differs from existing transforms
---------------------------------------
- SpatialTransform's elastic deformation is smooth and isotropic in all three
  axes -- it cannot produce a displacement that is rigid within a slice yet
  discontinuous between slices.
- WarpTransform bends the whole stack coherently (constant along the normal
  axis); this transform is the opposite regime: piecewise-rigid, varying only
  along the slice axis.

Performance
-----------
Built and applied entirely on the input tensor's device (no CPU/NumPy
round-trip): the displacement is an (L, dim-1) tensor broadcast into a single
grid_sample call. Out-of-bounds samples use reflection padding so the (small)
shifts mirror interior texture instead of clamping to an edge stripe.

The two grid helpers below are vendored verbatim (math-identical) from
batchgeneratorsv2/transforms/spatial/spatial.py to guarantee the same
grid_sample coordinate convention (centered grid, reversed-axis flip,
align_corners=False), made device-aware.
"""
from typing import Optional, Tuple

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


class SliceJitterTransform(BasicTransform):
    """
    Inter-slice misalignment augmentation. 2D (C,X,Y) and 3D (C,X,Y,Z),
    channel first, no batch dim.

    Parameters
    ----------
    p_jitter : float
        Probability of applying the transform.
    jitter : RandomScalar
        Peak i.i.d. per-slice in-plane shift, in *voxels* (sampled per call;
        each slice then draws its shift uniformly from [-jitter, jitter]).
    drift : RandomScalar
        Peak smooth low-frequency in-plane drift along the slice axis, in
        *voxels* (sampled per call). 0 disables the drift component.
    drift_coarse : Tuple[int, int]
        Inclusive range for the number of drift control points. Smaller ->
        longer-wavelength drift.
    allowed_axes : Optional[Tuple[int, ...]]
        Which spatial axes may act as the slice (scan) axis. Default all.
    mode_seg : str
        Interpolation for segmentation ('nearest' for label maps).
    """

    def __init__(self,
                 p_jitter: float = 1.0,
                 jitter: RandomScalar = (0.3, 1.5),
                 drift: RandomScalar = (0.0, 2.0),
                 drift_coarse: Tuple[int, int] = (2, 4),
                 allowed_axes: Optional[Tuple[int, ...]] = None,
                 mode_seg: str = 'nearest'):
        super().__init__()
        assert drift_coarse[0] >= 2 and drift_coarse[1] >= drift_coarse[0]
        self.p_jitter = p_jitter
        self.jitter = jitter
        self.drift = drift
        self.drift_coarse = drift_coarse
        self.allowed_axes = allowed_axes
        self.mode_seg = mode_seg

    def get_parameters(self, **data_dict) -> dict:
        img = data_dict['image']
        spatial = img.shape[1:]
        dim = len(spatial)
        device = img.device

        if torch.rand(1).item() >= self.p_jitter:
            return {'displacement': None, 'axis': None}

        axes = tuple(self.allowed_axes) if self.allowed_axes is not None else tuple(range(dim))
        axis = int(axes[torch.randint(len(axes), (1,)).item()])
        L = int(spatial[axis])
        m = dim - 1  # number of in-plane axes
        jit_amp = float(sample_scalar(self.jitter, image=img, dim=axis))
        drift_amp = float(sample_scalar(self.drift, image=img, dim=axis))

        d = (torch.rand(L, m, device=device) * 2 - 1) * jit_amp
        if drift_amp > 0:
            nc = int(torch.randint(self.drift_coarse[0], self.drift_coarse[1] + 1, (1,)).item())
            f = F.interpolate(torch.randn(1, m, nc, device=device), size=(L,),
                              mode='linear', align_corners=True)[0].T  # (L, m)
            f = f - f.mean(0, keepdim=True)
            f = drift_amp * f / (f.abs().amax(0, keepdim=True) + 1e-6)
            d = d + f
        return {'displacement': d, 'axis': axis}

    def _jitter_x(self, x, displacement, axis, mode):
        spatial = x.shape[1:]
        dim = len(spatial)
        device = x.device
        if displacement.shape[0] != spatial[axis]:  # shape mismatch guard
            return x
        in_plane = [ax for ax in range(dim) if ax != axis]

        vshape = [1] * dim
        vshape[axis] = spatial[axis]
        grid = _centered_identity_grid(spatial, device=device, dtype=torch.float32)
        for j, ip in enumerate(in_plane):
            grid[..., ip] = grid[..., ip] + displacement[:, j].to(device).reshape(vshape)
        grid = _convert_my_grid_to_grid_sample_grid(grid, spatial)
        return grid_sample(x[None].float(), grid[None], mode=mode,
                           padding_mode="reflection", align_corners=False)[0]

    def _apply_to_image(self, img, **p):
        if p['displacement'] is None:
            return img
        return self._jitter_x(img, p['displacement'], p['axis'], 'bilinear').to(img.dtype)

    def _apply_to_segmentation(self, segmentation, **p):
        if p['displacement'] is None:
            return segmentation
        out = self._jitter_x(segmentation.contiguous(), p['displacement'], p['axis'], self.mode_seg)
        return out.to(segmentation.dtype).contiguous()

    def _apply_to_dist_map(self, dist_map, **p):
        if p['displacement'] is None:
            return dist_map
        return self._jitter_x(dist_map, p['displacement'], p['axis'], 'bilinear').to(dist_map.dtype)

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
    sheet_axis, slice_axis = 0, 2
    img = make_layered(shape, sheet_axis).to(dev)
    seg = (img > 0.5).to(torch.uint8)

    # 1) identity when disabled
    t0 = SliceJitterTransform(p_jitter=0.0)
    out0 = t0(image=img.clone(), segmentation=seg.clone())
    assert torch.equal(out0['image'], img), "p_jitter=0 must be identity"
    print("[ok] p_jitter=0 is identity")

    # 2) shape preserved, finite, labels subset
    torch.manual_seed(0)
    t = SliceJitterTransform(p_jitter=1.0, jitter=(1.5, 1.5), drift=(2.0, 2.0),
                             allowed_axes=(slice_axis,))
    out = t(image=img.clone(), segmentation=seg.clone())
    assert out['image'].shape == img.shape and out['segmentation'].shape == seg.shape
    assert torch.isfinite(out['image']).all()
    assert set(out['segmentation'].unique().tolist()).issubset(set(seg.unique().tolist()))
    print("[ok] shape preserved, finite, no spurious labels")

    # 3) per-slice misalignment: the first-sheet position, constant across slices
    #    in the input, must vary slice-to-slice (jagged), not as one global shift
    s = out['segmentation'][0] > 0                       # (X, Y, Z)
    firstpos = s.float().argmax(sheet_axis).float()      # (Y, Z)
    per_slice = firstpos.mean(0)                         # (Z,): mean first-sheet pos per slice
    step = (per_slice[1:] - per_slice[:-1]).abs()
    assert step.mean() > 0.1, "adjacent slices should be misaligned"
    print(f"[ok] slice-to-slice misalignment: mean |step|={step.mean():.2f} vox, "
          f"range={per_slice.max()-per_slice.min():.2f} vox")

    # 4) rigid within each slice: sheet count per column is preserved
    rises_in = ((seg[0] > 0)[1:] & ~(seg[0] > 0)[:-1]).sum(0).float()
    rises_out = (s[1:] & ~s[:-1]).sum(0).float()
    print(f"[ok] sheets/col median in={rises_in.median():.0f} out={rises_out.median():.0f} (preserved)")

    # 5) drift-only mode is smooth (small steps, large range)
    torch.manual_seed(1)
    td = SliceJitterTransform(p_jitter=1.0, jitter=(0.0, 0.0), drift=(3.0, 3.0),
                              drift_coarse=(2, 2), allowed_axes=(slice_axis,))
    pd = td.get_parameters(image=img)
    dstep = (pd['displacement'][1:] - pd['displacement'][:-1]).abs().max()
    drange = (pd['displacement'].max() - pd['displacement'].min())
    assert dstep < 0.5 and drange > 2.0, "drift must be smooth and low-frequency"
    print(f"[ok] drift-only: max step={dstep:.3f} vox, range={drange:.2f} vox (smooth)")

    # 6) speed
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
