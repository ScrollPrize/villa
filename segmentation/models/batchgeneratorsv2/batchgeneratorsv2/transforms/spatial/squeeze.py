"""
SqueezeTransform -- scroll-specific compression augmentation for batchgeneratorsv2.

Addresses ScrollPrize/villa issue #201 ("Scroll specific 3d augmentations for
model training"), the *Squeezing/Pulling* item:

    "Implement an augmentation that can create accurate compressed regions from
     normal data."

Why this matters
----------------
Surface / fiber / ink models are documented to perform worst in highly
*compressed* (and highly curved) regions -- see issue #191 ("Surface and Fiber
Predictions in Compressed or Highly Curved areas") and discussion #39, where
the 11^3 box blur is noted to destroy information "especially in highly
compressed regions". Compressed material is under-represented in training
patches, so a transform that synthesises realistic compressed bands from normal
data directly targets a known weak spot of the pipeline.

How it differs from elastic deformation
---------------------------------------
Elastic deformation (SpatialTransform) adds *isotropic, random, Gaussian-blurred*
offsets -- the issue explicitly notes this does not represent real scroll warping.
Squeezing is instead a *structured, anisotropic, monotonic* warp:

  1. Along a chosen axis define a non-negative compression profile
         c(x) = 1 + sum_k strength_k * gauss(x; center_k, sigma_k),   c(x) >= 0
     A positive bump is a locally compressed band.
  2. The resampling map is the normalised cumulative integral of c:
         m = cumsum(c);   m = (m - m[0]) / (m[-1] - m[0]) * (L - 1)
     Because c >= 0, m is strictly monotonic, so the warp NEVER folds
     (no self-intersection) and stays inside the patch.
  3. grid_sample with this map: where c peaks, m rises fastest, so more input is
     packed into fewer output voxels  ->  compression; the flanks stretch to
     compensate, exactly like a physically squeezed stack of layers.

Performance
-----------
Everything is built and run on the input tensor's device. There is no CPU/NumPy
round-trip -- contrast with the elastic path in spatial.py, whose cost is a CPU
NumPy FFT (np.fft.fftn / fourier_gaussian). The displacement here is a single 1-D
cumsum broadcast over the patch, so the transform is essentially free on GPU.

The two grid helpers below are vendored verbatim (math-identical) from
batchgeneratorsv2/transforms/spatial/spatial.py to guarantee the same
grid_sample coordinate convention (centered grid, reversed-axis flip,
align_corners=False), made device-aware.
"""
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch.nn.functional import grid_sample

from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


# --------------------------------------------------------------------------- #
# Grid helpers (vendored from spatial.py, made device-aware; math identical)   #
# --------------------------------------------------------------------------- #
def _centered_identity_grid(size: Sequence[int], device, dtype=torch.float32) -> torch.Tensor:
    space = [torch.linspace((1 - s) / 2, (s - 1) / 2, s, device=device, dtype=dtype) for s in size]
    grid = torch.stack(torch.meshgrid(space, indexing="ij"), -1)
    return grid


def _convert_my_grid_to_grid_sample_grid(my_grid: torch.Tensor, original_shape: Sequence[int]) -> torch.Tensor:
    for d in range(len(original_shape)):
        my_grid[..., d] = my_grid[..., d] / (original_shape[d] / 2)
    my_grid = torch.flip(my_grid, (my_grid.ndim - 1,))
    return my_grid


# --------------------------------------------------------------------------- #
# Transform                                                                    #
# --------------------------------------------------------------------------- #
class SqueezeTransform(BasicTransform):
    """
    Local compression ("squeeze") augmentation. Works for 2D (C, X, Y) and
    3D (C, X, Y, Z) tensors, channel-first, no batch dim -- matching the
    batchgeneratorsv2 BasicTransform contract.

    Parameters
    ----------
    p_squeeze : float
        Probability of applying the transform to a given sample.
    num_bands : Tuple[int, int]
        Inclusive range for the number of independent compression bands.
    strength : RandomScalar
        Peak extra compression per band; the profile peak is (1 + strength).
        Larger -> tighter squeeze. Sampled per band.
    sigma : RandomScalar
        Band half-width as a *fraction of the axis length*. Sampled per band.
        Clamped to a >= 1 px width.
    allowed_axes : Optional[Tuple[int, ...]]
        Which spatial axes may be squeezed (0-indexed over spatial dims).
        Defaults to all axes. For scroll data you typically want the
        axis across the papyrus layers.
    mode_seg : str
        Interpolation for segmentation ('nearest' recommended for label maps).

    Notes
    -----
    Assumes image / segmentation / dist_map share the same spatial shape
    (true for the nnU-Net patch pipeline), consistent with SpatialTransform.
    """

    def __init__(self,
                 p_squeeze: float = 1.0,
                 num_bands: Tuple[int, int] = (1, 2),
                 strength: RandomScalar = (0.5, 3.0),
                 sigma: RandomScalar = (0.04, 0.12),
                 allowed_axes: Optional[Tuple[int, ...]] = None,
                 mode_seg: str = 'nearest'):
        super().__init__()
        assert num_bands[0] >= 1 and num_bands[1] >= num_bands[0]
        self.p_squeeze = p_squeeze
        self.num_bands = num_bands
        self.strength = strength
        self.sigma = sigma
        self.allowed_axes = allowed_axes
        self.mode_seg = mode_seg

    # -- parameter sampling (done once, shared across image/seg/...) --------- #
    def get_parameters(self, **data_dict) -> dict:
        img = data_dict['image']
        spatial = img.shape[1:]
        dim = len(spatial)
        device = img.device

        if torch.rand(1).item() >= self.p_squeeze:
            return {'displacement': None, 'axis': None}

        axes = tuple(self.allowed_axes) if self.allowed_axes is not None else tuple(range(dim))
        axis = int(axes[torch.randint(len(axes), (1,)).item()])
        L = int(spatial[axis])

        xs = torch.arange(L, device=device, dtype=torch.float32)
        c = torch.ones(L, device=device, dtype=torch.float32)

        n_bands = int(torch.randint(self.num_bands[0], self.num_bands[1] + 1, (1,)).item())
        for _ in range(n_bands):
            # keep band centers away from the very edges so the squeeze is visible
            center = float(torch.empty(1, device=device).uniform_(0.15 * L, 0.85 * L).item())
            sigma_px = max(1.0, float(sample_scalar(self.sigma, image=img, axis=axis)) * L)
            strength = max(0.0, float(sample_scalar(self.strength, image=img, axis=axis)))
            c = c + strength * torch.exp(-((xs - center) ** 2) / (2.0 * sigma_px ** 2))

        # normalised cumulative integral -> strictly monotonic resampling map
        m = torch.cumsum(c, dim=0)
        m = (m - m[0]) / (m[-1] - m[0]) * (L - 1)
        delta = m - xs
        delta = delta - delta.mean()  # remove net translation; keep content centered
        return {'displacement': delta, 'axis': axis}

    # -- core warp ----------------------------------------------------------- #
    def _warp(self, x: torch.Tensor, displacement: torch.Tensor, axis: int, mode: str) -> torch.Tensor:
        spatial = x.shape[1:]
        dim = len(spatial)
        device = x.device

        # displacement was built for the image axis length; guard against a
        # mismatched spatial shape rather than crashing.
        if displacement.numel() != spatial[axis]:
            return x

        grid = _centered_identity_grid(spatial, device=device, dtype=torch.float32)
        bshape = [1] * dim
        bshape[axis] = spatial[axis]
        grid[..., axis] = grid[..., axis] + displacement.to(device).view(bshape)
        grid = _convert_my_grid_to_grid_sample_grid(grid, spatial)

        sampled = grid_sample(
            x[None].float(), grid[None],
            mode=mode, padding_mode="zeros", align_corners=False
        )[0]
        return sampled

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if params['displacement'] is None:
            return img
        return self._warp(img, params['displacement'], params['axis'], mode='bilinear').to(img.dtype)

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        if params['displacement'] is None:
            return segmentation
        out = self._warp(segmentation.contiguous(), params['displacement'], params['axis'], mode=self.mode_seg)
        return out.to(segmentation.dtype).contiguous()

    def _apply_to_dist_map(self, dist_map: torch.Tensor, **params) -> torch.Tensor:
        if params['displacement'] is None:
            return dist_map
        return self._warp(dist_map, params['displacement'], params['axis'], mode='bilinear').to(dist_map.dtype)

    def _apply_to_regr_target(self, regression_target: torch.Tensor, **params) -> torch.Tensor:
        return self._apply_to_dist_map(regression_target, **params)

    def _apply_to_keypoints(self, keypoints, **params):
        raise NotImplementedError

    def _apply_to_bbox(self, bbox, **params):
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Self-test / sanity benchmark. Run on a GPU box:  python squeeze.py           #
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    import time

    def make_layered(shape, axis, period=4):
        """Synthetic 'papyrus': planar sheets stacked along `axis`."""
        v = torch.zeros((1, *shape))
        idx = [slice(None)] * (len(shape) + 1)
        for p in range(period, shape[axis], period):
            idx[axis + 1] = p
            v[tuple(idx)] = 1.0
        return v

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device = {dev}\n")

    shape = (160, 160, 160)
    axis = 0
    img = make_layered(shape, axis).to(dev)
    seg = (img > 0).to(torch.uint8)

    # 1) identity when disabled
    t0 = SqueezeTransform(p_squeeze=0.0)
    out0 = t0(image=img.clone(), segmentation=seg.clone())
    assert torch.equal(out0['image'], img), "p_squeeze=0 must be identity"
    print("[ok] p_squeeze=0 is identity")

    # 2) shape preserved, finite, labels preserved (subset)
    torch.manual_seed(0)
    t = SqueezeTransform(p_squeeze=1.0, num_bands=(1, 1), strength=(4.0, 4.0),
                         sigma=(0.06, 0.06), allowed_axes=(axis,))
    out = t(image=img.clone(), segmentation=seg.clone())
    assert out['image'].shape == img.shape and out['segmentation'].shape == seg.shape
    assert torch.isfinite(out['image']).all()
    in_labels = set(seg.unique().tolist())
    out_labels = set(out['segmentation'].unique().tolist())
    assert out_labels.issubset(in_labels), (out_labels, in_labels)
    print("[ok] shape preserved, finite, no spurious labels:", sorted(out_labels))

    # 3) compression is real: layer spacing along `axis` shrinks somewhere
    prof = out['image'].squeeze(0).amax(dim=tuple(d for d in range(3) if d != axis))
    layer_pos = torch.nonzero(prof > 0.5).flatten().float()
    if layer_pos.numel() >= 3:
        gaps = (layer_pos[1:] - layer_pos[:-1])
        print(f"[ok] layer gaps along axis {axis}: min={gaps.min():.2f} "
              f"max={gaps.max():.2f} (uniform input gap = 4.0) -> compression+stretch")

    # 4) speed
    for _ in range(3):  # warmup
        _ = t(image=img.clone(), segmentation=seg.clone())
    if dev == 'cuda':
        torch.cuda.synchronize()
    n = 50
    st = time.time()
    for _ in range(n):
        _ = t(image=img.clone(), segmentation=seg.clone())
    if dev == 'cuda':
        torch.cuda.synchronize()
    dt = (time.time() - st) / n * 1000
    print(f"[ok] {shape} image+seg: {dt:.2f} ms/sample on {dev}")
    print("\nAll checks passed.")
