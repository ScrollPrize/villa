from typing import List, Optional

import torch

from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform


class SeamArtifactTransform(ImageOnlyTransform):
    """
    Simulates CT grid-scan / tiling SEAM artifacts on the GPU.

    Physics
    -------
    Herculaneum scrolls are larger than the synchrotron detector field of view,
    so they are scanned with a "grid scan" technique and the high-resolution
    parallel-beam projections are tiled (stitched) together before / during
    filtered backprojection. Adjacent tiles can carry a small relative gain /
    offset mismatch, leaving a faint planar intensity STEP at the stitch boundary;
    across many tiles this accumulates into a low-amplitude staircase. A
    segmentation network that has never seen such a step can mistake the seam for
    a sheet boundary, or lose the surface as it crosses one. Training with
    simulated seams builds robustness to exactly the cube-boundary discontinuities
    the downstream meshing has to weld across.

    Why this is not an existing transform
    -------------------------------------
    The discontinuity is the point. `InhomogeneousSliceIlluminationTransform`
    applies a SMOOTH multiplicative illumination field; a seam is a C0 step (high
    spatial frequency localized at a plane). `BlankRectangleTransform` makes filled
    boxes, `SmearTransform` makes a directional blur, and brightness/gamma shift the
    whole patch. None of them produce a sharp planar level change at an arbitrary
    in-plane orientation. The accompanying ablation tests this directly: it is a
    fair, magnitude-matched comparison of this step against a smooth illumination
    field, and the transform is only worth adding if the step specifically helps.

    Implementation
    --------------
    Reuses the projected-coordinate machinery of the ring/streak transforms. Per
    call: pick an in-plane orientation theta, project plane coordinates onto it
    (t = a*cos(theta) + b*sin(theta)), build a 1-D profile that is a cumulative sum
    of smoothed Heaviside steps over t (a staircase), mean-center it so it is a
    planar redistribution rather than a global brightness shift, and gather it onto
    the volume by the per-voxel projected coordinate (linear interpolation),
    broadcast along the scan axis. Amplitudes are in per-channel std units by
    default (consistent with the relative / unitless reconstruction values and with
    z-scored nnU-Net inputs). Image-only; handles (C,X,Y) and (C,X,Y,Z); singleton
    spatial axes are never used to define the plane (no-op if < 2 non-singleton
    spatial axes).

    Parameters
    ----------
    num_seams : RandomScalar
        Number of seam planes (rounded to int) -> number of staircase steps.
    intensity : RandomScalar
        Per-step magnitude (std units if relative_to_std). Sign drawn per step.
    seam_softness : RandomScalar
        Width of the smoothed step transition, in voxels (0 -> nearly hard step,
        larger -> a gentler ramp). Real recon seams are 1-3 voxels.
    p_negative : float
        Probability a step is downward (darker beyond the seam).
    relative_to_std : bool
        Scale amplitudes by per-channel spatial std.
    seam_axes : Optional[List[int]]
        Candidate scan/extrusion axes (0-based spatial). None -> all non-singleton.
    p_per_channel : float
    synchronize_channels : bool
    benchmark : bool
        Cache plane coordinate grids keyed by shape/device.
    """

    def __init__(self,
                 num_seams: RandomScalar = (1, 3),
                 intensity: RandomScalar = (0.05, 0.3),
                 seam_softness: RandomScalar = (0.5, 2.5),
                 p_negative: float = 0.5,
                 relative_to_std: bool = True,
                 seam_axes: Optional[List[int]] = None,
                 p_per_channel: float = 1.0,
                 synchronize_channels: bool = False,
                 benchmark: bool = True):
        super().__init__()
        self.num_seams = num_seams
        self.intensity = intensity
        self.seam_softness = seam_softness
        self.p_negative = p_negative
        self.relative_to_std = relative_to_std
        self.seam_axes = seam_axes
        self.p_per_channel = p_per_channel
        self.synchronize_channels = synchronize_channels
        self.benchmark = benchmark
        self._grid_cache = {}

    def get_parameters(self, **data_dict) -> dict:
        img = data_dict["image"]
        c = img.shape[0]
        spatial = img.shape[1:]
        ndim = len(spatial)
        valid = [i for i in range(ndim) if spatial[i] > 1]
        if len(valid) < 2:
            return {"apply_idx": None}
        apply = torch.rand(c, device=img.device) < self.p_per_channel
        idx = apply.nonzero(as_tuple=False).flatten()
        n = int(idx.numel())
        if n == 0:
            return {"apply_idx": None}

        if ndim == 2 or len(valid) == 2:
            cand_rot = None
        else:
            cand = self.seam_axes if self.seam_axes is not None else list(range(ndim))
            cand_rot = [a for a in cand if a in valid] or list(valid)

        n_specs = 1 if self.synchronize_channels else n
        specs = [self._sample_one(valid, cand_rot, ndim) for _ in range(n_specs)]
        return {"apply_idx": idx, "specs": specs}

    def _sample_one(self, valid, cand_rot, ndim) -> dict:
        if ndim == 2:
            rot_axis = None
            plane_axes = (valid[0], valid[1])
        elif cand_rot is None:
            plane_axes = (valid[0], valid[1])
            rem = [a for a in range(ndim) if a not in valid]
            rot_axis = rem[0] if rem else None
        else:
            rot_axis = int(cand_rot[int(torch.randint(len(cand_rot), (1,)).item())])
            plane = [a for a in valid if a != rot_axis]
            plane_axes = (plane[0], plane[1])

        theta = float(torch.rand(1).item()) * 3.141592653589793  # [0, pi)
        k = max(1, int(round(sample_scalar(self.num_seams))))
        seams = []
        for _ in range(k):
            mag = abs(float(sample_scalar(self.intensity)))
            sign = -1.0 if torch.rand(1).item() < self.p_negative else 1.0
            seams.append({"pos": float(torch.rand(1).item()),  # fraction along t-range
                          "amp": sign * mag,
                          "soft": max(1e-3, float(sample_scalar(self.seam_softness)))})
        return {"rot_axis": rot_axis, "plane_axes": plane_axes, "theta": theta, "seams": seams}

    def _coords(self, H, W, device, dtype):
        key = (H, W, device, dtype)
        if self.benchmark and key in self._grid_cache:
            return self._grid_cache[key]
        a = torch.arange(H, device=device, dtype=dtype).view(H, 1)
        b = torch.arange(W, device=device, dtype=dtype).view(1, W)
        if self.benchmark:
            self._grid_cache[key] = (a, b)
        return a, b

    def _build_field(self, spec, spatial, device, dtype):
        a0, a1 = spec["plane_axes"]
        H, W = spatial[a0], spatial[a1]
        a, b = self._coords(H, W, device, dtype)
        import math
        t = a * math.cos(spec["theta"]) + b * math.sin(spec["theta"])  # (H,W)
        tmin = float(t.min().item())
        tmax = float(t.max().item())
        span = tmax - tmin
        if span <= 0:
            return None
        L = int(span) + 2
        grid = torch.arange(L, device=device, dtype=dtype) + tmin
        profile = torch.zeros(L, device=device, dtype=dtype)
        for s in spec["seams"]:
            tc = tmin + s["pos"] * span
            soft = s["soft"]
            # smoothed Heaviside step (sigmoid): 0 -> amp across ~few*soft voxels
            profile += s["amp"] * torch.sigmoid((grid - tc) / soft)
        tf = (t - tmin).reshape(-1)
        lo = torch.floor(tf).long().clamp_(0, L - 2)
        frac = tf - lo.to(dtype)
        field = (profile[lo] * (1.0 - frac) + profile[lo + 1] * frac).reshape(H, W)
        field = field - field.mean()  # planar redistribution, not a global brightness shift
        view = [1] * len(spatial)
        view[a0] = H
        view[a1] = W
        return field.view(view)

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        idx = params.get("apply_idx")
        if idx is None:
            return img
        spatial = img.shape[1:]
        device, dtype = img.device, img.dtype
        specs = params["specs"]
        for j, ch in enumerate(idx.tolist()):
            spec = specs[0] if self.synchronize_channels else specs[j]
            field = self._build_field(spec, spatial, device, dtype)
            if field is None:
                continue
            if self.relative_to_std:
                sc = img[ch].std()
                if not torch.isfinite(sc) or sc <= 0:
                    sc = torch.ones((), device=device, dtype=dtype)
                field = field * sc
            img[ch] = img[ch] + field
        return img


if __name__ == "__main__":
    from time import time
    import numpy as np
    torch.set_num_threads(1)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = SeamArtifactTransform(num_seams=(1, 3), intensity=(0.1, 0.4), p_per_channel=1.0)
    out = t(image=torch.randn((1, 192, 192, 64), device=dev))["image"]
    assert out.shape == (1, 192, 192, 64) and torch.isfinite(out).all()
    times = []
    for _ in range(200):
        d = {"image": torch.randn((1, 192, 192, 64), device=dev)}
        if dev == "cuda":
            torch.cuda.synchronize()
        st = time(); t(**d)
        if dev == "cuda":
            torch.cuda.synchronize()
        times.append(time() - st)
    print(f"{dev}: {np.mean(times) * 1e3:.3f} ms/call (192x192x64)")
