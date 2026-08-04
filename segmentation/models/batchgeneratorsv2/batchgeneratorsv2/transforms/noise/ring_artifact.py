from typing import List, Optional, Tuple

import torch

from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform


class RingArtifactTransform(ImageOnlyTransform):
    """
    Simulates CT ring artifacts directly on the GPU.

    Physics
    -------
    In reconstructed CT volumes, a miscalibrated or defective detector element
    produces a concentric ring centered on the center of rotation, lying in the
    axial (reconstruction) plane and extruded along the rotation/scan axis,
    consistent across all slices stacked along that axis. Several defective
    elements produce several rings at different radii. The perturbation is
    additive in the reconstructed intensity, narrow (a few voxels wide) and can
    be brighter or darker than the surrounding tissue.

    This is a strong domain-specific augmentation for scroll CT, where ring
    artifacts are a real and common acquisition artifact that the network must
    learn to ignore when segmenting papyrus sheets.

    Implementation
    --------------
    Per call we build a 1-D radial perturbation profile ``p(r)`` as a sum of
    Gaussian-shaped rings, then map it onto the volume by gathering ``p`` at the
    per-voxel radius (linear interpolation). Cost of the profile build scales
    with the number of rings; cost of the field evaluation is a single gather
    over the axial plane, broadcast (not materialized) along the rotation axis.
    Amplitudes are expressed in per-channel standard-deviation units by default,
    so the transform behaves consistently regardless of intensity normalization
    (e.g. nnU-Net z-scored inputs).

    Accepts (C, X, Y) and (C, X, Y, Z) inputs. Singleton spatial axes are never
    used to define the axial plane (avoids a degenerate radius map on pseudo-2D
    patches); if fewer than two non-singleton spatial axes exist the transform
    is a no-op.

    Parameters
    ----------
    num_rings : RandomScalar
        Number of rings to draw per affected channel (rounded to int).
    intensity : RandomScalar
        Ring amplitude magnitude. In std units when ``relative_to_std`` (default),
        otherwise in absolute intensity units. The sign is drawn separately.
    ring_width : RandomScalar
        Gaussian sigma of a ring, in voxels.
    radius_range : Tuple[float, float]
        Fraction of the maximum in-plane radius where ring centers may fall.
        Kept strictly inside (0, 1) so a ring never degenerates into a central
        blob (r -> 0) or a corner-only arc (r -> r_max).
    center_offset : RandomScalar
        Center-of-rotation jitter as a fraction of the plane half-size, applied
        independently to each in-plane axis.
    p_negative : float
        Probability that a given ring is dark (negative amplitude).
    relative_to_std : bool
        If True, amplitudes are multiplied by the per-channel spatial std.
    ring_axes : Optional[List[int]]
        Candidate rotation/scan axes (spatial-axis indices, 0-based) along which
        rings are extruded. One is sampled per affected channel from the
        non-singleton subset. None -> all non-singleton spatial axes are
        candidates. Ignored for 2-D input.
        For anisotropic data with a distinct acquisition axis (e.g. scroll CT,
        where the scan/rotation axis is the long axis), set this to that axis so
        rings lie in the true reconstruction plane; leaving it None randomizes
        the plane across all spatial axes, which on anisotropic patches can place
        artifacts in non-physical orientations.
    p_per_channel : float
        Probability of applying the transform to each channel.
    synchronize_channels : bool
        If True, identical rings are applied to all affected channels (physically
        natural: same detector defects affect every reconstructed channel).
    benchmark : bool
        If True, reuse cached coordinate grids keyed by plane shape/device only
        (skips per-call allocation); set False if input shapes vary wildly.
    """

    def __init__(self,
                 num_rings: RandomScalar = (1, 5),
                 intensity: RandomScalar = (0.05, 0.3),
                 ring_width: RandomScalar = (0.5, 3.0),
                 radius_range: Tuple[float, float] = (0.1, 0.9),
                 center_offset: RandomScalar = (-0.05, 0.05),
                 p_negative: float = 0.5,
                 relative_to_std: bool = True,
                 ring_axes: Optional[List[int]] = None,
                 p_per_channel: float = 1.0,
                 synchronize_channels: bool = False,
                 benchmark: bool = True):
        super().__init__()
        self.num_rings = num_rings
        self.intensity = intensity
        self.ring_width = ring_width
        self.radius_range = radius_range
        self.center_offset = center_offset
        self.p_negative = p_negative
        self.relative_to_std = relative_to_std
        self.ring_axes = ring_axes
        self.p_per_channel = p_per_channel
        self.synchronize_channels = synchronize_channels
        self.benchmark = benchmark
        # cache: (H, W, device, dtype) -> (xx, yy) integer coordinate grids
        self._grid_cache = {}

    # ------------------------------------------------------------------ params
    def get_parameters(self, **data_dict) -> dict:
        img = data_dict["image"]
        c = img.shape[0]
        spatial = img.shape[1:]
        ndim = len(spatial)

        # non-singleton spatial axes only (skip singletons; see docstring)
        valid_axes = [i for i in range(ndim) if spatial[i] > 1]
        if len(valid_axes) < 2:
            return {"apply_idx": None}

        apply = torch.rand(c, device=img.device) < self.p_per_channel
        idx = apply.nonzero(as_tuple=False).flatten()
        n = int(idx.numel())
        if n == 0:
            return {"apply_idx": None}

        # Candidate rotation axes. The axial plane must be two NON-singleton
        # axes, so a rotation axis is only a free choice when all three spatial
        # axes are non-singleton. With exactly two valid axes the plane is
        # forced and the rotation axis is the remaining (singleton) axis.
        if ndim == 2 or len(valid_axes) == 2:
            cand_rot = None  # forced; resolved per-channel
        else:  # ndim == 3 and all three axes non-singleton
            cand = self.ring_axes if self.ring_axes is not None else list(range(ndim))
            cand_rot = [a for a in cand if a in valid_axes]
            if len(cand_rot) == 0:
                cand_rot = list(valid_axes)

        n_specs = 1 if self.synchronize_channels else n
        specs = [self._sample_one_channel(valid_axes, cand_rot, ndim) for _ in range(n_specs)]

        return {"apply_idx": idx, "num_apply": n, "specs": specs}

    def _sample_one_channel(self, valid_axes, cand_rot, ndim) -> dict:
        # The axial plane is always two non-singleton axes; the rotation axis
        # (extrusion direction) is whatever spatial axis remains.
        if ndim == 2:
            rot_axis = None
            plane_axes = (valid_axes[0], valid_axes[1])
        elif cand_rot is None:  # exactly two valid axes -> plane forced
            plane_axes = (valid_axes[0], valid_axes[1])
            rem = [a for a in range(ndim) if a not in valid_axes]
            rot_axis = rem[0] if rem else None
        else:  # free choice among non-singleton axes
            rot_axis = int(cand_rot[int(torch.randint(len(cand_rot), (1,)).item())])
            plane = [a for a in valid_axes if a != rot_axis]
            plane_axes = (plane[0], plane[1])

        k = max(1, int(round(sample_scalar(self.num_rings))))
        rings = []
        for _ in range(k):
            mag = abs(float(sample_scalar(self.intensity)))
            sign = -1.0 if torch.rand(1).item() < self.p_negative else 1.0
            rings.append({
                "r_frac": sample_scalar(self.radius_range),     # in [0, 1] of r_max
                "amp": sign * mag,
                "width": max(1e-3, float(sample_scalar(self.ring_width))),
            })
        return {
            "rot_axis": rot_axis,
            "plane_axes": plane_axes,
            "rings": rings,
            "off0": float(sample_scalar(self.center_offset)),
            "off1": float(sample_scalar(self.center_offset)),
        }

    # ------------------------------------------------------------------- apply
    def _coords(self, H, W, device, dtype):
        key = (H, W, device, dtype)
        if self.benchmark and key in self._grid_cache:
            return self._grid_cache[key]
        yy = torch.arange(H, device=device, dtype=dtype).view(H, 1)
        xx = torch.arange(W, device=device, dtype=dtype).view(1, W)
        if self.benchmark:
            self._grid_cache[key] = (yy, xx)
        return yy, xx

    def _build_field(self, spec, spatial, device, dtype) -> torch.Tensor:
        a0, a1 = spec["plane_axes"]
        H, W = spatial[a0], spatial[a1]
        yy, xx = self._coords(H, W, device, dtype)

        cy = (H - 1) / 2.0 * (1.0 + spec["off0"])
        cx = (W - 1) / 2.0 * (1.0 + spec["off1"])
        radius = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)  # (H, W)
        r_max = float(radius.max().item())
        if r_max <= 0:
            return None

        # 1-D radial profile p(r), r = 0 .. r_max
        L = int(r_max) + 2
        rgrid = torch.arange(L, device=device, dtype=dtype)
        profile = torch.zeros(L, device=device, dtype=dtype)
        for ring in spec["rings"]:
            r0 = ring["r_frac"] * r_max
            w = ring["width"]
            profile += ring["amp"] * torch.exp(-((rgrid - r0) ** 2) / (2.0 * w * w))

        # gather profile at per-voxel radius with linear interpolation
        rf = radius.reshape(-1)
        lo = torch.floor(rf).long().clamp_(0, L - 2)
        frac = (rf - lo.to(dtype))
        field = profile[lo] * (1.0 - frac) + profile[lo + 1] * frac
        field = field.reshape(H, W)

        # reshape to broadcast over channel-stripped spatial layout
        view = [1] * len(spatial)
        view[a0] = H
        view[a1] = W
        return field.view(view)  # broadcasts along the rotation axis for free

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
                scale = img[ch].std()
                if not torch.isfinite(scale) or scale <= 0:
                    scale = torch.ones((), device=device, dtype=dtype)
                field = field * scale
            img[ch] += field
        return img


if __name__ == "__main__":
    from time import time
    import numpy as np

    torch.set_num_threads(1)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rat = RingArtifactTransform(num_rings=(2, 6), intensity=(0.1, 0.4), p_per_channel=1.0)

    # warmup + correctness
    out = rat(image=torch.randn((1, 192, 192, 64), device=dev))["image"]
    assert out.shape == (1, 192, 192, 64) and torch.isfinite(out).all()

    times = []
    for _ in range(200):
        data_dict = {"image": torch.randn((1, 192, 192, 64), device=dev)}
        if dev == "cuda":
            torch.cuda.synchronize()
        st = time()
        _ = rat(**data_dict)
        if dev == "cuda":
            torch.cuda.synchronize()
        times.append(time() - st)
    print(f"{dev}: {np.mean(times) * 1e3:.3f} ms/call  (192x192x64)")
