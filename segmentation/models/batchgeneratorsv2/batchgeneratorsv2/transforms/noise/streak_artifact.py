from typing import List, Optional

import torch

from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform


class StreakArtifactTransform(ImageOnlyTransform):
    """
    Simulates CT streak artifacts on the GPU (directional / undersampling streaks).

    Physics
    -------
    Streak artifacts are straight bright/dark lines that cross the reconstructed
    axial plane. They arise from sparse-angle / limited-angle acquisition
    (aliasing streaks) and from highly attenuating inclusions. Like rings, they
    live in the axial (reconstruction) plane and are extruded along the scan axis.
    Unlike rings (concentric) they are linear: a family of parallel lines at some
    in-plane orientation. They create spurious straight structures the network can
    mistake for sheet boundaries; training with them as an augmentation builds
    robustness.

    Implementation
    --------------
    Per call: pick an in-plane orientation theta, project the plane coordinates
    onto it (t = a·cosθ + b·sinθ), build a 1-D profile of Gaussian-shaped streaks
    over t, and gather it onto the volume by the per-voxel projected coordinate
    (linear interpolation), broadcast along the scan axis. Amplitudes are in
    per-channel std units by default. Image-only; (C,X,Y) and (C,X,Y,Z); singleton
    spatial axes are never used for the plane (no-op if < 2 non-singleton axes).

    Parameters
    ----------
    num_streaks : RandomScalar
        Number of streak lines (rounded to int).
    intensity : RandomScalar
        Streak amplitude magnitude (std units if relative_to_std). Sign drawn.
    streak_width : RandomScalar
        Gaussian sigma of a streak, in voxels.
    p_negative : float
        Probability a streak is dark.
    relative_to_std : bool
        Scale amplitudes by per-channel spatial std.
    streak_axes : Optional[List[int]]
        Candidate scan/extrusion axes (0-based spatial). None -> all non-singleton.
    p_per_channel : float
    synchronize_channels : bool
    benchmark : bool
        Cache plane coordinate grids keyed by shape/device.
    """

    def __init__(self,
                 num_streaks: RandomScalar = (1, 6),
                 intensity: RandomScalar = (0.05, 0.3),
                 streak_width: RandomScalar = (0.5, 2.0),
                 p_negative: float = 0.5,
                 relative_to_std: bool = True,
                 streak_axes: Optional[List[int]] = None,
                 p_per_channel: float = 1.0,
                 synchronize_channels: bool = False,
                 benchmark: bool = True):
        super().__init__()
        self.num_streaks = num_streaks
        self.intensity = intensity
        self.streak_width = streak_width
        self.p_negative = p_negative
        self.relative_to_std = relative_to_std
        self.streak_axes = streak_axes
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
            cand = self.streak_axes if self.streak_axes is not None else list(range(ndim))
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
        k = max(1, int(round(sample_scalar(self.num_streaks))))
        streaks = []
        for _ in range(k):
            mag = abs(float(sample_scalar(self.intensity)))
            sign = -1.0 if torch.rand(1).item() < self.p_negative else 1.0
            streaks.append({"pos": float(torch.rand(1).item()),  # fraction along t-range
                            "amp": sign * mag,
                            "width": max(1e-3, float(sample_scalar(self.streak_width)))})
        return {"rot_axis": rot_axis, "plane_axes": plane_axes, "theta": theta, "streaks": streaks}

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
        for s in spec["streaks"]:
            tc = tmin + s["pos"] * span
            w = s["width"]
            profile += s["amp"] * torch.exp(-((grid - tc) ** 2) / (2.0 * w * w))
        tf = (t - tmin).reshape(-1)
        lo = torch.floor(tf).long().clamp_(0, L - 2)
        frac = tf - lo.to(dtype)
        field = (profile[lo] * (1.0 - frac) + profile[lo + 1] * frac).reshape(H, W)
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
    t = StreakArtifactTransform(num_streaks=(2, 6), intensity=(0.1, 0.4), p_per_channel=1.0)
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
