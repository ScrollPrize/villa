"""
DecohesionTransform -- scroll-specific "beam scatter / decohesion" augmentation
for batchgeneratorsv2.

Addresses ScrollPrize/villa issue #201, the *Decohesion* item:
    "in dense areas the beam is scattered, producing an image that looks
     'blurred' or 'smeared' from previous layers"

IMAGE-ONLY: decohesion is an imaging artifact -- it changes appearance, NOT
geometry, so it must not touch the segmentation/labels.
"""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform


class DecohesionTransform(ImageOnlyTransform):
    def __init__(self,
                 p_decohesion: float = 1.0,
                 tau: RandomScalar = (1.0, 3.0),
                 strength: RandomScalar = (0.3, 0.8),
                 density_window: int = 5,
                 density_modulated: bool = True,
                 allowed_axes: Optional[Tuple[int, ...]] = None):
        super().__init__()
        assert density_window % 2 == 1, "density_window must be odd"
        self.p_decohesion = p_decohesion
        self.tau = tau
        self.strength = strength
        self.density_window = density_window
        self.density_modulated = density_modulated
        self.allowed_axes = allowed_axes

    def get_parameters(self, **data_dict) -> dict:
        img = data_dict['image']
        dim = img.ndim - 1
        if torch.rand(1).item() >= self.p_decohesion:
            return {'apply': False}
        axes = tuple(self.allowed_axes) if self.allowed_axes is not None else tuple(range(dim))
        axis = int(axes[torch.randint(len(axes), (1,)).item()])
        tau = max(0.3, float(sample_scalar(self.tau, image=img, dim=axis)))
        strength = float(min(1.0, max(0.0, sample_scalar(self.strength, image=img, dim=axis))))
        K = int(min(img.shape[axis + 1], max(3, round(3 * tau) + 1)))
        return {'apply': True, 'axis': axis, 'tau': tau, 'strength': strength, 'K': K}

    @staticmethod
    def _causal_smear(img: torch.Tensor, taxis: int, k: torch.Tensor) -> torch.Tensor:
        C = img.shape[0]
        dim = img.ndim - 1
        K = int(k.shape[0])

        if K == 1:
            return k[0] * img

        if dim not in (2, 3):
            raise ValueError(f"_causal_smear supports 2D or 3D inputs, got dim={dim}")

        spatial_axis = taxis - 1
        if not (0 <= spatial_axis < dim):
            raise ValueError(f"taxis={taxis} out of range for img of shape {tuple(img.shape)}")

        pad = [0, 0] * dim
        pad_idx = (dim - 1 - spatial_axis) * 2
        pad[pad_idx] = K - 1
        x = F.pad(img.unsqueeze(0), pad, mode='constant', value=0.0)

        ks = [1] * dim
        ks[spatial_axis] = K
        w = torch.flip(k, dims=[0]).to(dtype=x.dtype, device=x.device)
        w = w.view([1, 1] + ks).expand(C, 1, *ks).contiguous()

        if dim == 3:
            out = F.conv3d(x, w, groups=C)
        else:
            out = F.conv2d(x, w, groups=C)
        return out.squeeze(0)

    def _density_weight(self, img: torch.Tensor, strength: float) -> torch.Tensor:
        if not self.density_modulated:
            return torch.full_like(img, strength)
        w = self.density_window
        pad = (w - 1) // 2
        x = img[None]
        if img.ndim - 1 == 3:
            dens = F.avg_pool3d(x, w, stride=1, padding=pad, count_include_pad=False)[0]
        else:
            dens = F.avg_pool2d(x, w, stride=1, padding=pad, count_include_pad=False)[0]
        dmin = dens.amin()
        dmax = dens.amax()
        dens_n = (dens - dmin) / (dmax - dmin + 1e-8)
        return (strength * dens_n).clamp_(0.0, 1.0)

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if not params.get('apply', False):
            return img
        device = img.device
        taxis = params['axis'] + 1
        K = params['K']
        k = torch.exp(-torch.arange(K, device=device, dtype=torch.float32) / params['tau'])
        k = k / k.sum()
        smeared = self._causal_smear(img.float(), taxis, k)
        w = self._density_weight(img.float(), params['strength'])
        out = (1.0 - w) * img.float() + w * smeared
        return out.to(img.dtype)


if __name__ == '__main__':
    import time
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device = {dev}\n")
    shape = (160, 160, 160)
    axis = 0
    img = torch.zeros((1, *shape), device=dev)
    img[:, :, shape[1] // 2:, :] = 0.3
    img[:, 40:44, :, :] += 1.0
    seg = (img > 0.5).to(torch.uint8)

    t0 = DecohesionTransform(p_decohesion=0.0)
    out0 = t0(image=img.clone(), segmentation=seg.clone())
    assert torch.equal(out0['image'], img), "p=0 must be identity"
    print("[ok] p_decohesion=0 is identity")

    torch.manual_seed(0)
    t = DecohesionTransform(p_decohesion=1.0, tau=(2.0, 2.0), strength=(0.8, 0.8),
                            allowed_axes=(axis,))
    out = t(image=img.clone(), segmentation=seg.clone())
    assert torch.equal(out['segmentation'], seg), "segmentation must be unchanged (ImageOnly)"
    assert torch.isfinite(out['image']).all()
    print("[ok] segmentation untouched, image finite")

    prof = out['image'][0].mean(dim=(1, 2))
    base = img[0].mean(dim=(1, 2))
    lead = (prof[36:40] - base[36:40]).abs().sum().item()
    trail = (prof[44:54] - base[44:54]).abs().sum().item()
    print(f"[ok] one-sided smear: trailing={trail:.3f} >> leading={lead:.3f}")

    diff = (out['image'] - img).abs()
    chg_dense = diff[:, :, shape[1] // 2:, :].mean().item()
    chg_sparse = diff[:, :, :shape[1] // 2, :].mean().item()
    print(f"[ok] density-modulated: dense change={chg_dense:.4f} > sparse={chg_sparse:.4f}")

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
    print(f"[ok] {shape}: {(time.time() - st) / n * 1000:.2f} ms/sample on {dev}")
    print("\nAll checks passed.")