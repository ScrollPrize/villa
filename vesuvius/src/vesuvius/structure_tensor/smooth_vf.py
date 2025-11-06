# smooth_vf.py
import warnings
import torch
import torch.nn as nn
from typing import Optional

from ._compile_utils import _maybe_compile_function

class VectorFieldModule(nn.Module):
    """
    Given a 3×Dz×Dy×Dx block of u_s (or v_s) * mask, applies a 3‐component
    Gaussian smoothing (K * (mask·u_s)) via a grouped Conv3d.
    """
    def __init__(self, xi: float, device: Optional[torch.device]=None):
        super().__init__()
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.xi = xi

        # --- build Gaussian kernel as a buffer ---
        radius = int(3 * xi)
        coords = torch.arange(-radius, radius+1, dtype=torch.float32, device=self.device)
        g1 = torch.exp(-coords**2 / (2 * xi*xi))
        g1 /= g1.sum()
        g3 = g1[:,None,None] * g1[None,:,None] * g1[None,None,:]          # (D,H,W)
        kernel = g3[None,None,:,:,:]                                      # (1,1,D,H,W)
        self.register_buffer('kernel', kernel)

        # --- build grouped conv3d and register its weight as buffer ---
        D, H, W = g3.shape
        conv = nn.Conv3d(
            in_channels=3, out_channels=3,
            kernel_size=(D,H,W),
            padding=(radius, radius, radius),
            groups=3,
            bias=False
        )
        # expand kernel → (3,1,D,H,W)
        w = kernel.expand(3,1,D,H,W).clone()
        conv.weight.data.copy_(w)
        conv.weight.requires_grad_(False)
        self.conv = conv.to(self.device)
        self._smooth_fn = _maybe_compile_function(
            self._smooth_impl,
            compile_kwargs={"mode": "reduce-overhead", "fullgraph": True},
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (1, 3, Dz, Dy, Dx)   (i.e. [ batch=1, channels=3, spatial... ])
        returns: (1,3,Dz,Dy,Dx)
        """
        return self.conv(x)

    def _smooth_impl(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def smooth(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return self._smooth_fn(x)
        except RuntimeError as exc:
            message = str(exc)
            if "Compiler: cl is not found" in message or "torch._inductor.exc" in message:
                warnings.warn(
                    "Falling back to eager vector-field smoothing because torch.compile "
                    "failed (likely missing required compiler).",
                    RuntimeWarning,
                )
                self._smooth_fn = self._smooth_impl
                return self._smooth_fn(x)
            raise
