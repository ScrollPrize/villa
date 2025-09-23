# smooth_vf.py
import torch
import torch.nn as nn
from typing import Optional

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (1, 3, Dz, Dy, Dx)   (i.e. [ batch=1, channels=3, spatial... ])
        returns: (1,3,Dz,Dy,Dx)
        """
        return self.conv(x)

    @torch.compile(fullgraph=True, mode='reduce-overhead')
    def smooth(self, x: torch.Tensor) -> torch.Tensor:
        # compiling this makes the conv into a fused kernel
        return self.forward(x)
