"""
SIGReg (Sketched Isotropic Gaussian Regularization) loss for LeJEPA pretraining.

This implements the Epps-Pulley characteristic function test to enforce
embeddings follow an isotropic Gaussian distribution N(0, I).

Reference: LeJEPA paper - https://arxiv.org/abs/2511.08544
"""

import torch
import torch.nn as nn
import torch.distributed as dist


class SIGRegLoss(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization loss.

    Enforces embeddings to follow N(0, I) distribution using the Epps-Pulley
    characteristic function test with random projections for efficiency.

    This implementation uses real cos/sin arithmetic and exploits the symmetry
    of the characteristic function by integrating only over t >= 0.

    Args:
        num_slices: Number of random projection directions (M in paper)
        knots: Number of evaluation points for characteristic function integration
        lambd: Weight for SIGReg loss vs invariance loss (default 0.02)
    """

    def __init__(
        self,
        num_slices: int = 256,
        knots: int = 17,
        lambd: float = 0.02,
    ):
        super().__init__()
        self.num_slices = num_slices
        self.lambd = lambd

        # Evaluation points [0, 3] - exploit CF symmetry (only need t >= 0)
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)

        # Trapezoidal integration weights (2*dt interior, dt endpoints)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt

        # Gaussian window exp(-t^2/2) - the theoretical CF of N(0,1)
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)  # Target characteristic function
        self.register_buffer("weights", weights * window)  # Pre-multiply for efficiency

    def _sigreg(self, x: torch.Tensor, global_step: int) -> torch.Tensor:
        """
        Compute SIGReg loss for a single view's embeddings.

        Uses random projections with seeded generator for DDP synchronization.

        Args:
            x: (N, K) tensor of embeddings where N is batch size, K is embedding dim
            global_step: int, used for DDP-synchronized random seed

        Returns:
            statistic: (M,) tensor of test statistics per slice direction
        """
        N, K = x.shape
        device = x.device
        dtype = x.dtype

        # Generate random projection matrix (DDP-synced via deterministic seed)
        g = torch.Generator(device=device)
        g.manual_seed(global_step)
        A = torch.randn((K, self.num_slices), generator=g, device=device, dtype=dtype)
        # Normalize columns to unit vectors
        A = A / A.norm(p=2, dim=0, keepdim=True)

        # Project embeddings: (N, K) @ (K, M) -> (N, M)
        proj = x @ A

        # Compute x*t for all evaluation points: (N, M, T)
        x_t = proj.unsqueeze(-1) * self.t.to(device=device, dtype=dtype)

        # Empirical characteristic function via cos/sin (real arithmetic, ~2x faster)
        # ECF(t) = E[exp(i*t*X)] = E[cos(t*X)] + i*E[sin(t*X)]
        cos_mean = x_t.cos().mean(dim=0)  # (M, T)
        sin_mean = x_t.sin().mean(dim=0)  # (M, T)

        # DDP all-reduce for multi-GPU synchronization
        if dist.is_initialized():
            dist.all_reduce(cos_mean, op=dist.ReduceOp.AVG)
            dist.all_reduce(sin_mean, op=dist.ReduceOp.AVG)
            world_size = dist.get_world_size()
        else:
            world_size = 1

        # Squared error between empirical and theoretical CF
        # For N(0,1): theoretical CF is exp(-t^2/2) (real part) with 0 imaginary part
        phi = self.phi.to(device=device, dtype=dtype)
        err = (cos_mean - phi).square() + sin_mean.square()

        # Weighted integration over t dimension
        # Scale by total batch size (N * world_size) as per paper
        weights = self.weights.to(device=device, dtype=dtype)
        statistic = (err @ weights) * (N * world_size)

        return statistic  # (M,)

    def forward(
        self,
        global_proj: torch.Tensor,
        all_proj: torch.Tensor,
        global_step: int,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute full LeJEPA loss: (1-lambda)*invariance + lambda*sigreg

        Args:
            global_proj: (V_g, B, K) global view projections
            all_proj: (V, B, K) all view projections (global + local)
            global_step: int, current training step for SIGReg seed

        Returns:
            loss: scalar loss tensor
            loss_dict: dict with individual loss components for logging
        """
        V, B, K = all_proj.shape
        V_g = global_proj.shape[0]

        # Invariance loss: MSE between each view and mean of GLOBAL projections
        # Reference paper: centers computed from global views only, all views match to it
        # centers: mean projection across global views (B, K)
        centers = global_proj.mean(dim=0)

        # Each view should match the global view mean
        invariance_loss = (all_proj - centers.unsqueeze(0)).square().mean()

        # SIGReg loss: average over all views
        sigreg_losses = []
        for v in range(V):
            T = self._sigreg(all_proj[v], global_step)
            sigreg_losses.append(T.mean())
        sigreg_loss = torch.stack(sigreg_losses).mean()

        # Combined loss: lambda * sigreg + (1 - lambda) * invariance
        # Following the original paper's formulation exactly
        loss = self.lambd * sigreg_loss + (1 - self.lambd) * invariance_loss

        loss_dict = {
            "loss": loss.item(),
            "invariance_loss": invariance_loss.item(),
            "sigreg_loss": sigreg_loss.item(),
        }

        return loss, loss_dict
