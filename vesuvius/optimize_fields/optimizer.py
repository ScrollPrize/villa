# optimize_fields/optimizer.py

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from optimize_fields.utils import gradient, divergence, rotate_by_quaternion

#TODO: use NORMAL N in loss!

#TODO: optimize Ising problem for orientation!

class PatchOptimizer:
    """
    Per‐patch optimizer for u,v with:
      - smoothness loss term via gradient()
      - compressibility loss term via divergence()
    """

    def __init__(
        self,
        mu1: float,
        mu4: float,
        lr: float,
        iters: int,
        device: torch.device
    ):
        self.device  = device
        self.mu1, self.mu4 = mu1, mu4
        self.lr, self.iters = lr, iters

    @torch.compile(fullgraph=True, dynamic=True)
    def _step(self, q_raw, u0, v0, U, V, n):
        # 1) normalize quaternion
        norm = q_raw.norm(dim=0, keepdim=True).clamp_min(1e-8)
        q = q_raw / norm       # shape (4,D,H,W)

        # 2) rotate initial frame
        u = rotate_by_quaternion(q.permute(1,2,3,0),    # → (D,H,W,4)
                                 u0.permute(1,2,3,0) )  # → (D,H,W,3)
        v = rotate_by_quaternion(q.permute(1,2,3,0),
                                 v0.permute(1,2,3,0))
        # put back to (3,D,H,W)
        u = u.permute(3,0,1,2)
        v = v.permute(3,0,1,2)

        # 3) data fidelity
        data = - (u*U).sum() - (v*V).sum()

        data *= 10**(-3) # TODO: insert another weight here
        # 4) smoothness on rotated fields
        grad_u = gradient(u)
        grad_v = gradient(v)
        smooth = self.mu1 * (grad_u.pow(2).sum() + grad_v.pow(2).sum())

        # 5) divergence penalty
        div_pen = self.mu4 * (divergence(u).pow(2).sum() + divergence(v).pow(2).sum())

        return data + smooth + div_pen

    def __call__(self, u0, v0, U, V, n) -> torch.Tensor:
        # initialize raw quaternion to identity at every voxel
        # q_raw[0]=1, q_raw[1:]=0 → identity rotation
        q_raw = torch.zeros((4,)+U.shape[1:], device=self.device, dtype=U.dtype)
        q_raw[0].fill_(1.0)
        q_raw.requires_grad_(True)
        optimizer = AdamW([q_raw], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.iters, eta_min=self.lr*0.01
        )
        pbar = tqdm(range(self.iters), desc="  optim", leave=False)

        for _ in pbar:
            optimizer.zero_grad()
            loss = self._step(q_raw, u0, v0, U, V, n)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_([q_raw], 1.0) # TODO: put an intelligent value
            optimizer.step()
            scheduler.step()

            pbar.set_postfix(
                loss=f"{loss.item():.2e}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )

        return q_raw.detach()
