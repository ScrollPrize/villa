# optimize_fields/optimizer.py

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from optimize_fields.utils import gradient, divergence, rotate_by_quaternion



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
    def _step(self, q_raw, U, V, N):
        # 1) normalize quaternion
        norm = q_raw.norm(dim=0, keepdim=True).clamp_min(1e-8)
        q = q_raw / norm       # shape (4,D,H,W)


        u0 = torch.zeros_like(U)  # → (3, D, H, W)
        v0 = torch.zeros_like(U)
        n0 = torch.zeros_like(U)

        # u0 = e₁ everywhere:
        u0[0, ...] = 1.0   
        # v0 = e₂ everywhere:
        v0[1, ...] = 1.0   
        # n0 = e₃ everywhere:
        n0[2, ...] = 1.0   
        
        # 2) rotate initial frame
        u = rotate_by_quaternion(q.permute(1,2,3,0),    # → (D,H,W,4)
                                 u0.permute(1,2,3,0) )  # → (D,H,W,3)
        v = rotate_by_quaternion(q.permute(1,2,3,0),
                                 v0.permute(1,2,3,0))
        n = rotate_by_quaternion(q.permute(1,2,3,0),
                                 n0.permute(1,2,3,0))
        
        # put back to (3,D,H,W)
        u = u.permute(3,0,1,2)
        v = v.permute(3,0,1,2)
        n = n.permute(3,0,1,2)

        # 3) data fidelity
        term_u = ( (u * U).sum(dim=0) ).pow(2).sum()
        term_v = ( (v * V).sum(dim=0) ).pow(2).sum()
        term_n = ( (n * N).sum(dim=0) ).pow(2).sum()
        
        data = -(term_u + term_v + term_n)

        data *= 10**(-3) # TODO: insert another weight here or find a good normalization strategy
        # 4) smoothness on rotated fields
        grad_u = gradient(u)
        grad_v = gradient(v)
        grad_n = gradient(n)
        smooth = self.mu1 * (grad_u.pow(2).sum() + grad_v.pow(2).sum() + grad_n.pow(2).sum())

        # 5) divergence penalty
        div_pen = self.mu4 * (divergence(u).pow(2).sum() + divergence(v).pow(2).sum() + divergence(n).pow(2).sum())

        return data + smooth + div_pen

    def __call__(self, U, V, N) -> torch.Tensor:
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
            loss = self._step(q_raw, U, V, N)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_([q_raw], 1.0) # TODO: put an intelligent value
            optimizer.step()
            scheduler.step()

            pbar.set_postfix(
                loss=f"{loss.item():.2e}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )

        return q_raw.detach()
