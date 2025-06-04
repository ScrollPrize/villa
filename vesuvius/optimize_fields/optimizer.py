# optimize_fields/optimizer.py

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from optimize_fields.utils import gradient, divergence, rotate_by_quaternion
import wandb


class PatchOptimizer:
    """
    Per‐patch optimizer for u,v,n with:
      - smoothness loss term via gradient()
      - compressibility loss term via divergence()
      + optional wandb logging every N iters, but only for 1 out of every K patches.
    """

    def __init__(
        self,
        mu0: float,
        mu1: float,
        mu4: float,
        lr: float,
        iters: int,
        device: torch.device,
        log_iter_every: int = 10,
        log_patch_every: int = 10,
    ):
        self.device = device
        self.mu0, self.mu1, self.mu4 = mu0, mu1, mu4
        self.lr, self.iters = lr, iters

        # Logging controls:
        self.log_iter_every = log_iter_every
        self.log_patch_every = log_patch_every

        # This will increment across every single iteration of every chunk,
        # so that wandb steps don’t collide.
        self.global_step = 0

    @torch.compile(fullgraph=True, dynamic=True)
    def _step(self, q_raw, U, V, N):
        # 1) normalize quaternion
        norm = q_raw.norm(dim=0, keepdim=True).clamp_min(1e-8)
        q = q_raw / norm       # shape (4,D,H,W)

        u0 = torch.zeros_like(U)  # → (3, D, H, W)
        v0 = torch.zeros_like(U)
        n0 = torch.zeros_like(U)

        u0[0, ...] = 1.0   # e₁ everywhere
        v0[1, ...] = 1.0   # e₂ everywhere
        n0[2, ...] = 1.0   # e₃ everywhere

        # 2) rotate initial frame
        u = rotate_by_quaternion(q.permute(1,2,3,0), u0.permute(1,2,3,0))
        v = rotate_by_quaternion(q.permute(1,2,3,0), v0.permute(1,2,3,0))
        n = rotate_by_quaternion(q.permute(1,2,3,0), n0.permute(1,2,3,0))

        # put back to (3,D,H,W)
        u = u.permute(3,0,1,2)
        v = v.permute(3,0,1,2)
        n = n.permute(3,0,1,2)

        # 3) data fidelity
        term_u = ((u * U).sum(dim=0)).pow(2).sum()
        term_v = ((v * V).sum(dim=0)).pow(2).sum()
        term_n = ((n * N).sum(dim=0)).pow(2).sum()

        data = -self.mu0*(term_u + term_v + term_n)

        # 4) smoothness on rotated fields
        grad_u = gradient(u)
        grad_v = gradient(v)
        grad_n = gradient(n)
        smooth = self.mu1 * (
            grad_u.pow(2).sum() + grad_v.pow(2).sum() + grad_n.pow(2).sum()
        )

        # 5) divergence penalty
        div_pen = self.mu4 * (
            divergence(u).pow(2).sum()
            + divergence(v).pow(2).sum()
            + divergence(n).pow(2).sum()
        )

        total = data + smooth + div_pen
        return total, data.detach(), smooth.detach(), div_pen.detach()

    def __call__(self, U: torch.Tensor, V: torch.Tensor, N: torch.Tensor, chunk_idx: int) -> torch.Tensor:
        """
        Runs `self.iters` of AdamW on q_raw. If chunk_idx % self.log_patch_every == 0,
        then inside each iteration we also call wandb.log(...) every self.log_iter_every steps.
        Returns the optimized q_raw (4×D×H×W).
        """

        # Initialize raw quaternion to identity at every voxel
        q_raw = torch.zeros((4,) + U.shape[1:], device=self.device, dtype=U.dtype)
        q_raw[0].fill_(1.0)
        q_raw.requires_grad_(True)

        optimizer = AdamW([q_raw], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.iters, eta_min=self.lr * 0.01
        )

        pbar = tqdm(range(self.iters), desc=f"  optim patch {chunk_idx}", leave=False)

        # Should we log all inner‐loop stats for this patch?
        do_log_this_patch = (chunk_idx % self.log_patch_every == 0)

        for iter_i in pbar:
            optimizer.zero_grad()

            # _step now returns a tuple: (total_loss, data_term, smooth_term, div_term)
            total_loss, data_term, smooth_term, div_term = self._step(q_raw, U, V, N)
            total_loss.backward()

            # Optional: gradient clipping if you want:
            # torch.nn.utils.clip_grad_norm_([q_raw], 1.0)

            optimizer.step()
            scheduler.step()

            # Always update progress bar with the total loss & current lr
            lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(
                total=f"{total_loss.item():.2e}",
                lr=f"{lr:.2e}"
            )

            # ─── Only log to wandb if (1) this is one of the chosen patches, and (2) iter % log_iter_every == 0 ─────
            if do_log_this_patch and ((iter_i % self.log_iter_every) == 0):
                wandb.log({
                    "patch_idx": chunk_idx,
                    "iter_in_patch": iter_i,
                    "loss/total": total_loss.item(),
                    "loss/data": data_term.item(),
                    "loss/smooth": smooth_term.item(),
                    "loss/div": div_term.item(),
                    "lr": lr
                }, step=self.global_step)
            self.global_step += 1

        # ─── After finishing this patch, log the final total loss (if it’s a “logged” patch) ─────────────────────────
        if do_log_this_patch:
            # Recompute final losses one more time (no grads needed)
            with torch.no_grad():
                final_total, final_data, final_smooth, final_div = self._step(q_raw, U, V, N)
                wandb.log({
                    "patch_idx": chunk_idx,
                    "iter_in_patch": self.iters,
                    "loss/total_final": final_total.item(),
                    "loss/data_final": final_data.item(),
                    "loss/smooth_final": final_smooth.item(),
                    "loss/div_final": final_div.item()
                }, step=self.global_step)

        return q_raw.detach()
