# optimize_fields/optimizer.py

import torch
import geoopt
import numpy as np
from geoopt import ManifoldParameter
from geoopt.optim import RiemannianAdam
from tqdm.auto import tqdm
from optimize_fields.utils import gradient, rotate_by_quaternion, matrix_to_quaternion
import wandb

# ── 1) Batched SVD initialization ─────────────────────
#TODO: compile this
def _init_svd(
    U: torch.Tensor,  # (3, D, H, W)
    V: torch.Tensor,  # (3, D, H, W)
    N: torch.Tensor,  # (3, D, H, W)
    eps: float        # e.g. 1e-8
) -> torch.Tensor:   # returns (4, D, H, W)
    """
    Perform per‐voxel SVD‐based orthonormalization of [U, V, N]:
    1) Normalize each U_vec, V_vec, N_vec, and replace zeros with canonical axes.
    2) Compute A_cols = [U_norm, V_norm, N_norm] as shape (num_flat, 3, 3).
    3) Do a batched SVD and correct det<0 via a pure‐tensor mask.
    4) Reorder axes so that the third column aligns best with N_norm.
    5) Convert each 3×3 frame to a quaternion.
    """
    # Shapes
    D, H, W = U.shape[1], U.shape[2], U.shape[3]
    num_flat = D * H * W

    # 1) Flatten each field to (num_flat, 3)
    U_vec = U.permute(1, 2, 3, 0).reshape(num_flat, 3)  # (num_flat, 3)
    V_vec = V.permute(1, 2, 3, 0).reshape(num_flat, 3)
    N_vec = N.permute(1, 2, 3, 0).reshape(num_flat, 3)

    # Compute norms
    norm_U = U_vec.norm(dim=-1)  # (num_flat,)
    norm_V = V_vec.norm(dim=-1)
    norm_N = N_vec.norm(dim=-1)

    # Mask for “valid” vectors (all three > eps)
    mask_flat = (norm_U > eps) & (norm_V > eps) & (norm_N > eps)  # (num_flat,)

    # 1a) Safe divide by norm (clamped at eps)
    U_norm_pre = U_vec / norm_U.clamp(min=eps).unsqueeze(-1)  # (num_flat, 3)
    V_norm_pre = V_vec / norm_V.clamp(min=eps).unsqueeze(-1)
    N_norm_pre = N_vec / norm_N.clamp(min=eps).unsqueeze(-1)

    # 1b) For invalid entries, choose canonical axes:
    #      invalid_mask = ~mask_flat
    #      U_invalid = [1,0,0], V_invalid = [0,1,0], N_invalid = [0,0,1]
    invalid_mask = ~mask_flat  # (num_flat,)
    invalid_mask3 = invalid_mask.unsqueeze(-1).expand(-1, 3)  # (num_flat, 3)

    U_canon = torch.tensor([1.0, 0.0, 0.0], device=U.device, dtype=U.dtype).view(1, 3).expand(num_flat, 3)
    V_canon = torch.tensor([0.0, 1.0, 0.0], device=U.device, dtype=U.dtype).view(1, 3).expand(num_flat, 3)
    N_canon = torch.tensor([0.0, 0.0, 1.0], device=U.device, dtype=U.dtype).view(1, 3).expand(num_flat, 3)

    U_norm = torch.where(invalid_mask3, U_canon, U_norm_pre)  # (num_flat,3)
    V_norm = torch.where(invalid_mask3, V_canon, V_norm_pre)
    N_norm = torch.where(invalid_mask3, N_canon, N_norm_pre)

    # 2) Build A_cols: (num_flat, 3, 3)
    A_cols = torch.stack([U_norm, V_norm, N_norm], dim=-1).reshape(num_flat, 3, 3)

    # 3) Batched SVD
    U_svd, S_vals, Vh = torch.linalg.svd(A_cols, full_matrices=False)  # U_svd:(num_flat,3,3), Vh:(num_flat,3,3)
    R0 = torch.matmul(U_svd, Vh)  # (num_flat, 3, 3)

    # 3a) Fix any det(R0) < 0 via pure‐tensor masking
    dets = torch.linalg.det(R0)         # (num_flat,)
    neg_det = (dets < 0.0).unsqueeze(-1).unsqueeze(-1)  # (num_flat,1,1)

    # If det<0, flip the third column of U_svd. Build a “flipped” version of U_svd:
    flip_col = U_svd.clone()  # clone to avoid in-place
    flip_col[:, :, 2] = -flip_col[:, :, 2]  # flip third column
    U_svd_fixed = torch.where(neg_det, flip_col, U_svd)  # (num_flat,3,3)

    R0 = torch.matmul(U_svd_fixed, Vh)  # (num_flat, 3, 3)  # now det(R0) ≥ 0

    # 4) Reorder so that column-2 best aligns with N_norm.
    c0 = R0[..., :, 0]  # (num_flat, 3)
    c1 = R0[..., :, 1]
    c2 = R0[..., :, 2]

    # Dot products with N_norm
    dot0 = (c0 * N_norm).sum(dim=-1)  # (num_flat,)
    dot1 = (c1 * N_norm).sum(dim=-1)
    dot2 = (c2 * N_norm).sum(dim=-1)

    abs0 = dot0.abs()
    abs1 = dot1.abs()
    abs2 = dot2.abs()
    dots_abs = torch.stack([abs0, abs1, abs2], dim=1)  # (num_flat, 3)
    best_i = torch.argmax(dots_abs, dim=1)             # (num_flat,)

    # s = sign of (c_best ⋅ N_norm), with zeros replaced by 1
    idx = torch.arange(num_flat, device=U.device)
    gather_dot = torch.stack([dot0, dot1, dot2], dim=1)  # (num_flat,3)
    s = gather_dot[idx, best_i].sign()                   # (num_flat,)
    s = torch.where(s == 0.0, torch.ones_like(s), s)      # replace 0 by 1

    # leftover indices for column 2
    leftovers = torch.tensor([[1, 2], [0, 2], [0, 1]], device=U.device)  # (3,2)
    i_a = leftovers[best_i, 0]  # (num_flat,)
    i_b = leftovers[best_i, 1]  # (num_flat,)

    # determine where to swap based on sign mismatch
    sign_perm = torch.ones_like(best_i, dtype=torch.float32, device=U.device)
    sign_perm = torch.where(best_i == 1, -sign_perm, sign_perm)  # if best_i==1, sign_perm=-1
    swap_mask = (s != sign_perm)    # (num_flat,)

    # For each flat index, pick c_ia, c_ib, c_best
    c_ia = R0[idx, :, i_a]   # (num_flat, 3)
    c_ib = R0[idx, :, i_b]   # (num_flat, 3)
    c_best = R0[idx, :, best_i]  # (num_flat, 3)

    new_n = s.unsqueeze(-1) * c_best  # (num_flat, 3)
    swap_mask3 = swap_mask.unsqueeze(-1)  # (num_flat, 1)

    new_u = torch.where(swap_mask3, c_ib, c_ia)  # (num_flat, 3)
    new_v = torch.where(swap_mask3, c_ia, c_ib)  # (num_flat, 3)

    # 5) Build final R matrix per voxel: (num_flat, 3, 3)
    new_R = torch.stack([new_u, new_v, new_n], dim=-1)  # (num_flat, 3, 3)

    # Convert each 3×3 to quaternion (num_flat, 4)
    q_flat = matrix_to_quaternion(new_R)  # (num_flat, 4)
    assert torch.allclose(q_flat.norm(dim=-1), torch.ones_like(dets), atol=1e-6)

    # For any invalid voxels, force quaternion = [1,0,0,0]
    q_canon = torch.tensor([1.0, 0.0, 0.0, 0.0], device=U.device, dtype=U.dtype).view(1, 4).expand(num_flat, 4)
    invalid_mask2 = invalid_mask.unsqueeze(-1).expand(-1, 4)  # (num_flat, 4)
    q_flat = torch.where(invalid_mask2, q_canon, q_flat)      # (num_flat, 4)

    # 6) Reshape back to (D, H, W, 4) → permute → (4, D, H, W)
    q_init_nhwc = q_flat.reshape(D, H, W, 4)               # (D, H, W, 4)
    q_init = q_init_nhwc.permute(3, 0, 1, 2).contiguous()  # (4, D, H, W)
    return q_init

# ── 2) Identity initialization  ───────────────────────────────────────
#TODO: compile this
def _init_identity(
    D: int, H: int, W: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:  # returns (4, D, H, W)
    """
    Build a constant quaternion field q=[1,0,0,0] at every voxel.
    """
    q_init = torch.zeros((4, D, H, W), device=device, dtype=dtype)
    q_init[0, ...] = 1.0
    return q_init


# ── 3) Random initialization  ────────────────────────────────────────
#TODO: COMPILE THIS
def _init_random(
    D: int, H: int, W: int, device: torch.device, dtype: torch.dtype, eps: float
) -> torch.Tensor:  # returns (4, D, H, W)
    """
    Sample a random unit quaternion at each voxel via Gaussian + normalize.
    """
    num_flat = D * H * W
    q_rand_flat = torch.randn((num_flat, 4), device=device, dtype=dtype)
    norm_rand = q_rand_flat.norm(dim=1, keepdim=True).clamp_min(eps)  # (num_flat,1)
    q_rand_flat = q_rand_flat / norm_rand  # (num_flat, 4)
    q_init_nhwc = q_rand_flat.view(D, H, W, 4)             # (D, H, W, 4)
    q_init = q_init_nhwc.permute(3, 0, 1, 2).contiguous()   # (4, D, H, W)
    return q_init

class PatchOptimizer:
    """
    Per‐patch optimizer for u,v,n with option to pick one of:
      - 'adam'   → vanilla torch.optim.Adam + manual re‐projection onto S³
      - 'sgd'    → vanilla torch.optim.SGD  + manual re‐projection onto S³
      - 'radam'  → RiemannianAdam on S³ (using geoopt.ManifoldParameter)


    We compute a loss combining:
      - Data‐term: alignment of (u,v,n) to external fields U,V,N (masked).
      - Smoothness: Frobenius norm of ∇u, ∇v, ∇n.
      - Divergence penalty: (div u)² + (div v)² + (div n)².

    The quaternion field q lives on S³ (unit‐norm per‐voxel). Each iteration:
      1) Compute losses via _compute_losses().
      2) Backpropagate.
      3) Step the chosen optimizer.
      4) Reproject onto S³ if using vanilla Adam/SGD.
      5) Enforce q₀ ≥ 0 (to avoid sign flips).
      6) Log to wandb if requested.
    """

    def __init__(
        self,
        mu0: float,
        mu1: float,
        mu4: float,
        mu5: float,
        lr: float,
        iters: int,
        device: torch.device,
        optimizer_type: str = "adam",
        init_type: str = "svd",
        log_iter_every: int = 10,
        log_patch_every: int = 10,
        decay_max: float = 0.5,
        grad_clip: float = 0.0
    ):
        self.device = device
        self.mu0, self.mu1, self.mu4, self.mu5 = float(mu0), float(mu1), float(mu4), float(mu5)
        self.lr, self.iters = lr, iters
        self.decay_max = decay_max
        self.log_iter_every = log_iter_every
        self.log_patch_every = log_patch_every
        self.grad_clip = grad_clip
        optimizer_type = optimizer_type.lower()
        if optimizer_type not in ("radam", "adam", "sgd"):
            raise ValueError(f"optimizer_type must be one of 'radam', 'adam', or 'sgd'; got {optimizer_type}")
        self.optimizer_type = optimizer_type

        # validate init_type
        init_type = init_type.lower()
        if init_type not in ("svd", "random", "identity"):
            raise ValueError(f"init_type must be one of 'svd','random','identity'; got {init_type}")
        self.init_type = init_type

        # Sphere manifold for quaternion retraction (used by RiemannianAdam)
        self.sphere = geoopt.Sphere()

        # Global step counter (for wandb logging across patches)
        self.global_step = 0

    #@torch.compile(fullgraph=True, mode="max-autotune", dynamic=True)
    def _compute_losses(self, q: torch.Tensor, U: torch.Tensor, V: torch.Tensor, N: torch.Tensor, alpha: float):
        """
        Given a normalized quaternion field q of shape (4, D, H, W),
        compute (total, data, smooth, div). The data‐term is masked so that
        if U(x)=0, we skip ⟨u(x),U(x)⟩²; similarly for V and N.
        """
        # 1) Permute q from (4, D, H, W) → (D, H, W, 4)
        q_nhwc = q.permute(1, 2, 3, 0)  # (D, H, W, 4)

        # 2) Build canonical bases e₁,e₂,e₃ in (D, H, W, 3) form
        D, H, W = U.shape[1:]
        u0 = torch.zeros((D, H, W, 3), device=self.device, dtype=U.dtype)
        v0 = torch.zeros_like(u0)
        n0 = torch.zeros_like(u0)
        u0[..., 0] = 1.0  # (1,0,0)
        v0[..., 1] = 1.0  # (0,1,0)
        n0[..., 2] = 1.0  # (0,0,1)

        # 3) Rotate by q:
        u_rot = rotate_by_quaternion(q_nhwc, u0)  # → (D, H, W, 3)
        v_rot = rotate_by_quaternion(q_nhwc, v0)  # → (D, H, W, 3)
        n_rot = rotate_by_quaternion(q_nhwc, n0)  # → (D, H, W, 3)

        # 4) Permute back to (3, D, H, W)
        u = u_rot.permute(3, 0, 1, 2)  # (3, D, H, W)
        v = v_rot.permute(3, 0, 1, 2)
        n = n_rot.permute(3, 0, 1, 2)

        # ── Masked data‐term ───────────────────────────────────────────────────────────
        mask_u = (U.norm(dim=0) > 0.0)  # (D, H, W)
        mask_v = (V.norm(dim=0) > 0.0)
        mask_n = (N.norm(dim=0) > 0.0)

        dot_u = (u * U).sum(dim=0)  # shape (D, H, W)
        dot_v = (v * V).sum(dim=0)
        dot_n = (n * N).sum(dim=0)

        term_u = (dot_u.pow(2) * mask_u).sum()
        term_v = (dot_v.pow(2) * mask_v).sum()
        term_n = (dot_n.pow(2) * mask_n).sum()

        data = - (1.0 - alpha) * self.mu0 * (term_u + term_v + term_n)

        # ── Smoothness term (unmasked) ───────────────────────────────────────────────
        grad_u = gradient(u)
        grad_v = gradient(v)
        grad_n = gradient(n)
        smooth = alpha * self.mu1 * (
            grad_u.pow(2).sum() + grad_v.pow(2).sum() + grad_n.pow(2).sum()
        )

        # ── Divergence penalty (unmasked) ────────────────────────────────────────────
        div_pen = alpha * self.mu4 * (
            (grad_u[0,0] + grad_u[1,1] + grad_u[2,2]).pow(2).sum() +
            (grad_v[0,0] + grad_v[1,1] + grad_v[2,2]).pow(2).sum() +
            (grad_n[0,0] + grad_n[1,1] + grad_n[2,2]).pow(2).sum()
        )

        # ── Curl‐of‐curl penalty ───────────────────────────────────────────────────────
        # We will compute curl_u = ∇×u, then curlcurl_u = ∇×(∇×u), and likewise for v,n.
        if self.mu5 != 0:
            # 1) first curl:
            curl_u = torch.zeros_like(u)  # shape (3,D,H,W)
            curl_v = torch.zeros_like(v)
            curl_n = torch.zeros_like(n)

            # (curl u)_x = ∂_y u_z - ∂_z u_y
            curl_u[2] = grad_u[0, 1] - grad_u[1, 0]
            curl_v[2] = grad_v[0, 1] - grad_v[1, 0]
            curl_n[2] = grad_n[0, 1] - grad_n[1, 0]

            # (curl u)_y = ∂_z u_x - ∂_x u_z
            curl_u[1] = -grad_u[0, 2] + grad_u[2, 0]
            curl_v[1] = -grad_v[0, 2] + grad_v[2, 0]
            curl_n[1] = -grad_n[0, 2] + grad_n[2, 0]

            # (curl u)_z = ∂_x u_y - ∂_y u_x
            curl_u[0] = grad_u[1, 2] - grad_u[2, 1]
            curl_v[0] = grad_v[1, 2] - grad_v[2, 1]
            curl_n[0] = grad_n[1, 2] - grad_n[2, 1]

            # 2) second curl (i.e. curl of curl):
            g2_u = gradient(curl_u)  # shape (3,3,D,H,W)
            g2_v = gradient(curl_v)
            g2_n = gradient(curl_n)

            curlcurl_u = torch.zeros_like(curl_u)  # shape (3,D,H,W)
            curlcurl_v = torch.zeros_like(curl_v)
            curlcurl_n = torch.zeros_like(curl_n)

            # (curlcurl u)_x = ∂_y (curl u)_z - ∂_z (curl u)_y
            curlcurl_u[2] = g2_u[0, 1] - g2_u[1, 0]
            curlcurl_v[2] = g2_v[0, 1] - g2_v[1, 0]
            curlcurl_n[2] = g2_n[0, 1] - g2_n[1, 0]

            # (curlcurl u)_y = ∂_z (curl u)_x - ∂_x (curl u)_z
            curlcurl_u[1] = -g2_u[0, 2] + g2_u[2, 0]
            curlcurl_v[1] = -g2_v[0, 2] + g2_v[2, 0]
            curlcurl_n[1] = -g2_n[0, 2] + g2_n[2, 0]

            # (curlcurl u)_z = ∂_x (curl u)_y - ∂_y (curl u)_x
            curlcurl_u[0] = g2_u[1, 2] - g2_u[2, 1]
            curlcurl_v[0] = g2_v[1, 2] - g2_v[2, 1]
            curlcurl_n[0] = g2_n[1, 2] - g2_n[2, 1]

            # 3) squared‐L2 norms:
            curlcurl_term = alpha * self.mu5 * (curlcurl_u.pow(2).sum() \
                        + curlcurl_v.pow(2).sum() \
                        + curlcurl_n.pow(2).sum())

            total = data + smooth + div_pen + curlcurl_term
        
        else:
            curlcurl_term = torch.from_numpy(np.array([0])).to("cuda")
            total = data + smooth + div_pen

        return total, data.detach(), smooth.detach(), div_pen.detach(), curlcurl_term.detach()

    def __call__(self, U: torch.Tensor, V: torch.Tensor, N: torch.Tensor, chunk_idx: int) -> torch.Tensor:
        """
        Run the chosen optimizer on S³ for `self.iters` steps. Returns a (4, D, H, W)
        quaternion field that is guaranteed unit‐norm.

        Steps:
          1) Sanitize inputs.
          2) If patch is all zeros, return identity immediately.
          3) Build initial quaternion field q_init via batched SVD + reordering.
          4) Wrap q_init as ManifoldParameter (for RiemannianAdam) or nn.Parameter (for Adam/SGD).
          5) Compute loss at identity; if ≈0, skip optimization and return identity.
          6) Otherwise, run `self.iters` iterations of chosen optimizer + retraction or reprojection.
        """

        eps = 1e-8
        eps_flip = 1e-6
        # ── 1) Sanitize any NaN/Inf in inputs
        U = torch.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
        V = torch.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
        N = torch.nan_to_num(N, nan=0.0, posinf=0.0, neginf=0.0)

        D, H, W = U.shape[1], U.shape[2], U.shape[3]

        # ── 2) If the patch is entirely zero after sanitization, return identity
        if (U.abs().sum() < 1e-12) and (V.abs().sum() < 1e-12) and (N.abs().sum() < 1e-12):
            q_identity = torch.zeros((4, D, H, W), device=self.device, dtype=U.dtype)
            q_identity[0, ...] = 1.0
            return q_identity

        if self.init_type == "svd":
            q_init = _init_svd(U, V, N, eps)  # returns (4, D, H, W)
        elif self.init_type == "identity":
            q_init = _init_identity(D, H, W, self.device, U.dtype)
        elif self.init_type == "random":
            q_init = _init_random(D, H, W, self.device, U.dtype, eps)
        else:
            raise ValueError(
                f"Unknown init_type '{self.init_type}' (must be 'svd','random','identity')."
            )

        # 4) Wrap q_init depending on optimizer_type
        if self.optimizer_type == "radam":
            q_manifold = ManifoldParameter(q_init.to(self.device), manifold=self.sphere)
            params = [q_manifold]
        else:
            q_manifold = torch.nn.Parameter(q_init.to(self.device))
            params = [q_manifold]

        # 5) Compute loss at identity WITHOUT backward or step
        with torch.no_grad():
            total0, data0, smooth0, div0, curlcurl0 = self._compute_losses(q_manifold, U, V, N, alpha=0.5)

        if abs(total0.item()) < 1e-12:
            # Log once at step=0 if needed
            if (chunk_idx % self.log_patch_every) == 0:
                wandb.log({
                    "patch_idx": chunk_idx,
                    "iter_in_patch": 0,
                    "alpha": 0.5,
                    "loss/total": float(total0.item()),
                    "loss/data": float(data0.item()),
                    "loss/smooth": float(smooth0.item()),
                    "loss/div": float(div0.item()),
                    "loss/curlcurl": float(curlcurl0.item()),
                    "lr": 0.0
                }, step=self.global_step)
                self.global_step += 1
            return q_manifold.detach()  # identity

        # 6) Build the optimizer + scheduler
        if self.optimizer_type == "radam":
            optimizer = RiemannianAdam(params, lr=self.lr)
        elif self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(params, lr=self.lr)
        else:  # "sgd"
            optimizer = torch.optim.SGD(params, lr=self.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.iters, eta_min=self.lr * 0.01
        )

        do_log_this_patch = (chunk_idx % self.log_patch_every == 0)
        pbar = tqdm(range(1, self.iters + 1), desc=f"{self.optimizer_type.upper()} patch {chunk_idx}", leave=False)

        for iter_i in pbar:
            optimizer.zero_grad()

            # alpha_i can grow linearly up to decay_max, but here we keep it fixed at 0.5
            alpha_i = 0.5
            
            total_loss, data_term, smooth_term, div_term, curlcurl_term = self._compute_losses(q_manifold, U, V, N, alpha_i)

            if not torch.isfinite(total_loss):
                print(f"[patch {chunk_idx}, iter {iter_i}] ⚠ total_loss is not finite: {total_loss}")
                break

            total_loss.backward()

            #(Optional) For vanilla Adam/SGD, you may clip gradients on q_manifold:
            #if self.optimizer_type in ("adam", "sgd"):
            if self.grad_clip and (self.grad_clip > 0.0):
                torch.nn.utils.clip_grad_norm_([q_manifold], max_norm=self.grad_clip)

            optimizer.step()

            # Reproject onto S³ (apparently RADAM not working very well, need to do)
            with torch.no_grad():
                if self.optimizer_type in ("adam", "sgd"):
                        q_nhwc = q_manifold.data.permute(1, 2, 3, 0).contiguous()  # (D, H, W, 4)
                        norm_q = q_nhwc.norm(dim=-1, keepdim=True).clamp_min(eps)
                        q_proj = q_nhwc / norm_q
                        q_manifold.data.copy_(q_proj.permute(3, 0, 1, 2))

                mask_sign = q_manifold.data[0] < - eps_flip  # (D, H, W)
                if mask_sign.any():
                    neg_mask = mask_sign.unsqueeze(0).expand_as(q_manifold.data)
                    q_manifold.data[neg_mask] *= -1.0

            scheduler.step()

            lr_now = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(total=f"{total_loss.item():.2e}", lr=f"{lr_now:.2e}")

            if do_log_this_patch and ((iter_i % self.log_iter_every) == 0):
                wandb.log({
                    "patch_idx": chunk_idx,
                    "iter_in_patch": iter_i,
                    "alpha": alpha_i,
                    "loss/total": float(total_loss.item()),
                    "loss/data": float(data_term.item()),
                    "loss/smooth": float(smooth_term.item()),
                    "loss/div": float(div_term.item()),
                    "loss/curlcurl": float(curlcurl_term.item()),
                    "lr": float(lr_now)
                }, step=self.global_step)
                self.global_step += 1

        # Final log after finishing all iterations (if we did any real steps)
        if do_log_this_patch and torch.isfinite(total_loss) and (self.global_step > 0):
            with torch.no_grad():
                final_total, final_data, final_smooth, final_div, final_curlcurl = self._compute_losses(
                    q_manifold, U, V, N, alpha=alpha_i
                )
            wandb.log({
                "patch_idx": chunk_idx,
                "iter_in_patch": self.iters,
                "alpha": alpha_i,
                "loss/total_final": float(final_total.item()),
                "loss/data_final": float(final_data.item()),
                "loss/smooth_final": float(final_smooth.item()),
                "loss/div_final": float(final_div.item()),
                "loss/curlcurl_final": float(final_curlcurl.item()),
            }, step=self.global_step)

        return q_manifold.detach()

if hasattr(torch, "compile"):
    PatchOptimizer._compute_losses = torch.compile(
        PatchOptimizer._compute_losses,
        fullgraph=True,
        mode="max-autotune",
        dynamic=True,
    )


# ── Standalone sanity check with an “intelligent” UVN ───────────────────────────────────
if __name__ == "__main__":
    """
    Instead of all‐zeros, we build a simple synthetic UVN field:
    1) Define a constant “ground‐truth” rotation q_true across the volume
       (e.g., 45° rotation about the z‐axis).
    2) Rotate the canonical basis vectors (1,0,0), (0,1,0), (0,0,1) by q_true
       to form U, V, N fields.
    3) Feed U, V, N into PatchOptimizer and see if it recovers (or approximates)
       the same quaternion field.
    """

    wandb.init(project="test-riemannian", mode="disabled")

    # 1) Volume size
    D, H, W = 8, 8, 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    theta = torch.tensor(45.0 * torch.pi / 180.0, device=device)
    q0 = torch.cos(theta / 2)        # scalar part
    qz = torch.sin(theta / 2)        # “z” in the 3‐vector
    q_true = torch.tensor([q0, qz, 0.0, 0.0],
                        device=device,
                        dtype=torch.float32)
    # Expand to a full field of shape (D, H, W, 4)
    q_true_field = q_true.view(1, 1, 1, 4).expand(D, H, W, 4).contiguous()  # (D, H, W, 4)

    # 3) Rotate canonical bases by q_true_field to get U_rot, V_rot, N_rot
    #    First, build canonical basis in (D, H, W, 3) form
    u0 = torch.zeros((D, H, W, 3), device=device)
    v0 = torch.zeros((D, H, W, 3), device=device)
    n0 = torch.zeros((D, H, W, 3), device=device)
    u0[..., 0] = 1.0  # (1, 0, 0)
    v0[..., 1] = 1.0  # (0, 1, 0)
    n0[..., 2] = 1.0  # (0, 0, 1)

    # Use rotate_by_quaternion to form U_rot, V_rot, N_rot in shape (D, H, W, 3)
    U_nhwc = rotate_by_quaternion(q_true_field, u0)
    V_nhwc = rotate_by_quaternion(q_true_field, v0)
    N_nhwc = rotate_by_quaternion(q_true_field, n0)

    
    # Permute to (3, D, H, W) so that PatchOptimizer expects them in that format
    U = U_nhwc.permute(3, 0, 1, 2).contiguous()  # (3, D, H, W)
    V = V_nhwc.permute(3, 0, 1, 2).contiguous()
    N = N_nhwc.permute(3, 0, 1, 2).contiguous()

    # 4) Initialize PatchOptimizer with a few iterations to see if it recovers q_true
    patch_opt = PatchOptimizer(
        mu0=1e-3,
        mu1=1e-2,
        mu4=1e-2,
        mu5=1e-3,
        lr=0.1,
        iters=10000,
        device=device,
        optimizer_type="adam",
        init_type="svd",
        log_iter_every=50,
        log_patch_every=1,
        decay_max=0.5,
        grad_clip=0.1,
    )

    # Run optimizer on this single “patch” (no chunk indexing needed)
    print("Launching optimizer on synthetic UVN fields...")
    q_out = patch_opt(U, V, N, chunk_idx=0)  # (4, D, H, W)

    # Check how close q_out is to q_true_field
    # Compute per‐voxel dot product between q_out and q_true
    q_out_nhwc = q_out.permute(1, 2, 3, 0).contiguous()  # (D, H, W, 4)

    # Normalize both (for safety)
    eps = 1e-12
    q_out_norm = q_out_nhwc.norm(dim=-1, keepdim=True).clamp_min(eps)
    q_out_unit = q_out_nhwc / q_out_norm

    q_true_norm = q_true_field.norm(dim=-1, keepdim=True).clamp_min(eps)
    q_true_unit = q_true_field / q_true_norm

    # Dot product: shape (D, H, W)
    dot = (q_out_unit * q_true_unit).sum(dim=-1).abs()  # absolute to handle sign ambiguity
    mean_dot = dot.mean().item()
    print(f"Average |⟨q_out, q_true⟩| over all voxels: {mean_dot:.4f} (should be ~1.0 if recovered)")

    # Also check minimum and maximum dot
    min_dot = dot.min().item()
    max_dot = dot.max().item()
    print(f"Min |dot|: {min_dot:.4f}, Max |dot|: {max_dot:.4f}")

    # Finally, print norms to verify unit‐length
    norms = torch.norm(q_out, dim=0)
    print("Final min norm in q_out:", norms.min().item(), "max norm:", norms.max().item())

    print("Done.")
    wandb.finish()
