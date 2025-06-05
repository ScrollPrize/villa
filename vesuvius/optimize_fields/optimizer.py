# optimize_fields/optimizer.py

import torch
import geoopt
from geoopt import ManifoldParameter
from geoopt.optim import RiemannianAdam
from tqdm.auto import tqdm
from optimize_fields.utils import gradient, divergence, rotate_by_quaternion, matrix_to_quaternion
import wandb

# TODO: Fix this error: patch 0, iter 1] ⚠ post‐proj ‖q‖ not ≈1: min=nan, max=nan                                                                             | 0/2000 [00:00<?, ?it/s]
# [patch 0, iter 1] ⚠ NaN detected in q_manifold AFTER projection!

class PatchOptimizer:
    """
    Per‐patch optimizer for u,v,n with:
      - Smoothness loss term via gradient()
      - Compressibility loss term via divergence()
      - Masked data‐term (skip voxels where U, V, or N = zero)
      + Optional wandb logging every N iters (only for 1 out of every K patches)

    We first compute the loss at initialization. If that loss is
    essentially zero, we skip the entire inner loop and return identity,
    avoiding any chance of a zero‐gradient step → NaN. Otherwise, we run
    the RiemannianAdam loop as usual (optimizer.step() + retraction).
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
        decay_max: float = 0.5,
    ):
        self.device = device
        self.mu0, self.mu1, self.mu4 = mu0, mu1, mu4
        self.lr, self.iters = lr, iters
        self.decay_max = decay_max
        self.log_iter_every = log_iter_every
        self.log_patch_every = log_patch_every

        # Increments across all iterations of all patches so wandb steps don’t collide
        self.global_step = 0

        # Sphere manifold for quaternion retraction
        self.sphere = geoopt.Sphere()

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
            divergence(u).pow(2).sum() +
            divergence(v).pow(2).sum() +
            divergence(n).pow(2).sum()
        )

        total = data + smooth + div_pen
        return total, data.detach(), smooth.detach(), div_pen.detach()

    def __call__(self, U: torch.Tensor, V: torch.Tensor, N: torch.Tensor, chunk_idx: int) -> torch.Tensor:
        """
        Run RiemannianAdam on S³ for `self.iters` steps. Returns a (4, D, H, W)
        quaternion field that is guaranteed unit‐norm.

        1) Sanitize inputs.
        2) If patch is all zeros, return identity immediately.
        3) Create identity quaternion field, compute its total_loss. If total_loss < tol,
           return identity immediately (skip any step). This prevents any 0/0 or NaN.
        4) Otherwise, run the full RiemannianAdam loop (optimizer.step + project) from iter=1…iters.
        """

        eps = 1e-8
        # ── 1) Sanitize any NaN/Inf in inputs
        U = torch.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
        V = torch.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
        N = torch.nan_to_num(N, nan=0.0, posinf=0.0, neginf=0.0)

        D, H, W = U.shape[1], U.shape[2], U.shape[3]

        # ── 2) If the patch is entirely zero after sanitization, immediately return identity
        if (U.abs().sum() < 1e-12) and (V.abs().sum() < 1e-12) and (N.abs().sum() < 1e-12):
            D, H, W = U.shape[1], U.shape[2], U.shape[3]
            q_identity = torch.zeros((4, D, H, W), device=self.device, dtype=U.dtype)
            q_identity[0, ...] = 1.0
            return q_identity

        # ── 1) Build per‐voxel mask of (U,V,N) nonzero ────────────────────────────────────
        U_vec = U.permute(1, 2, 3, 0)  # (D,H,W,3)
        V_vec = V.permute(1, 2, 3, 0)
        N_vec = N.permute(1, 2, 3, 0)

        norm_U = U_vec.norm(dim=-1)   # (D,H,W)
        norm_V = V_vec.norm(dim=-1)
        norm_N = N_vec.norm(dim=-1)

        mask_uvn = (norm_U > eps) & (norm_V > eps) & (norm_N > eps)  # (D,H,W) bool

        # ── 2) Flatten to a “batch” of size D*H*W ───────────────────────────────────────
        num_flat = D * H * W
        mask_flat = mask_uvn.reshape(num_flat)                            # (num_flat,)

        U_flat = U_vec.reshape(num_flat, 3)  # (num_flat,3)
        V_flat = V_vec.reshape(num_flat, 3)
        N_flat = N_vec.reshape(num_flat, 3)

        # Normalize only on “valid” voxels; fill invalids with arbitrary orthonormal triad
        U_norm = U_flat / norm_U.reshape(-1, 1).clamp(min=eps)
        V_norm = V_flat / norm_V.reshape(-1, 1).clamp(min=eps)
        N_norm = N_flat / norm_N.reshape(-1, 1).clamp(min=eps)

        # For invalid voxels, overwrite with standard basis so SVD won’t blow up
        # (They’ll get overwritten to identity quat later.)
        invalid_idx = ~mask_flat
        if invalid_idx.any():
            U_norm[invalid_idx] = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            V_norm[invalid_idx] = torch.tensor([0.0, 1.0, 0.0], device=self.device)
            N_norm[invalid_idx] = torch.tensor([0.0, 0.0, 1.0], device=self.device)

        # ── 3) Stack columns into A ∈ (num_flat, 3, 3) ───────────────────────────────────
        #    A[i] = [ U_norm[i] | V_norm[i] | N_norm[i] ] as columns
        A_cols = torch.stack([U_norm, V_norm, N_norm], dim=-1)  # (num_flat, 3, 3)

        # ── 4) Batched SVD on A_cols ────────────────────────────────────────────────────
        #    A_cols = U_svd @ diag(S_vals) @ Vh, with U_svd, Vh ∈ (num_flat,3,3)
        U_svd, S_vals, Vh = torch.linalg.svd(A_cols, full_matrices=False)
        # Now form R0 = U_svd @ Vh
        R0 = torch.matmul(U_svd, Vh)  # (num_flat, 3, 3)

        # ── 5) Enforce det(R0) = +1 by flipping last column of U_svd where needed ───────
        dets = torch.linalg.det(R0)    # (num_flat,)
        neg_det = (dets < 0.0)
        if neg_det.any():
            # Flip U_svd[...,2] on those voxels, then recompute R0 there
            U_svd[neg_det, :, 2] *= -1.0
            R0[neg_det] = torch.matmul(U_svd[neg_det], Vh[neg_det])

        # At this point every R0[i] is a proper rotation (det = +1), but its 3 columns
        # are in “SVD order,” not necessarily with the third column best aligned to N.

        # ── 6) REORDER so that the third column is the one |dot(⋅, N_norm)|-max, and preserve right-handedness ─
        #    (a) Extract the three columns c0,c1,c2 from R0:
        c0 = R0[...,  :, 0]  # (num_flat, 3)
        c1 = R0[...,  :, 1]
        c2 = R0[...,  :, 2]

        #    (b) Compute dot‐products with N_norm:
        d0 = (c0 * N_norm).sum(dim=-1).abs()  # (num_flat,)
        d1 = (c1 * N_norm).sum(dim=-1).abs()
        d2 = (c2 * N_norm).sum(dim=-1).abs()

        #    (c) Pick best_i = argmax_j d_j  (which column best aligns (in abs) with N_norm)
        dots_abs = torch.stack([d0, d1, d2], dim=1)    # (num_flat, 3)
        best_i = torch.argmax(dots_abs, dim=1)         # (num_flat,) each ∈ {0,1,2}

        #    (d) Compute “sign” s[i] = +1 if that column’s dot > 0, else s[i] = –1
        #        so that we’ll flip c_best → s⋅c_best to make it point positively.
        dot0 = (c0 * N_norm).sum(dim=-1)  # (num_flat,)
        dot1 = (c1 * N_norm).sum(dim=-1)
        dot2 = (c2 * N_norm).sum(dim=-1)
        dots = torch.stack([dot0, dot1, dot2], dim=1)    # (num_flat,3)
        # Gather the signed‐dot corresponding to best_i:
        idx = torch.arange(num_flat, device=self.device)
        s = dots[idx, best_i].sign()                    # (num_flat,) in {+1,0,–1}
        # If exactly zero (rare), treat as +1:
        s[s == 0] = 1.0

        #    (e) For each best_i, define the “other two” indices in ascending order:
        #        If best_i=0 → leftovers = (1,2)
        #        If best_i=1 → leftovers = (0,2)
        #        If best_i=2 → leftovers = (0,1)
        leftovers = torch.tensor([[1, 2],
                                [0, 2],
                                [0, 1]], device=self.device)    # (3,2)
        i_a = leftovers[best_i, 0]  # (num_flat,)
        i_b = leftovers[best_i, 1]  # (num_flat,)

        #    (f) Determine parity(sign_perm) of the permutation (i_a, i_b, best_i) relative to (0,1,2):
        #        One can show: if best_i == 1 → sign_perm = –1; else sign_perm = +1.
        sign_perm = torch.ones_like(best_i, dtype=torch.float32, device=self.device)
        sign_perm[best_i == 1] = -1.0   # shape (num_flat,)

        #    (g) If s[i] != sign_perm[i], then we must swap the “two leftovers” to restore det=+1.
        swap_mask = (s != sign_perm) & mask_flat  # only valid voxels matter

        #    (h) Gather columns by advanced indexing:
        #        R0 has shape (num_flat, 3, 3).  We want, per‐voxel:
        #          new_n[i] =  s[i] * R0[i,:,best_i[i]]
        #          new_u[i], new_v[i] = R0[i,:,i_a[i]], R0[i,:,i_b[i]]  but possibly swapped if swap_mask[i]==True.

        # First build indices = [0,1,2,…, num_flat-1]
        indices = torch.arange(num_flat, device=self.device)

        # Pick out the best‐aligned column:
        c_best = R0[indices, :, best_i]        # (num_flat, 3)
        new_n = (s.unsqueeze(-1) * c_best)     # (num_flat, 3)

        # Pick out the two leftover columns (in ascending index order):
        c_ia = R0[indices, :, i_a]             # (num_flat, 3)
        c_ib = R0[indices, :, i_b]             # (num_flat, 3)

        # If swap_mask==False → new_u = c_ia, new_v = c_ib
        # If swap_mask==True  → new_u = c_ib, new_v = c_ia  (swap them)
        swap_mask3 = swap_mask.unsqueeze(-1)   # (num_flat, 1)
        new_u = torch.where(swap_mask3, c_ib, c_ia)  # (num_flat, 3)
        new_v = torch.where(swap_mask3, c_ia, c_ib)  # (num_flat, 3)

        #    (i) Now stack into a “reordered” rotation‐matrix new_R[i] = [new_u[i] | new_v[i] | new_n[i]]
        new_R = torch.stack([new_u, new_v, new_n], dim=-1)  # (num_flat, 3, 3)

        # ── 7) Convert each new_R[i] ∈ SO(3) → quaternion via matrix_to_quaternion ────────
        # Note: matrix_to_quaternion expects shape (N,3,3) → (N,4)
        q_flat = matrix_to_quaternion(new_R)   # (num_flat, 4)

        # ── 8) Wherever mask_flat is False, we override to identity quaternion ─────────────
        if invalid_idx.any():
            q_flat[invalid_idx] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=U.dtype)

        # ── 9) Reshape back → (D, H, W, 4) → permute → (4, D, H, W) ──────────────────────
        q_init_nhwc = q_flat.reshape(D, H, W, 4)              # (D, H, W, 4)
        q_init = q_init_nhwc.permute(3, 0, 1, 2).contiguous() # (4, D, H, W)

        # From here on, you can wrap q_init into a ManifoldParameter on the sphere:
        q_manifold = ManifoldParameter(q_init, manifold=self.sphere)

        # At this point ‖q_manifold‖ = 1 for all voxels.

        # Compute the loss at identity WITHOUT any backward or step:
        with torch.no_grad():
            total0, data0, smooth0, div0 = self._compute_losses(q_manifold, U, V, N, alpha=0.5)

        # If that total loss is essentially zero, no gradient can move us anywhere—
        # so we return identity and skip all optimization steps.
        if abs(total0.item()) < 1e-12:
            # (Optional) Log it once at step=0
            if (chunk_idx % self.log_patch_every) == 0:
                wandb.log({
                    "patch_idx": chunk_idx,
                    "iter_in_patch": 0,
                    "alpha": 0.5,
                    "loss/total": float(total0.item()),
                    "loss/data": float(data0.item()),
                    "loss/smooth": float(smooth0.item()),
                    "loss/div": float(div0.item()),
                    "lr": 0.0
                }, step=self.global_step)
                self.global_step += 1

            return q_manifold.detach()  # identity

        # ── 4) Otherwise, build the optimizer + scheduler and run iter=1…iters
        optimizer = RiemannianAdam([q_manifold], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.iters, eta_min=self.lr * 0.01
        )

        do_log_this_patch = (chunk_idx % self.log_patch_every == 0)
        pbar = tqdm(range(1, self.iters + 1), desc=f"RiAd patch {chunk_idx}", leave=False)

        for iter_i in pbar:
            optimizer.zero_grad()

            # ── Compute “alpha_i” that linearly grows from 0 → decay_max over [1..iters]
            # TODO, setting it fixed to 0.5 for the moment
            alpha_i = 0.5 #(self.decay_max * iter_i / self.iters)
            #if alpha_i > self.decay_max:
            #     alpha_i = self.decay_max

            # Compute loss at the current q_manifold
            total_loss, data_term, smooth_term, div_term = self._compute_losses(q_manifold, U, V, N, alpha_i)

            # If any component is NaN/Inf, bail immediately
            if not torch.isfinite(total_loss):
                print(f"[patch {chunk_idx}, iter {iter_i}] ⚠ total_loss is not finite: {total_loss}")
                break
            if not torch.isfinite(data_term):
                print(f"[patch {chunk_idx}, iter {iter_i}] ⚠ data_term is not finite: {data_term}")
                break
            if not torch.isfinite(smooth_term):
                print(f"[patch {chunk_idx}, iter {iter_i}] ⚠ smooth_term is not finite: {smooth_term}")
                break
            if not torch.isfinite(div_term):
                print(f"[patch {chunk_idx}, iter {iter_i}] ⚠ div_term is not finite: {div_term}")
                break

            total_loss.backward()

            # Clip gradient to prevent catastrophic jumps
            # torch.nn.utils.clip_grad_norm_([q_manifold], max_norm=0.1)

            # (Optional) Check quaternion norms before stepping
            with torch.no_grad():
                norms_pre = torch.norm(q_manifold.data, dim=0)
                mn_pre, mx_pre = norms_pre.min().item(), norms_pre.max().item()
                if not (0.9 < mn_pre < 1.1 and 0.9 < mx_pre < 1.1):
                    print(f"[patch {chunk_idx}, iter {iter_i}] ⚠ pre‐step ‖q‖ out of [0.9,1.1]: min={mn_pre:.3e}, max={mx_pre:.3e}")

            optimizer.step()

            # Explicit projection onto S³ (clamping tiny norms to ε)
            with torch.no_grad():
                q_nhwc = q_manifold.data.permute(1, 2, 3, 0).contiguous()  # (D,H,W,4)
                eps = 1e-8
                norm = q_nhwc.norm(dim=-1, keepdim=True).clamp_min(eps)
                q_proj = q_nhwc / norm

                num_clamped = (norm == eps).sum().item()
                if num_clamped > 0:
                    print(f"[patch {chunk_idx}, iter {iter_i}] ⚠ {num_clamped} quaternion(s) had norm < eps; clamped")

                q_manifold.data.copy_(q_proj.permute(3, 0, 1, 2))  # back to (4,D,H,W)

                # Post‐projection norm check
                norms_post = torch.norm(q_manifold.data, dim=0)
                mn_post, mx_post = norms_post.min().item(), norms_post.max().item()
                if not (0.9999 < mn_post < 1.0001 and 0.9999 < mx_post < 1.0001):
                    print(f"[patch {chunk_idx}, iter {iter_i}] ⚠ post‐proj ‖q‖ not ≈1: min={mn_post:.3e}, max={mx_post:.3e}")

                if torch.isnan(q_manifold.data).any():
                    print(f"[patch {chunk_idx}, iter {iter_i}] ⚠ NaN detected in q_manifold AFTER projection!")
                    break

            # Enforce q₀ ≥ 0 to avoid quaternion sign‐flips
            # with torch.no_grad():
            #    mask = q_manifold.data[0] < 0  # (D,H,W)
            #    if mask.any():
            #        negative_mask = mask.unsqueeze(0).expand_as(q_manifold.data)
            #        q_manifold.data[negative_mask] *= -1.0

            scheduler.step()

            # Log metrics to wandb (if desired)
            lr_now = optimizer.param_groups[0]['lr']
            pbar.set_postfix(
                total=f"{total_loss.item():.2e}",
                lr=f"{lr_now:.2e}"
            )
            if do_log_this_patch and ((iter_i % self.log_iter_every) == 0):
                wandb.log({
                    "patch_idx": chunk_idx,
                    "iter_in_patch": iter_i,
                    "alpha": alpha_i,
                    "loss/total": float(total_loss.item()),
                    "loss/data": float(data_term.item()),
                    "loss/smooth": float(smooth_term.item()),
                    "loss/div": float(div_term.item()),
                    "lr": float(lr_now)
                }, step=self.global_step)
                self.global_step += 1

        # After loop: final wandb log if we did any real step and total_loss was finite
        if do_log_this_patch and torch.isfinite(total_loss) and (self.global_step > 0):
            with torch.no_grad():
                final_total, final_data, final_smooth, final_div = self._compute_losses(
                    q_manifold, U, V, N, alpha=alpha_i
                )
            wandb.log({
                "patch_idx": chunk_idx,
                "iter_in_patch": self.iters,
                "alpha": alpha_i,
                "loss/total_final": float(final_total.item()),
                "loss/data_final": float(final_data.item()),
                "loss/smooth_final": float(final_smooth.item()),
                "loss/div_final": float(final_div.item())
            }, step=self.global_step)

        return q_manifold.detach()


# ── Standalone sanity check ────────────────────────────────────────────────────
if __name__ == "__main__":
    wandb.init(project="test-riemannian", mode="disabled")

    D, H, W = 8, 8, 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy inputs with all zeros (will return identity immediately)
    U = torch.zeros((3, D, H, W), device=device)
    V = torch.zeros_like(U)
    N = torch.zeros_like(U)

    patch_opt = PatchOptimizer(
        mu0=1e-3, mu1=1e-2, mu4=1e-2,
        lr=0.5, iters=50,
        device=device,
        log_iter_every=10,
        log_patch_every=1,
    )

    print("Launching on a patch of all zeros (should return identity immediately)...")
    q_out = patch_opt(U, V, N, chunk_idx=0)
    print("Done. q_out shape:", q_out.shape)
    norms = torch.norm(q_out, dim=0)
    print("Final min norm:", norms.min().item(), "max norm:", norms.max().item())
