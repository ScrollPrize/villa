# optimize_fields/runner.py

import argparse
import torch
import wandb
import matplotlib.pyplot as plt
import numpy as np
import zarr
import yaml

from tqdm.auto import tqdm
from optimize_fields.io_utils import open_zarr, create_output_store
from optimize_fields.utils import rotate_by_quaternion
from optimize_fields.dataset import ChunkDataset
from optimize_fields.optimizer import PatchOptimizer


def parse_args():
    p = argparse.ArgumentParser(
        description="Optimize (u,v,n) fields chunk-wise and write to Zarr"
    )
    p.add_argument('--config', type=str, required=True,
                   help="Path to a YAML config file containing all other parameters.")
    p.add_argument('--store', dest='input_store',
                   help="Zarr group containing datasets ['U','V','N'] obtained from fibers")
    p.add_argument('--store-normal', dest='input_store_normal',
                   help="Zarr group containing datasets ['U','V','N'] obtained from surfaces")
    p.add_argument('--output',
                   help="Output Zarr directory for u,v,n")
    p.add_argument('--chunk', default="384,384,384",
                   help="Patch size cz,cy,cz")
    p.add_argument('--mu0', type=float, default=1e-3,
                   help="Penalty coefficient for the coupling term with the external fields")
    p.add_argument('--mu1', type=float, default=1e-2,
                   help="Penalty coefficient for the Frobenius norm of the Jacobian of u,v, and n")
    p.add_argument('--mu4', type=float, default=1e-2,
                   help="Penalty coefficient for the divergence of u,v, and n")
    p.add_argument('--mu5', type=float, default=1e-2,
                   help="Penalty coefficient for the curl of curl of u,v, and n")
    p.add_argument('--decay_max', type=float, default=0.5,
                   help="Maximum blending weight for smoothness term (will linearly grow up to this).")
    p.add_argument('--lr', type=float, default=1.0,
                   help="Learning rate for the inner optimizer")
    p.add_argument('--iters', type=int, default=1000,
                   help="Number of inner‐loop optimization steps per patch")
    p.add_argument('--ghost', type=int, default=1,
                   help="Padding voxels on each side for finite differences")
    p.add_argument('--device', default='cuda',
                   help="Torch device (e.g., 'cpu' or 'cuda')")

    p.add_argument('--labels_store', type=str, required=False,
                   help="Path to Zarr of integer labels (so we can plot the same chunk).")
    p.add_argument('--skip', type=int, default=10,
                   help="Subsampling stride for quiver arrows (in XY).")
    p.add_argument('--viz_every', type=int, default=10,
                   help="Log a figure every K patches (if labels_store is set).")

    p.add_argument('--log_iter_every', type=int, default=10,
                   help="Log wandb metrics every N inner‐loop iterations (only for selected patches)")
    p.add_argument('--log_patch_every', type=int, default=10,
                   help="Only launch per‐iteration wandb.log for every Kth patch.")
    p.add_argument('--wandb_project', type=str, default="optimize_fields",
                   help="wandb project name")

    p.add_argument('--optimizer_type', type=str, default="adam",
                   choices=["radam", "adam", "sgd"],
                   help="Which inner‐loop optimizer to use: radam (geoopt), adam (vanilla + reproject), or sgd (vanilla + reproject).")
    p.add_argument('--grad_clip', type=float, default=0.0,
                   help="Maximum gradient norm to clip the quaternion during optimization (0 = no clipping).")
    p.add_argument('--init_type', type=str, default="svd",
                   choices=["svd","random","identity"],
                   help="How to initialize the quaternion field inside each patch: "
                        "'svd' = batched‐SVD (current behavior), "
                        "'random' = per‐voxel random unit‐quaternion, "
                        "'identity' = [1,0,0,0] everywhere.")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load YAML config and override any CLI args
    if args.config is not None:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        for key, val in cfg.items():
            if hasattr(args, key):
                setattr(args, key, val)
            else:
                raise ValueError(f"Unknown key '{key}' in config file")

    device = torch.device(args.device)
    cz, cy, cx = map(int, args.chunk.split(','))

    # ── Initialize wandb ─────────────────────────────────────────────────────────────
    wandb.init(
        project=args.wandb_project,
        config={
            "chunk_size": (cz, cy, cx),
            "mu0": args.mu0,
            "mu1": args.mu1,
            "mu4": args.mu4,
            "mu5": args.mu5,
            "lr": args.lr,
            "decay_max": args.decay_max,
            "iters_per_patch": args.iters,
            "ghost": args.ghost,
            "optimizer_type": args.optimizer_type,
            "grad_clip": args.grad_clip,
            "init_type": args.init_type
        }
    )
    config = wandb.config

    # ── Open inputs ───────────────────────────────────────────────────────────────────
    store = open_zarr(args.input_store, 'r')
    src_U = store['U']
    src_V = store['V']
    if args.input_store_normal is None:
        src_N = store['N']
    else:
        store_normal = open_zarr(args.input_store_normal, 'r')
        src_N = store_normal['N']

    # ── Output shape and create output store ──────────────────────────────────────────
    C, Z, Y, X = src_U.shape
    out_shape = (3, Z, Y, X)

    _, ds_u, ds_v, ds_n = create_output_store(
        args.output,
        volume_shape=out_shape,
        chunk_size=(3, cz, cy, cx)
    )

    # ── Build dataset & optimizer ────────────────────────────────────────────────────
    ds = ChunkDataset(src_U, src_V, src_N,
                      chunk_size=(cz, cy, cx), ghost=args.ghost)

    optimizer = PatchOptimizer(
        mu0=args.mu0,
        mu1=args.mu1,
        mu4=args.mu4,
        mu5=args.mu5,
        lr=args.lr,
        iters=args.iters,
        device=device,
        optimizer_type=args.optimizer_type,
        init_type=args.init_type,
        log_iter_every=args.log_iter_every,
        log_patch_every=args.log_patch_every,
        decay_max=args.decay_max
    )

    # If user passed --labels_store, open it once:
    if args.labels_store is not None:
        label_zarr = zarr.open(args.labels_store, mode='r')
    else:
        label_zarr = None

    # Stash label_zarr and viz parameters on the optimizer object
    optimizer.label_zarr = label_zarr
    optimizer.skip = args.skip
    optimizer.viz_every = args.viz_every

    # ── Loop over patches ─────────────────────────────────────────────────────────────
    for patch_idx, ((z0, z1, y0, y1, x0, x1), (zp0, zp1, yp0, yp1, xp0, xp1),
                    U_np, V_np, N_np) in enumerate(tqdm(ds, total=len(ds), desc="patches")):

        # Skip empty blocks
        if (U_np == 0).all() and (V_np == 0).all() and (N_np == 0).all():
            ds_u[:, z0:z1, y0:y1, x0:x1] = 0
            ds_v[:, z0:z1, y0:y1, x0:x1] = 0
            ds_n[:, z0:z1, y0:y1, x0:x1] = 0
            continue

        # Move to torch Tensors
        U = torch.from_numpy(U_np).to(device)
        V = torch.from_numpy(V_np).to(device)
        N = torch.from_numpy(N_np).to(device)

        # Optimize this patch → returns raw quaternion field (4×D×H×W)
        q_raw = optimizer(U, V, N, patch_idx)

        # Normalize to unit quaternions (safety, but q_raw should already be unit‐norm)
        eps = 1e-8
        norm = q_raw.norm(dim=0, keepdim=True).clamp_min(eps)
        q = q_raw / norm  # (4, D, H, W)

        # Rotate the canonical basis vectors (e₁,e₂,e₃) by q
        q_nhwc = q.permute(1, 2, 3, 0)  # (D, H, W, 4)

        # Build e₁, e₂, e₃ in (D, H, W, 3) form
        u0 = torch.zeros_like(U)
        v0 = torch.zeros_like(U)
        n0 = torch.zeros_like(U)
        u0[0].fill_(1.0)
        v0[1].fill_(1.0)
        n0[2].fill_(1.0)
        u0_nhwc = u0.permute(1, 2, 3, 0)  # (D, H, W, 3)
        v0_nhwc = v0.permute(1, 2, 3, 0)
        n0_nhwc = n0.permute(1, 2, 3, 0)

        u_rot = rotate_by_quaternion(q_nhwc, u0_nhwc)  # → (D, H, W, 3)
        v_rot = rotate_by_quaternion(q_nhwc, v0_nhwc)
        n_rot = rotate_by_quaternion(q_nhwc, n0_nhwc)

        # Back to (3, D, H, W) and convert to numpy
        u_opt = u_rot.permute(3, 0, 1, 2).cpu().numpy()  # (3, D, H, W)
        v_opt = v_rot.permute(3, 0, 1, 2).cpu().numpy()  # (3, D, H, W)
        n_opt = n_rot.permute(3, 0, 1, 2).cpu().numpy()  # (3, D, H, W)

        # Trim off the “ghost” padding so we only keep the central region
        dz0, dz1 = z0 - zp0, z1 - zp0
        dy0, dy1 = y0 - yp0, y1 - yp0
        dx0, dx1 = x0 - xp0, x1 - xp0

        u_trim = u_opt[:, dz0:dz1, dy0:dy1, dx0:dx1]
        v_trim = v_opt[:, dz0:dz1, dy0:dy1, dx0:dx1]
        n_trim = n_opt[:, dz0:dz1, dy0:dy1, dx0:dx1]

        # Write back only the central (z0:z1, y0:y1, x0:x1) region
        ds_u[:, z0:z1, y0:y1, x0:x1] = u_trim
        ds_v[:, z0:z1, y0:y1, x0:x1] = v_trim
        ds_n[:, z0:z1, y0:y1, x0:x1] = n_trim

        # ── Optional visualization for every Kth patch (if labels_store provided) ─────
        if (optimizer.label_zarr is not None) and (patch_idx % optimizer.viz_every == 0):
            labels_chunk = optimizer.label_zarr[z0:z1, y0:y1, x0:x1]  # (dz, dy, dx)
            dz = z1 - z0
            mid = dz // 2
            bg = labels_chunk[mid, :, :]  # (dy, dx)

            mask = (bg > 100)

            # Subsample the 2D grid
            skip = optimizer.skip
            H_img, W_img = bg.shape
            Yg, Xg = np.mgrid[0:H_img, 0:W_img]

            Xq = Xg[::skip, ::skip]  # ((H//skip), (W//skip))
            Yq = Yg[::skip, ::skip]

            # Pull out each vector‐field on that slice
            slice_n = n_trim[:, mid, :, :]  # (3, dy, dx)
            slice_u = u_trim[:, mid, :, :]
            slice_v = v_trim[:, mid, :, :]

            nx = slice_n[2, ::skip, ::skip]
            ny = slice_n[1, ::skip, ::skip]

            ux = slice_u[2, ::skip, ::skip]
            uy = slice_u[1, ::skip, ::skip]

            vx = slice_v[2, ::skip, ::skip]
            vy = slice_v[1, ::skip, ::skip]

            mask_q = mask[::skip, ::skip]

            Xf = Xq[mask_q]
            Yf = Yq[mask_q]
            Nfx = nx[mask_q]
            Nfy = ny[mask_q]
            Ufx = ux[mask_q]
            Ufy = uy[mask_q]
            Vfx = vx[mask_q]
            Vfy = vy[mask_q]

            # Create the figure
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(bg, cmap="gray", origin="lower")
            ax.axis("off")

            # Draw each arrow‐set with contrasting colors and a smaller 'scale'
            ax.quiver(
                Xf, Yf, Nfx, Nfy,
                color="green",
                scale=20,
                width=0.008,
                headlength=3,
                headaxislength=3
            )
            ax.quiver(
                Xf, Yf, Ufx, Ufy,
                color="red",
                scale=20,
                width=0.008,
                headlength=3,
                headaxislength=3
            )
            ax.quiver(
                Xf, Yf, Vfx, Vfy,
                color="blue",
                scale=20,
                width=0.008,
                headlength=3,
                headaxislength=3
            )

            # Log to wandb with a changing step so you get a slider
            wandb.log(
                {"slice_viz": wandb.Image(fig)},
                step=optimizer.global_step
            )
            plt.close(fig)

    print("✅ Finished. Outputs at:", args.output)
    wandb.finish()


if __name__ == '__main__':
    main()
