# optimize_fields/runner.py

import argparse
import torch
import wandb                            # ⬅ NEW: wandb client
import matplotlib.pyplot as plt         # ⬅ NEW: for plotting
import numpy as np                      # ⬅ NEW: for array manipulation
import zarr                              # ⬅ NEW: to open the labels Zarr store
import yaml                              # ⬅ NEW: parse YAML config

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
                   help="Patch size cz,cy,cx")
    p.add_argument('--mu0', type=float, default=1e-3,
                   help="Penalty coefficient for the coupling term with the external fields")
    p.add_argument('--mu1', type=float, default=1e-2,
                   help="Penalty coefficient for the Frobenius norm of the Jacobian of u,v, and n")
    p.add_argument('--mu4', type=float, default=1e-2,
                   help="Penalty coefficient for the divergence of u,v, and n")
    p.add_argument('--decay_max', type=float, default=0.5,
                   help="Maximum blending weight for smoothness term (will linearly grow up to this).")
    p.add_argument('--lr',  type=float, default=1.0)
    p.add_argument('--iters', type=int, default=1000, 
                   help="Number of inner‐loop optimization steps per patch")
    p.add_argument('--ghost', type=int, default=1,
                   help="Padding voxels on each side for finite diffs")
    p.add_argument('--device', default='cuda')
    

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
    return p.parse_args()


def main():
    args = parse_args()

    # ── If a YAML config file was given, load it and override the corresponding args ─────────
    if args.config is not None:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        # For each key in the YAML that matches an argparse name, set it on args.
        for key, val in cfg.items():
            if hasattr(args, key):
                setattr(args, key, val)
            else:
                raise ValueError(f"Unknown key '{key}' in config file")
    device = torch.device(args.device)
    cz, cy, cx = map(int, args.chunk.split(','))

    # ─── 1) Initialize wandb ───────────────────────────────────────────────────
    wandb.init(
        project=args.wandb_project,
        config={
            "chunk_size": (cz, cy, cx),
            "mu0": args.mu0,
            "mu1": args.mu1,
            "mu4": args.mu4,
            "lr": args.lr,
            "decay_max": args.decay_max,
            "iters_per_patch": args.iters,
            "ghost": args.ghost,
        }
    )
    # You can override any config fields from CLI or let wandb capture them automatically
    config = wandb.config

    # 2) Open inputs
    store = open_zarr(args.input_store, 'r')
    src_U = store['U']
    src_V = store['V']
    if args.input_store_normal is None: # if surface normals are not give, take normal from fibers
        src_N = store['N']
    else:
        store_normal = open_zarr(args.input_store_normal, 'r')
        src_N = store_normal['N']

    # 3) Figure out volume shape for output (always 3×Z×Y×X)
    C, Z, Y, X = src_U.shape
    out_shape = (3, Z, Y, X)

    # 4) Create output store
    _, ds_u, ds_v, ds_n = create_output_store(
        args.output,
        volume_shape=out_shape,
        chunk_size=(3, cz, cy, cx)
    )

    # 5) Build dataset & optimizer
    ds = ChunkDataset(src_U, src_V, src_N,
                      chunk_size=(cz, cy, cx), ghost=args.ghost)

    optimizer = PatchOptimizer(
        mu0=args.mu0,
        mu1=args.mu1,
        mu4=args.mu4,
        lr=args.lr,
        iters=args.iters,
        device=device,
        log_iter_every=args.log_iter_every,
        log_patch_every=args.log_patch_every,
        decay_max=args.decay_max
    )
    # ── NEW: If user passed --labels_store, open it once and remember:
    if args.labels_store is not None:
        label_zarr = zarr.open(args.labels_store, mode='r')
        # We'll slice this in exactly the same (z0:z1, y0:y1, x0:x1) that each patch uses.
    else:
        label_zarr = None

    # Stash both label_zarr and viz params on the optimizer so we can access inside the loop:
    optimizer.label_zarr = label_zarr
    optimizer.skip = args.skip
    optimizer.viz_every = args.viz_every

    # 6) Loop over patches
    for patch_idx, ((z0, z1, y0, y1, x0, x1), (zp0, zp1, yp0, yp1, xp0, xp1),
                    U_np, V_np, N_np) in enumerate(tqdm(ds, total=len(ds), desc="patches")):

        # Skip empty blocks
        if (U_np == 0).all() and (V_np == 0).all() and (N_np == 0).all():
            ds_u[:, z0:z1, y0:y1, x0:x1] = 0
            ds_v[:, z0:z1, y0:y1, x0:x1] = 0
            ds_n[:, z0:z1, y0:y1, x0:x1] = 0
            continue

        # To torch
        U = torch.from_numpy(U_np).to(device)
        V = torch.from_numpy(V_np).to(device)
        N = torch.from_numpy(N_np).to(device)

        # Optimize patch → returns raw quaternion field (4×D×H×W)
        q_raw = optimizer(U, V, N, patch_idx)

        # Normalize to unit quaternions
        eps = 1e-8
        norm = q_raw.norm(dim=0, keepdim=True).clamp_min(eps)
        q = q_raw / norm               # (4, D, H, W)

        # Rotate initial fields
        q_nhwc = q.permute(1, 2, 3, 0)      # (D,H,W,4)
        # e₁,e₂,e₃ initial
        u0 = torch.zeros_like(U); v0 = torch.zeros_like(U); n0 = torch.zeros_like(U)
        u0[0].fill_(1.0); v0[1].fill_(1.0); n0[2].fill_(1.0)
        u0_nhwc = u0.permute(1, 2, 3, 0)    # (D,H,W,3)
        v0_nhwc = v0.permute(1, 2, 3, 0)
        n0_nhwc = n0.permute(1, 2, 3, 0)

        u_rot = rotate_by_quaternion(q_nhwc, u0_nhwc)  # → (D,H,W,3)
        v_rot = rotate_by_quaternion(q_nhwc, v0_nhwc)
        n_rot = rotate_by_quaternion(q_nhwc, n0_nhwc)

        # back to (3,D,H,W) and numpy
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


        if (optimizer.label_zarr is not None) and (patch_idx % optimizer.viz_every == 0):
            labels_chunk = optimizer.label_zarr[z0:z1, y0:y1, x0:x1]   # (dz, dy, dx)
            dz = z1 - z0
            mid = dz // 2
            bg = labels_chunk[mid, :, :]   # (dy, dx)

            mask = (bg > 100)

            # Subsample the 2D grid
            skip = optimizer.skip
            H, W = bg.shape
            Yg, Xg = np.mgrid[0:H, 0:W]

            Xq = Xg[::skip, ::skip]   # shape ((H//skip),(W//skip))
            Yq = Yg[::skip, ::skip]

            # Pull out your three vector‐fields on that slice
            # (Adjust indices [0],[1], etc. according to how rotate_by_quaternion packed them)
            slice_n = n_trim[:, mid, :, :]  # (3, dy, dx)
            slice_u = u_trim[:, mid, :, :]
            slice_v = v_trim[:, mid, :, :]

            nx = slice_n[1, ::skip, ::skip]
            ny = slice_n[0, ::skip, ::skip]

            ux = slice_u[0, ::skip, ::skip]
            uy = slice_u[1, ::skip, ::skip]

            vx = slice_v[0, ::skip, ::skip]
            vy = slice_v[1, ::skip, ::skip]

            mask_q = mask[::skip, ::skip]
            #num_arrows = np.count_nonzero(mask_q)
            #print(f"[Viz] patch {patch_idx}: drawing {num_arrows} arrows out of {mask_q.size}")

            Xf = Xq[mask_q]
            Yf = Yq[mask_q]
            Nfx = nx[mask_q]; Nfy = ny[mask_q]
            Ufx = ux[mask_q]; Ufy = uy[mask_q]
            Vfx = vx[mask_q]; Vfy = vy[mask_q]

            # Create the figure
            fig, ax = plt.subplots(figsize=(5, 5))

            # ▸ Use origin='lower' so that (0,0) in both imshow and quiver is bottom-left
            ax.imshow(bg, cmap="gray", origin="lower")
            ax.axis("off")

            # Draw each arrow‐set with contrasting colors and a smaller 'scale'
            ax.quiver(
                Xf, Yf, Nfx, Nfy,
                color="green",   # choose black on gray/white
                scale=20,        # smaller scale → longer arrows
                width=0.008,
                headlength=3,    # nonzero headlength so you see the arrowheads
                headaxislength=3
            )
            ax.quiver(
                Xf, Yf, Ufx, Ufy,
                color="red",
                scale=20,
                width=0.008,
                headlength=3, headaxislength=3
            )
            ax.quiver(
                Xf, Yf, Vfx, Vfy,
                color="blue",
                scale=20,
                width=0.008,
                headlength=3, headaxislength=3
            )

            # 8) Log to wandb with a **changing** step so you get a slider
            wandb.log(
                {"slice_viz": wandb.Image(fig)},
                step=optimizer.global_step
            )
            plt.close(fig)



    print("✅ Finished. Outputs at:", args.output)
    wandb.finish()


if __name__ == '__main__':
    main()
