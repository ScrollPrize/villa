#optimize_fields / runner.py
import argparse
import torch
from tqdm.auto import tqdm

from optimize_fields.io_utils import open_zarr, create_output_store
from optimize_fields.utils import rotate_by_quaternion
from optimize_fields.dataset import ChunkDataset
from optimize_fields.optimizer import PatchOptimizer

def parse_args():
    p = argparse.ArgumentParser(
        description="Optimize (u,v) fields chunk-wise and write to Zarr"
    )
    p.add_argument('--u_init', dest='input_u_init', required=True,
                   help="Zarr of initial u field (3×Z×Y×X)")
    p.add_argument('--v_init', dest='input_v_init', required=True,
                   help="Zarr of initial v field (3×Z×Y×X)")
    p.add_argument('--store', dest='input_store', required=True,
                  help="Zarr group containing datasets ['U','V','n']")
    p.add_argument('--output', required=True,
                   help="Output Zarr directory for u,v")
    p.add_argument('--chunk', default="384,384,384",
                   help="Patch size cz,cy,cx")
    p.add_argument('--mu1', type=float, default=1e-2)
    p.add_argument('--mu4', type=float, default=1e-2)
    p.add_argument('--lr',  type=float, default=1e-1)
    p.add_argument('--iters', type=int, default=100) # TODO: callback for convergence?
    p.add_argument('--ghost', type=int, default=1,
                   help="Padding voxels on each side for finite diffs")
    p.add_argument('--device', default='cuda')
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    cz, cy, cx = map(int, args.chunk.split(','))

    # Open inputs
    src_u0 = open_zarr(args.input_u_init, 'r')['U']
    src_v0 = open_zarr(args.input_v_init, 'r')['V']
    store  = open_zarr(args.input_store, 'r')
    src_U  = store['U']
    src_V  = store['V']
    src_n  = store['N']
    # Figure out volume shape for output (always 3×Z×Y×X)
    # If src_u is 9×Z×Y×X, take Z,Y,X = src_u.shape[1:]
    C, Z, Y, X = src_u0.shape
    out_shape = (3, Z, Y, X)

    # Create output store
    _, ds_u, ds_v = create_output_store(
        args.output,
        volume_shape=out_shape,
        chunk_size=(3, cz, cy, cx)
    )

    # Build dataset & optimizer
    ds = ChunkDataset(src_u0, src_v0, src_U, src_V, src_n,
                       chunk_size=(cz,cy,cx), ghost=args.ghost)
    optimizer = PatchOptimizer(
        args.mu1, args.mu4,
        args.lr, args.iters, device
    )

    # Loop over patches
    for (z0,z1,y0,y1,x0,x1), (zp0,zp1,yp0,yp1,xp0,xp1), \
            u0_np, v0_np, U_np, V_np, n_np \
            in tqdm(ds, total=len(ds), desc="patches"):
        # Skip empty
        if (U_np == 0).all() and (V_np == 0).all():
            ds_u[:, z0:z1, y0:y1, x0:x1] = 0
            ds_v[:, z0:z1, y0:y1, x0:x1] = 0
            continue

        # To torch
        u0 = torch.from_numpy(u0_np).to(device)
        v0 = torch.from_numpy(v0_np).to(device)
        U  = torch.from_numpy(U_np).to(device)
        V  = torch.from_numpy(V_np).to(device)
        n  = torch.from_numpy(n_np).to(device)

        # Optimize patch → returns raw quaternion field (4×D×H×W)
        q_raw = optimizer(u0, v0, U, V, n)

        # Normalize to unit quaternions
        eps  = 1e-8
        norm = q_raw.norm(dim=0, keepdim=True).clamp_min(eps)
        q    = q_raw / norm               # (4, D, H, W)

        # Rotate initial fields
        # bring into (...,4) and (...,3) shape for cross‐prod version
        q_nhwc  = q.permute(1,2,3,0)      # (D,H,W,4)
        u0_nhwc = u0.permute(1,2,3,0)     # (D,H,W,3)
        v0_nhwc = v0.permute(1,2,3,0)     # (D,H,W,3)

        u_rot = rotate_by_quaternion(q_nhwc, u0_nhwc)
        v_rot = rotate_by_quaternion(q_nhwc, v0_nhwc)

        # back to (3,D,H,W) and numpy
        u_opt = u_rot.permute(3,0,1,2).cpu().numpy()
        v_opt = v_rot.permute(3,0,1,2).cpu().numpy()


        # now **trim** off the ghost halo:
        dz0, dz1 = z0 - zp0, z1 - zp0
        dy0, dy1 = y0 - yp0, y1 - yp0
        dx0, dx1 = x0 - xp0, x1 - xp0

        u_trim = u_opt[:, dz0:dz1, dy0:dy1, dx0:dx1]
        v_trim = v_opt[:, dz0:dz1, dy0:dy1, dx0:dx1]

        # Write back only the *central* region
        ds_u[:, z0:z1, y0:y1, x0:x1] = u_trim
        ds_v[:, z0:z1, y0:y1, x0:x1] = v_trim

    print("✅ Finished. Outputs at:", args.output)

if __name__ == '__main__':
    main()
