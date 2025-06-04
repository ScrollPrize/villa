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
        description="Optimize (u,v,n) fields chunk-wise and write to Zarr"
    )
    p.add_argument('--store', dest='input_store', required=True,
                  help="Zarr group containing datasets ['U','V','N'] obtained from fibers")
    p.add_argument('--store-normal', dest='input_store_normal', required=False,
                  help="Zarr group containing datasets ['U','V','N'] obtained from surfaces")
    p.add_argument('--output', required=True,
                   help="Output Zarr directory for u,v,n")
    p.add_argument('--chunk', default="384,384,384",
                   help="Patch size cz,cy,cx")
    p.add_argument('--mu1', type=float, default=1e-2,
                   help="Penalty coefficient for the Frobenious norm of the Jacobian of u,v, and n")
    p.add_argument('--mu4', type=float, default=1e-2,
                   help="Penalty coefficient for the divergence of u,v, and n")
    p.add_argument('--lr',  type=float, default=1.0)
    p.add_argument('--iters', type=int, default=1000) # TODO: callback for convergence?
    p.add_argument('--ghost', type=int, default=1,
                   help="Padding voxels on each side for finite diffs")
    p.add_argument('--device', default='cuda')
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    cz, cy, cx = map(int, args.chunk.split(','))

    # Open inputs

    store  = open_zarr(args.input_store, 'r')
    src_U  = store['U']
    src_V  = store['V']
    if args.input_store_normal is None: #if surface directions are not given, take normal from fibers
        src_N  = store['N']
    else:
        store_normal  = open_zarr(args.input_store_normal, 'r')
        src_N  = store_normal['N']
    
    # Figure out volume shape for output (always 3×Z×Y×X)
    C, Z, Y, X = src_U.shape
    out_shape = (3, Z, Y, X)

    # Create output store
    _, ds_u, ds_v, ds_n = create_output_store(
        args.output,
        volume_shape=out_shape,
        chunk_size=(3, cz, cy, cx)
    )

    # Build dataset & optimizer
    ds = ChunkDataset(src_U, src_V, src_N,
                       chunk_size=(cz,cy,cx), ghost=args.ghost)
    optimizer = PatchOptimizer(
        args.mu1, args.mu4,
        args.lr, args.iters, device
    )
    

    # Loop over patches
    for (z0,z1,y0,y1,x0,x1), (zp0,zp1,yp0,yp1,xp0,xp1), \
            U_np, V_np, N_np \
            in tqdm(ds, total=len(ds), desc="patches"):
        # Skip empty
        if (U_np == 0).all() and (V_np == 0).all()  and (N_np == 0).all():
            ds_u[:, z0:z1, y0:y1, x0:x1] = 0
            ds_v[:, z0:z1, y0:y1, x0:x1] = 0
            ds_n[:, z0:z1, y0:y1, x0:x1] = 0
            continue

        # To torch
        U  = torch.from_numpy(U_np).to(device)
        V  = torch.from_numpy(V_np).to(device)
        N  = torch.from_numpy(N_np).to(device)


        u0 = torch.zeros_like(U)  # (3, D, H, W)
        v0 = torch.zeros_like(U)
        n0 = torch.zeros_like(U)

        # Set the appropriate channel = 1 at every voxel
        u0[0].fill_(1.0)  # e₁ everywhere
        v0[1].fill_(1.0)  # e₂ everywhere
        n0[2].fill_(1.0)  # e₃ everywhere
        
        
        # Optimize patch → returns raw quaternion field (4×D×H×W)
        q_raw = optimizer( U, V, N)

        # Normalize to unit quaternions
        eps  = 1e-8
        norm = q_raw.norm(dim=0, keepdim=True).clamp_min(eps)
        q    = q_raw / norm               # (4, D, H, W)

        # Rotate initial fields
        # bring into (...,4) and (...,3) shape for cross‐prod version
        q_nhwc  = q.permute(1,2,3,0)      # (D,H,W,4)
        
        u0_nhwc = u0.permute(1, 2, 3, 0)  # (D, H, W, 3)
        v0_nhwc = v0.permute(1, 2, 3, 0)  # (D, H, W, 3)
        n0_nhwc = n0.permute(1, 2, 3, 0)  # (D, H, W, 3)

        u_rot = rotate_by_quaternion(q_nhwc, u0_nhwc)  # → (D, H, W, 3)
        v_rot = rotate_by_quaternion(q_nhwc, v0_nhwc)
        n_rot = rotate_by_quaternion(q_nhwc, n0_nhwc)

        # back to (3,D,H,W) and numpy
        u_opt = u_rot.permute(3, 0, 1, 2).cpu().numpy()  # (3, D, H, W)
        v_opt = v_rot.permute(3, 0, 1, 2).cpu().numpy()  # (3, D, H, W)
        n_opt = n_rot.permute(3, 0, 1, 2).cpu().numpy()  # (3, D, H, W)

        #
        # Trim off the “ghost” padding so we only keep the central region
        #
        dz0, dz1 = z0 - zp0, z1 - zp0
        dy0, dy1 = y0 - yp0, y1 - yp0
        dx0, dx1 = x0 - xp0, x1 - xp0

        u_trim = u_opt[:, dz0:dz1, dy0:dy1, dx0:dx1]
        v_trim = v_opt[:, dz0:dz1, dy0:dy1, dx0:dx1]
        n_trim = n_opt[:, dz0:dz1, dy0:dy1, dx0:dx1]

        #
        #  Write back only the central (z0:z1, y0:y1, x0:x1) region
        #
        ds_u[:, z0:z1, y0:y1, x0:x1] = u_trim
        ds_v[:, z0:z1, y0:y1, x0:x1] = v_trim
        ds_n[:, z0:z1, y0:y1, x0:x1] = n_trim

    print("✅ Finished. Outputs at:", args.output)

if __name__ == '__main__':
    main()
