### Giorgio Angelotti - 2024

import torch
from torch.nn.functional import normalize
from tqdm import tqdm
from scipy.spatial import KDTree
import numpy as np

def to_device(tensor, device):
    """Move tensor to a specified device."""
    return tensor.to(device, non_blocking=True)

def points_in_triangles(pts, tri_pts):
    # pts B x 2
    # tri_pts B x tri x 3 x 2
    # triangles B x tri x 3
    v0 = tri_pts[:, :, 2, :] - tri_pts[:, :, 0, :]
    v1 = tri_pts[:, :, 1, :] - tri_pts[:, :, 0, :]
    v2 = pts.unsqueeze(1) - tri_pts[:, :, 0, :]

    dot00 = v0.pow(2).sum(dim=2)
    dot01 = (v0 * v1).sum(dim=2)
    dot11 = v1.pow(2).sum(dim=2)
    dot02 = (v2 * v0).sum(dim=2)
    dot12 = (v2 * v1).sum(dim=2)

    invDenom = 1 / (dot00 * dot11 - dot01.pow(2))
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom

    is_inside = (u >= 0) & (v >= 0) & ((u + v) <= 1 )

    bary_coords = torch.zeros((u.shape[0], 3), dtype=torch.float64, device=pts.device)
    triangle_indices = torch.where(is_inside.any(dim=1), is_inside.float().argmax(dim=1), torch.tensor(-1, device=pts.device, dtype=torch.int32))
    inside_mask = triangle_indices != -1

    u_vals = torch.gather(u[inside_mask], 1, triangle_indices[inside_mask].unsqueeze(1)).squeeze(1)
    v_vals = torch.gather(v[inside_mask], 1, triangle_indices[inside_mask].unsqueeze(1)).squeeze(1)
    w_vals = 1 - u_vals - v_vals
    bary_coords[inside_mask] = torch.stack([u_vals, v_vals, w_vals], dim=1)
    bary_coords = normalize(bary_coords, p=1, dim=1)
    return triangle_indices, bary_coords

def build_kdtree(vertices, triangles):
    centroids = vertices.view(triangles.size(0), 3, 2).mean(dim=1).cpu().numpy()  # Compute centroids and move to CPU for KDTree
    kdtree = KDTree(centroids, balanced_tree=True)
    return kdtree

def query_kdtree(kdtree, points, top_k=16):
    # Convert points to numpy and query KDTree
    _, indices = kdtree.query(points.cpu().numpy(), k=top_k, workers=-1)
    return torch.from_numpy(indices)  # Convert back to tensor and move to original device


def points_in_triangles_batched(ppm_path, shape, pts, vertices, triangles, vertices3d, normals3d, pts_batch_size=2048, tri_batch_size=64):
    # Compute centroids of the triangles and build KDTree
    kdtree = build_kdtree(vertices, triangles)
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    num_pts = pts.size(0)
    y_size, x_size = shape
    new_order = [2,1,0]
    
    ppm = np.memmap(ppm_path, dtype=np.float64, mode='w+', shape=(y_size, x_size, 6))

    for pts_batch_idx in tqdm(range((num_pts + pts_batch_size - 1) // pts_batch_size), desc="Batch"):
        
        pts_start_idx = pts_batch_idx * pts_batch_size
        pts_end_idx = min(pts_start_idx + pts_batch_size, num_pts)
        batch_pts = pts[pts_start_idx:pts_end_idx]

        final_triangle_indices = torch.full((min(pts_batch_size, pts_end_idx - pts_start_idx),), -1, dtype=torch.int64, device=device)
        # Query KDTree for each point in the batch to find the top 16 closest triangles
        closest_tri_indices = query_kdtree(kdtree, batch_pts, top_k=tri_batch_size)

        closest_tri_indices = to_device(closest_tri_indices, device)
        
        batch_vertices = vertices.view(triangles.size(0), 3, 2)[closest_tri_indices]
        triangle_indices, bary_coords = points_in_triangles(to_device(batch_pts, device), to_device(batch_vertices, device))
        valid_mask = (triangle_indices != -1).clone()
        valid_points = valid_mask.nonzero().squeeze()
        # For each valid point, select the correct triangle index from closest_tri_indices
        if valid_points.numel() > 0:  # Check if there are any valid points
            valid_tri_indices = triangle_indices[valid_mask]
            valid_closest_tri_indices = closest_tri_indices[valid_mask,valid_tri_indices]
            # Update final_triangle_indices for valid points
            final_triangle_indices[valid_mask] = valid_closest_tri_indices
            
            # Update final_bary_coords for valid points

            tri_v = vertices3d[triangles[final_triangle_indices,:]]
            tri_norm = normals3d[triangles[final_triangle_indices,:]]

            tri_v = tri_v[:,new_order,:]
            tri_norm = tri_norm[:,new_order,:]
            
            batch_pts = batch_pts.cpu().to(torch.int32).numpy()


            coords = torch.einsum('ijk,ij->ik', tri_v, bary_coords).squeeze().numpy()
            norms = normalize(torch.einsum('ijk,ij->ik', tri_norm, bary_coords).squeeze(),dim=1).numpy()

            ppm[batch_pts[:,0], batch_pts[:,1], :3] = coords
            ppm[batch_pts[:,0], batch_pts[:,1], 3:] = norms
    
    print('Final flush')
    ppm.flush()

    print(f"PPM saved at {ppm_path}", end="\n")