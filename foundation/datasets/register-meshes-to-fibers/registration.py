# Multi-Mesh Registration with Skeleton Constraints and Inter-Mesh Intersection Penalty
# Giorgio Angelotti - 2025

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import open3d as o3d
import random
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Any

# -------------------------------------------------
# Setup Logging
# -------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Set Seeds for Reproducibility and Device
# -------------------------------------------------
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# -------------------------------------------------
# MeshSurface Dataclass to encapsulate mesh info.
# -------------------------------------------------
@dataclass
class MeshSurface:
    vertices: torch.Tensor                       # (N, 3)
    edges: List[List[int]]
    skeleton_curves: List[np.ndarray]
    vertices_np: np.ndarray
    padded_indices: torch.Tensor = field(init=False)
    pad_mask: torch.Tensor = field(init=False)
    edge_tensor: torch.Tensor = field(init=False)
    displacement: torch.Tensor = field(init=False)
    skeleton_points: torch.Tensor = field(init=False)

    def __post_init__(self):
        num_vertices: int = self.vertices.shape[0]
        # Precompute neighbor info.
        neighbors: Dict[int, Set[int]] = compute_neighbors(num_vertices, self.edges)
        self.padded_indices, self.pad_mask = pad_neighbors(neighbors, num_vertices)
        # Precompute edge tensor for elasticity term.
        self.edge_tensor = torch.tensor(self.edges, device=self.vertices.device, dtype=torch.long)
        # Initialize displacement variable.
        self.displacement = torch.zeros_like(self.vertices, requires_grad=True)
        # Skeleton points will be assigned later.
        self.skeleton_points = torch.empty((0, 3), device=self.vertices.device)

# -------------------------------------------------
# Helper Functions for Neighbor Computation
# -------------------------------------------------
def compute_neighbors(num_vertices: int, edges: List[List[int]]) -> Dict[int, Set[int]]:
    neighbors: Dict[int, Set[int]] = {i: set() for i in range(num_vertices)}
    for i, j in edges:
        neighbors[i].add(j)
        neighbors[j].add(i)
    return neighbors

def pad_neighbors(neighbors: Dict[int, Set[int]], N: int) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len: int = max(len(neighbors[i]) for i in range(N))
    padded: List[List[int]] = []
    mask: List[List[float]] = []
    for i in range(N):
        nlist = list(neighbors[i])
        L = len(nlist)
        padded.append(nlist + [0] * (max_len - L))
        mask.append([1.0] * L + [0.0] * (max_len - L))
    padded_tensor: torch.Tensor = torch.tensor(padded, dtype=torch.long, device=device)
    mask_tensor: torch.Tensor = torch.tensor(mask, dtype=torch.float32, device=device)
    return padded_tensor, mask_tensor

# -------------------------------------------------
# Energy Terms
# -------------------------------------------------
def arap_term_vectorized(vertices: torch.Tensor,
                         displacements: torch.Tensor,
                         padded_indices: torch.Tensor,
                         pad_mask: torch.Tensor,
                         lambda_arap: float = 1.0,
                         eps: float = 1e-8) -> torch.Tensor:
    """
    ARAP energy term with a small regularizer for stability.
    """
    deformed: torch.Tensor = vertices + displacements
    N, M = padded_indices.shape
    v_neighbors: torch.Tensor = vertices[padded_indices]         # (N, M, 3)
    d_neighbors: torch.Tensor = deformed[padded_indices]           # (N, M, 3)
    v_i: torch.Tensor = vertices.unsqueeze(1).expand(-1, M, -1)    # (N, M, 3)
    d_i: torch.Tensor = deformed.unsqueeze(1).expand(-1, M, -1)      # (N, M, 3)
    orig_diff: torch.Tensor = (v_i - v_neighbors) 
    def_diff: torch.Tensor = (d_i - d_neighbors)
    mask_exp: torch.Tensor = pad_mask.unsqueeze(-1)
    orig_diff = orig_diff * mask_exp
    def_diff = def_diff * mask_exp

    # Covariance matrix with regularization for stability.
    covariance: torch.Tensor = torch.einsum('nmi,nmj->nij', def_diff, orig_diff)
    covariance += eps * torch.eye(3, device=vertices.device).unsqueeze(0)
    covariance_fp32: torch.Tensor = covariance.float()  # full precision for SVD
    U, S, Vh = torch.linalg.svd(covariance_fp32, full_matrices=False)
    R: torch.Tensor = torch.matmul(Vh.transpose(1, 2), U.transpose(1, 2))
    det_R: torch.Tensor = torch.linalg.det(R.float())
    reflection_mask: torch.Tensor = det_R < 0
    if reflection_mask.any():
        Vh_corrected: torch.Tensor = Vh.clone()
        Vh_corrected[reflection_mask, -1, :] = -Vh_corrected[reflection_mask, -1, :]
        R_corrected: torch.Tensor = torch.matmul(Vh_corrected.transpose(1, 2), U.transpose(1, 2))
        R = torch.where(reflection_mask.view(-1, 1, 1), R_corrected, R)
    R = R.to(vertices.dtype)
    
    predicted: torch.Tensor = torch.matmul(orig_diff, R.transpose(1, 2))
    residual: torch.Tensor = def_diff - predicted
    error: torch.Tensor = torch.sum(residual**2, dim=2) * pad_mask
    return lambda_arap * error.sum()

def laplacian_term_vectorized(vertices: torch.Tensor,
                              displacements: torch.Tensor,
                              padded_indices: torch.Tensor,
                              pad_mask: torch.Tensor,
                              lambda_lap: float = 1.0) -> torch.Tensor:
    deformed: torch.Tensor = vertices + displacements
    orig_neighbors: torch.Tensor = vertices[padded_indices]
    def_neighbors: torch.Tensor = deformed[padded_indices]
    mask_exp: torch.Tensor = pad_mask.unsqueeze(-1)
    orig_neighbors = orig_neighbors * mask_exp
    def_neighbors = def_neighbors * mask_exp
    count: torch.Tensor = pad_mask.sum(dim=1, keepdim=True)
    count = torch.clamp(count, min=1.0)
    avg_orig: torch.Tensor = orig_neighbors.sum(dim=1) / count
    avg_def: torch.Tensor = def_neighbors.sum(dim=1) / count
    L_def: torch.Tensor = deformed - avg_def
    L_orig: torch.Tensor = vertices - avg_orig
    lap_diff: torch.Tensor = torch.norm(L_def - L_orig, dim=1)**2
    return lambda_lap * lap_diff.sum()

def sdf_term(vertices: torch.Tensor,
             displacements: torch.Tensor,
             skeleton_points: torch.Tensor,
             tau_sdf: float = 0.01,
             lambda_sdf: float = 1.0) -> torch.Tensor:
    deformed: torch.Tensor = vertices + displacements
    diff: torch.Tensor = deformed.unsqueeze(1) - skeleton_points.unsqueeze(0)
    dists: torch.Tensor = torch.norm(diff, dim=2)
    weights: torch.Tensor = F.softmax(-dists / tau_sdf, dim=1)
    d_soft: torch.Tensor = torch.sum(weights * dists, dim=1)
    return lambda_sdf * torch.sum(d_soft**2)

def smooth_barrier(d: torch.Tensor, delta: float = 0.05, beta: float = 10.0) -> torch.Tensor:
    return F.softplus(delta - d, beta=beta)**2

def data_term(vertices: torch.Tensor,
              displacements: torch.Tensor,
              skeleton_points: torch.Tensor,
              tau: float = 0.01,
              lambda_data: float = 1.0) -> torch.Tensor:
    deformed: torch.Tensor = vertices + displacements
    diff: torch.Tensor = deformed.unsqueeze(1) - skeleton_points.unsqueeze(0)
    dists_sq: torch.Tensor = torch.sum(diff**2, dim=2)
    weights: torch.Tensor = F.softmax(-dists_sq / tau, dim=1)
    target: torch.Tensor = torch.sum(weights.unsqueeze(2) * skeleton_points.unsqueeze(0), dim=1)
    E_data: torch.Tensor = torch.sum((deformed - target)**2)
    return lambda_data * E_data

def displacement_term(displacements: torch.Tensor, lambda_disp: float = 1e-3) -> torch.Tensor:
    return lambda_disp * torch.sum(displacements**2)

def elasticity_term(vertices: torch.Tensor,
                      displacements: torch.Tensor,
                      edge_tensor: torch.Tensor,
                      lambda_elastic: float = 1.0) -> torch.Tensor:
    deformed: torch.Tensor = vertices + displacements
    v1: torch.Tensor = deformed[edge_tensor[:, 0]]
    v2: torch.Tensor = deformed[edge_tensor[:, 1]]
    orig_v1: torch.Tensor = vertices[edge_tensor[:, 0]]
    orig_v2: torch.Tensor = vertices[edge_tensor[:, 1]]
    E_elastic: torch.Tensor = torch.norm(v1 - v2, dim=1) - torch.norm(orig_v1 - orig_v2, dim=1)
    return lambda_elastic * torch.sum(E_elastic**2)

def self_intersection_term_batched(vertices: torch.Tensor,
                                   displacements: torch.Tensor,
                                   lambda_self: float = 1.0,
                                   delta: float = 0.05,
                                   beta: float = 10.0,
                                   batch_size: int = 8) -> torch.Tensor:
    """
    Computes the self-intersection penalty in a batched manner to avoid OOM errors.
    
    Args:
        vertices: Original vertex positions (N, 3).
        displacements: Displacement tensor (N, 3).
        lambda_self: Weight for the self-intersection term.
        delta, beta: Parameters for the smooth barrier function.
        batch_size: Number of vertices to process at once.
        
    Returns:
        A scalar tensor representing the self-intersection penalty.
    """
    deformed = vertices + displacements  # (N, 3)
    N = deformed.shape[0]
    total_penalty = 0.0
    # Process in batches over the first index.
    for i in range(0, N, batch_size):
        end_i = min(N, i + batch_size)
        # Process batch i:end_i against all vertices.
        diff = deformed[i:end_i].unsqueeze(1) - deformed.unsqueeze(0)  # (batch, N, 3)
        dists = torch.sqrt(torch.sum(diff**2, dim=2) + 1e-8)  # (batch, N)
        # Create a mask: for the batch, set self pairs to zero.
        mask = torch.ones_like(dists)
        if end_i - i == N:  # If batch equals entire set, set diagonal to zero.
            mask.fill_diagonal_(0)
        else:
            # For indices in the batch, zero out corresponding positions.
            for j in range(i, end_i):
                mask[j - i, j] = 0
        penalty = torch.sum(torch.nn.functional.softplus(delta - dists, beta=beta)**2 * mask)
        total_penalty += penalty
    return lambda_self * total_penalty


def inter_mesh_intersection_term_vectorized(meshes: List[MeshSurface],
                                            lambda_inter: float = 0.1,
                                            delta_inter: float = 0.05,
                                            beta_inter: float = 10.0) -> torch.Tensor:
    """
    Vectorized computation of the inter-mesh intersection penalty.
    All deformed vertices are concatenated and a pairwise distance matrix is computed.
    Only pairs coming from different meshes are penalized.
    """
    deformed_list: List[torch.Tensor] = []
    mesh_ids: List[torch.Tensor] = []
    for i, mesh in enumerate(meshes):
        V_i = mesh.vertices + mesh.displacement  # (N_i, 3)
        deformed_list.append(V_i)
        mesh_ids.append(torch.full((V_i.shape[0],), i, device=V_i.device, dtype=torch.long))
    all_deformed: torch.Tensor = torch.cat(deformed_list, dim=0)  # (Total, 3)
    all_ids: torch.Tensor = torch.cat(mesh_ids, dim=0)             # (Total,)

    total = all_deformed.shape[0]
    diff = all_deformed.unsqueeze(0) - all_deformed.unsqueeze(1)     # (Total, Total, 3)
    dists = torch.sqrt(torch.sum(diff**2, dim=2) + 1e-8)              # (Total, Total)

    # Create mask: penalize only pairs from different meshes, exclude self-pairs.
    id_mask = (all_ids.unsqueeze(0) != all_ids.unsqueeze(1)).float()
    self_mask = 1 - torch.eye(total, device=device)
    mask = id_mask * self_mask

    penalty = F.softplus(delta_inter - dists, beta=beta_inter)**2 * mask
    # Divide by 2 to avoid double counting symmetric pairs.
    total_penalty = penalty.sum() / 2.0
    return lambda_inter * total_penalty

# -------------------------------------------------
# Total Energy for a Single Mesh (Vectorized)
# -------------------------------------------------
def total_energy_vectorized(mesh: MeshSurface,
                            lambda_data: float = 1.0,
                            lambda_disp: float = 1e-3,
                            lambda_elastic: float = 1.0,
                            lambda_self: float = 1.0,
                            lambda_lap: float = 1.0,
                            lambda_arap: float = 1.0,
                            lambda_sdf: float = 1.0,
                            tau: float = 0.01,
                            delta: float = 0.05,
                            beta: float = 10.0,
                            tau_sdf: float = 0.01,
                            batch_size: int = 8) -> torch.Tensor:
    E = (data_term(mesh.vertices, mesh.displacement, mesh.skeleton_points, tau, lambda_data) +
         displacement_term(mesh.displacement, lambda_disp) +
         elasticity_term(mesh.vertices, mesh.displacement, mesh.edge_tensor, lambda_elastic) +
         self_intersection_term_batched(mesh.vertices, mesh.displacement, lambda_self, delta, beta, batch_size) +
         laplacian_term_vectorized(mesh.vertices, mesh.displacement, mesh.padded_indices, mesh.pad_mask, lambda_lap) +
         arap_term_vectorized(mesh.vertices, mesh.displacement, mesh.padded_indices, mesh.pad_mask, lambda_arap) +
         sdf_term(mesh.vertices, mesh.displacement, mesh.skeleton_points, tau_sdf, lambda_sdf))
    return E

# -------------------------------------------------
# Open3D Visualization Helpers
# -------------------------------------------------
def create_line_set(points: np.ndarray, edges: List[List[int]], color: List[float] = [1, 0, 0]) -> o3d.geometry.LineSet:
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(edges)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(edges))])
    return line_set

def create_point_cloud(points: np.ndarray, color: List[float] = [0, 0, 1]) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.tile(np.array(color), (points.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def create_skeleton_lines(skel_line: np.ndarray, color: List[float] = [0, 1, 0]) -> o3d.geometry.LineSet:
    num_pts = len(skel_line)
    skel_edges = [[i, i+1] for i in range(num_pts - 1)]
    return create_line_set(skel_line, skel_edges, color=color)

# -------------------------------------------------
# Synthetic Data Simulation (Smoother Meshes)
# -------------------------------------------------
def simulate_surface(surface_idx: int, N_side: int = 20,
                     num_skel_curves: int = 10, num_skel_points: int = 7) -> MeshSurface:
    x = np.linspace(0, 1, N_side)
    y = np.linspace(0, 1, N_side)
    X, Y = np.meshgrid(x, y)
    X_noise = X + 0.005 * np.random.randn(*X.shape)
    Y_noise = Y + 0.005 * np.random.randn(*Y.shape)
    Z = 0.02 * np.random.randn(*X.shape) + 0.25 * surface_idx
    grid = np.stack([X_noise, Y_noise, Z], axis=-1)
    vertices_np = grid.reshape(-1, 3).astype(np.float32)
    vertices = torch.tensor(vertices_np, device=device)
    
    edges: List[List[int]] = []
    for i in range(N_side):
        for j in range(N_side):
            idx = i * N_side + j
            if j < N_side - 1:
                edges.append([idx, idx + 1])
            if i < N_side - 1:
                edges.append([idx, idx + N_side])
    
    skeleton_curves: List[np.ndarray] = []
    for curve_idx in range(num_skel_curves):
        skel_curve: List[List[float]] = []
        base_y = (curve_idx + 1) / (num_skel_curves + 1)
        for k in range(num_skel_points):
            x_val = k / (num_skel_points - 1)
            y_val = base_y + 0.02 * np.sin(2 * np.pi * x_val + surface_idx) + 0.005 * np.random.randn()
            z_val = 0.2 * surface_idx + 0.002 * np.random.randn()
            skel_curve.append([x_val, y_val, z_val])
        skeleton_curves.append(np.array(skel_curve, dtype=np.float32))
    
    mesh = MeshSurface(vertices=vertices, edges=edges,
                       skeleton_curves=skeleton_curves, vertices_np=vertices_np)
    return mesh

def assign_skeletons_to_mesh(mesh_vertices_np: np.ndarray, global_skel_curves: List[np.ndarray],
                             thresh_z: float = 0.1) -> torch.Tensor:
    mesh_z = np.mean(mesh_vertices_np[:, 2])
    assigned_curves: List[np.ndarray] = []
    for curve in global_skel_curves:
        curve_z = np.mean(curve[:, 2])
        if abs(curve_z - mesh_z) < thresh_z:
            assigned_curves.append(curve)
    if len(assigned_curves) == 0:
        assigned_curves = global_skel_curves
    all_points = np.vstack(assigned_curves)
    return torch.tensor(all_points, dtype=torch.float32, device=device)

# -------------------------------------------------
# Multi-Mesh Optimization with Scheduler
# -------------------------------------------------
def optimize_all_registration(meshes: List[MeshSurface],
                              global_skel_curves: List[np.ndarray],
                              num_iters: int = 1000,
                              lr: float = 1e-2,
                              lambda_data: float = 1.0,
                              lambda_disp: float = 1e-3,
                              lambda_elastic: float = 1.0,
                              lambda_self: float = 1e-1,
                              lambda_lap: float = 1e-1,
                              lambda_arap: float = 1.0,
                              lambda_sdf: float = 1.0,
                              tau: float = 0.01,
                              delta: float = 0.05,
                              beta: float = 10.0,
                              tau_sdf: float = 0.01,
                              lambda_inter: float = 0.1,
                              delta_inter: float = 0.05,
                              beta_inter: float = 10.0,
                              batch_size: int = 8) -> List[torch.Tensor]:
    # Assign skeleton points for each mesh.
    for mesh in meshes:
        mesh.skeleton_points = assign_skeletons_to_mesh(mesh.vertices_np, global_skel_curves, thresh_z=0.1)
    
    displacement_list = [mesh.displacement for mesh in meshes]
    optimizer = torch.optim.AdamW(displacement_list, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    use_amp: bool = device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp)
    
    best_energy = float('inf')
    no_improvement_count = 0
    tolerance = 1e-6  # minimal energy improvement threshold

    pbar = tqdm(range(num_iters), desc="Registration Optimization", ncols=100)
    start_time = time.time()
    for it in pbar:
        optimizer.zero_grad()
        E_total = torch.tensor(0.0, device=device)
        for mesh in meshes:
            E_total += total_energy_vectorized(mesh,
                                               lambda_data, lambda_disp,
                                               lambda_elastic, lambda_self,
                                               lambda_lap, lambda_arap,
                                               lambda_sdf, tau, delta, beta,
                                               tau_sdf, batch_size)
        E_total += inter_mesh_intersection_term_vectorized(meshes, lambda_inter, delta_inter, beta_inter)
        
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            E = E_total
        scaler.scale(E).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        current_energy = E.item()
        pbar.set_postfix({"Energy": f"{current_energy:.2f}"})
        
        if current_energy < best_energy - tolerance:
            best_energy = current_energy
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        if no_improvement_count >= 400:
            pbar.write(f"Early stopping at iteration {it} as energy did not decrease for 400 consecutive iterations.")
            break

    pbar.close()
    logger.info("Total Optimization time: {:.2f}s".format(time.time() - start_time))
    
    new_vertices_list: List[torch.Tensor] = []
    for mesh in meshes:
        new_vertices_list.append(mesh.vertices + mesh.displacement)
    return new_vertices_list

# -------------------------------------------------
# Main Script
# -------------------------------------------------
if __name__ == "__main__":
    num_surfaces: int = 3
    num_skel_curves_per_surface: int = 10
    meshes: List[MeshSurface] = []
    global_skel_curves: List[np.ndarray] = []
    all_mesh_geoms_initial: List[o3d.geometry.Geometry] = []
    all_mesh_geoms_registered: List[o3d.geometry.Geometry] = []
    all_skel_geoms: List[o3d.geometry.Geometry] = []
    
    colors_mesh: List[List[float]] = [[0, 0, 1], [0, 0.6, 1], [0, 1, 1]]
    colors_skel: List[List[float]] = [[1, 0, 0], [1, 0.5, 0], [1, 0, 1]]

    
    # Simulate surfaces and collect global skeleton curves.
    for s_idx in range(num_surfaces):
        mesh = simulate_surface(surface_idx=s_idx, N_side=20,
                                num_skel_curves=num_skel_curves_per_surface,
                                num_skel_points=7)
        meshes.append(mesh)
        global_skel_curves.extend(mesh.skeleton_curves)
        
        # Create Open3D geometries for the mesh.
        mesh_pcd = create_point_cloud(mesh.vertices_np, color=colors_mesh[s_idx % len(colors_mesh)])
        mesh_lines = create_line_set(mesh.vertices_np, mesh.edges, color=[0.8, 0.8, 0.8])
        all_mesh_geoms_initial.extend([mesh_pcd, mesh_lines])
        for curve in mesh.skeleton_curves:
            skel_geom = create_skeleton_lines(curve, color=colors_skel[s_idx % len(colors_skel)])
            all_skel_geoms.append(skel_geom)
    
    o3d.visualization.draw_geometries(all_mesh_geoms_initial + all_skel_geoms,
                                      window_name="Initial Meshes and Global Skeleton Curves")
    
    new_vertices_list = optimize_all_registration(meshes, global_skel_curves,
                                                  num_iters=1000, lr=1e-2,
                                                  lambda_data=1.0, lambda_disp=1e-3,
                                                  lambda_elastic=1.0, lambda_self=1e-1,
                                                  lambda_lap=1e-1, lambda_arap=1.0, lambda_sdf=1.0,
                                                  tau=0.01, delta=0.05, beta=10.0, tau_sdf=0.01,
                                                  lambda_inter=0.1, delta_inter=0.05, beta_inter=10.0)
    
    registered_meshes_np: List[np.ndarray] = []
    for new_vertices in new_vertices_list:
        registered_meshes_np.append(new_vertices.detach().cpu().numpy())
    
    for i, mesh in enumerate(meshes):
        reg_pcd = create_point_cloud(registered_meshes_np[i], color=colors_mesh[i % len(colors_mesh)])
        reg_lines = create_line_set(registered_meshes_np[i], mesh.edges, color=[0.8, 0.8, 0.8])
        all_mesh_geoms_registered.extend([reg_pcd, reg_lines])
    
    o3d.visualization.draw_geometries(all_mesh_geoms_registered + all_skel_geoms,
                                      window_name="Registered Meshes with Global Skeleton Curves")
