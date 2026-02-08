"""
Trainer for row/col conditioned displacement field prediction.

Trains a model to predict dense 3D displacement fields from extrapolated surfaces,
with optional SDT (Signed Distance Transform) prediction.
"""
import os
import json
import click
import torch
import wandb
import random
import accelerate
import numpy as np
from tqdm import tqdm

from vesuvius.neural_tracing.datasets.dataset_rowcol_cond import EdtSegDataset
from vesuvius.neural_tracing.loss.displacement_losses import surface_sampled_loss, smoothness_loss
from vesuvius.models.training.loss.nnunet_losses import DC_and_BCE_loss
from vesuvius.models.training.loss.skeleton_recall import DC_SkelREC_and_CE_loss
from vesuvius.models.training.optimizers import create_optimizer
from vesuvius.models.training.lr_schedulers import get_scheduler
from vesuvius.neural_tracing.models import make_model
from accelerate.utils import TorchDynamoPlugin

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)


def seed_worker(worker_id):
    """Seed worker for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate_with_padding(batch):
    """Collate batch with padding for variable-length point data."""
    # Stack fixed-size tensors normally
    vol = torch.stack([b['vol'] for b in batch])
    cond = torch.stack([b['cond'] for b in batch])
    extrap_surface = torch.stack([b['extrap_surface'] for b in batch])

    # Pad variable-length data
    coords_list = [b['extrap_coords'] for b in batch]
    disp_list = [b['gt_displacement'] for b in batch]
    weight_list = [
        b['point_weights'] if 'point_weights' in b else torch.ones(len(b['extrap_coords']), dtype=torch.float32)
        for b in batch
    ]
    max_points = max(len(c) for c in coords_list)

    B = len(batch)
    padded_coords = torch.zeros(B, max_points, 3)
    padded_disp = torch.zeros(B, max_points, 3)
    valid_mask = torch.zeros(B, max_points)
    padded_point_weights = torch.zeros(B, max_points)

    for i, (c, d, w) in enumerate(zip(coords_list, disp_list, weight_list)):
        n = len(c)
        padded_coords[i, :n] = c
        padded_disp[i, :n] = d
        valid_mask[i, :n] = 1.0
        padded_point_weights[i, :n] = w

    result = {
        'vol': vol, 'cond': cond, 'extrap_surface': extrap_surface,
        'extrap_coords': padded_coords, 'gt_displacement': padded_disp,
        'valid_mask': valid_mask, 'point_weights': padded_point_weights
    }

    # Optional SDT
    if 'sdt' in batch[0]:
        result['sdt'] = torch.stack([b['sdt'] for b in batch])

    # Optional heatmap target
    if 'heatmap_target' in batch[0]:
        result['heatmap_target'] = torch.stack([b['heatmap_target'] for b in batch])

    # Optional segmentation target (full segmentation + skeleton)
    if 'segmentation' in batch[0]:
        result['segmentation'] = torch.stack([b['segmentation'] for b in batch])
        result['segmentation_skel'] = torch.stack([b['segmentation_skel'] for b in batch])

    # Optional other_wraps
    if 'other_wraps' in batch[0]:
        result['other_wraps'] = torch.stack([b['other_wraps'] for b in batch])

    return result


def prepare_batch(batch, use_sdt=False, use_heatmap=False, use_segmentation=False):
    """Prepare batch tensors for training."""
    vol = batch['vol'].unsqueeze(1)                    # [B, 1, D, H, W]
    cond = batch['cond'].unsqueeze(1)                  # [B, 1, D, H, W]
    extrap_surf = batch['extrap_surface'].unsqueeze(1) # [B, 1, D, H, W]

    input_list = [vol, cond, extrap_surf]
    if 'other_wraps' in batch:
        other_wraps = batch['other_wraps'].unsqueeze(1)  # [B, 1, D, H, W]
        input_list.append(other_wraps)

    inputs = torch.cat(input_list, dim=1)  # [B, 3 or 4, D, H, W]

    extrap_coords = batch['extrap_coords']       # [B, N, 3]
    gt_displacement = batch['gt_displacement']   # [B, N, 3]
    valid_mask = batch['valid_mask']             # [B, N]
    point_weights = batch['point_weights'] if 'point_weights' in batch else torch.ones_like(valid_mask)  # [B, N]

    sdt_target = batch['sdt'].unsqueeze(1) if use_sdt and 'sdt' in batch else None  # [B, 1, D, H, W]
    heatmap_target = batch['heatmap_target'].unsqueeze(1) if use_heatmap and 'heatmap_target' in batch else None  # [B, 1, D, H, W]

    seg_target = None
    seg_skel = None
    if use_segmentation and 'segmentation' in batch:
        seg_target = batch['segmentation'].unsqueeze(1)  # [B, 1, D, H, W]
        seg_skel = batch['segmentation_skel'].unsqueeze(1)  # [B, 1, D, H, W]

    return inputs, extrap_coords, gt_displacement, valid_mask, point_weights, sdt_target, heatmap_target, seg_target, seg_skel


def rasterize_sparse_to_slice(coords, values, valid_mask, slice_idx, shape, tol=1.5, axis='z'):
    """Rasterize sparse 3D points to a 2D slice.

    Args:
        coords: (N, 3) array of z, y, x coordinates
        values: (N,) array of values at each point
        valid_mask: (N,) boolean mask for valid points
        slice_idx: coordinate of the slice along the specified axis
        shape: (dim0, dim1) output shape
        tol: tolerance for including points near the slice
        axis: 'z' (XY plane), 'y' (XZ plane), or 'x' (YZ plane)

    Returns:
        2D array with rasterized values (0 where no points)
    """
    dim0, dim1 = shape
    result = np.zeros((dim0, dim1), dtype=np.float32)
    counts = np.zeros((dim0, dim1), dtype=np.float32)

    for i in range(len(coords)):
        if not valid_mask[i]:
            continue
        z, y, x = coords[i]

        if axis == 'z':
            # Z-slice: output is (H, W) indexed by (y, x)
            if abs(z - slice_idx) <= tol:
                i0, i1 = int(round(y)), int(round(x))
                if 0 <= i0 < dim0 and 0 <= i1 < dim1:
                    result[i0, i1] += values[i]
                    counts[i0, i1] += 1
        elif axis == 'y':
            # Y-slice: output is (D, W) indexed by (z, x)
            if abs(y - slice_idx) <= tol:
                i0, i1 = int(round(z)), int(round(x))
                if 0 <= i0 < dim0 and 0 <= i1 < dim1:
                    result[i0, i1] += values[i]
                    counts[i0, i1] += 1
        elif axis == 'x':
            # X-slice: output is (D, H) indexed by (z, y)
            if abs(x - slice_idx) <= tol:
                i0, i1 = int(round(z)), int(round(y))
                if 0 <= i0 < dim0 and 0 <= i1 < dim1:
                    result[i0, i1] += values[i]
                    counts[i0, i1] += 1

    # Average overlapping points
    return np.divide(result, counts, where=counts > 0, out=result)


def make_visualization(inputs, disp_pred, extrap_coords, gt_displacement, valid_mask,
                       sdt_pred=None, sdt_target=None,
                       heatmap_pred=None, heatmap_target=None,
                       seg_pred=None, seg_target=None,
                       save_path=None):
    """Create and save PNG visualization of Z, Y, and X slices."""
    import matplotlib.pyplot as plt

    b = 0
    D, H, W = inputs.shape[2], inputs.shape[3], inputs.shape[4]

    # Precompute 3D arrays
    vol_3d = inputs[b, 0].cpu().numpy()
    cond_3d = inputs[b, 1].cpu().numpy()
    extrap_surf_3d = inputs[b, 2].cpu().numpy()
    other_wraps_3d = inputs[b, 3].cpu().numpy() if inputs.shape[1] > 3 else None

    # Displacement: [3, D, H, W] where components are (dz, dy, dx)
    disp_3d = disp_pred[b].cpu().numpy()
    disp_mag_3d = np.linalg.norm(disp_3d, axis=0)

    # GT displacement processing
    gt_disp_np = gt_displacement[b].cpu().numpy()  # (N, 3)
    gt_disp_mag = np.linalg.norm(gt_disp_np, axis=-1)
    coords_np = extrap_coords[b].cpu().numpy()  # (N, 3) - z, y, x
    valid_np = valid_mask[b].cpu().numpy().astype(bool)

    # Precompute pred displacement sampled at extrap coords
    pred_sampled_mag = np.zeros(len(coords_np), dtype=np.float32)
    for i in range(len(coords_np)):
        if valid_np[i]:
            zi = np.clip(int(round(coords_np[i, 0])), 0, D - 1)
            yi = np.clip(int(round(coords_np[i, 1])), 0, H - 1)
            xi = np.clip(int(round(coords_np[i, 2])), 0, W - 1)
            pred_sampled_mag[i] = np.linalg.norm(disp_3d[:, zi, yi, xi])

    # === Compute statistics for text panel ===
    # Sample predicted displacement vectors (not just magnitude) at extrap coords
    pred_sampled_vectors = np.zeros((len(coords_np), 3), dtype=np.float32)
    for i in range(len(coords_np)):
        if valid_np[i]:
            zi = np.clip(int(round(coords_np[i, 0])), 0, D - 1)
            yi = np.clip(int(round(coords_np[i, 1])), 0, H - 1)
            xi = np.clip(int(round(coords_np[i, 2])), 0, W - 1)
            pred_sampled_vectors[i] = disp_3d[:, zi, yi, xi]

    # Filter to valid points only
    valid_gt = gt_disp_np[valid_np]         # [N_valid, 3]
    valid_pred = pred_sampled_vectors[valid_np]  # [N_valid, 3]
    valid_gt_mag = gt_disp_mag[valid_np]    # [N_valid]
    valid_pred_mag = pred_sampled_mag[valid_np]  # [N_valid]

    # Per-component stats (dz=0, dy=1, dx=2)
    component_names = ['dz', 'dy', 'dx']
    gt_comp_stats = {}
    pred_comp_stats = {}
    for c, name in enumerate(component_names):
        gt_vals = valid_gt[:, c]
        pred_vals = valid_pred[:, c]
        gt_comp_stats[name] = {
            'mean': np.mean(gt_vals) if len(gt_vals) > 0 else 0.0,
            'median': np.median(gt_vals) if len(gt_vals) > 0 else 0.0,
            'max': np.max(np.abs(gt_vals)) if len(gt_vals) > 0 else 0.0
        }
        pred_comp_stats[name] = {
            'mean': np.mean(pred_vals) if len(pred_vals) > 0 else 0.0,
            'median': np.median(pred_vals) if len(pred_vals) > 0 else 0.0,
            'max': np.max(np.abs(pred_vals)) if len(pred_vals) > 0 else 0.0
        }

    # Magnitude stats
    gt_mag_stats = {
        'mean': np.mean(valid_gt_mag) if len(valid_gt_mag) > 0 else 0.0,
        'median': np.median(valid_gt_mag) if len(valid_gt_mag) > 0 else 0.0,
        'max': np.max(valid_gt_mag) if len(valid_gt_mag) > 0 else 0.0
    }
    pred_mag_stats = {
        'mean': np.mean(valid_pred_mag) if len(valid_pred_mag) > 0 else 0.0,
        'median': np.median(valid_pred_mag) if len(valid_pred_mag) > 0 else 0.0,
        'max': np.max(valid_pred_mag) if len(valid_pred_mag) > 0 else 0.0
    }

    # Residual error: |pred - gt| at each valid point
    residual_vectors = valid_pred - valid_gt
    residual_mag = np.linalg.norm(residual_vectors, axis=-1)
    residual_stats = {
        'mean': np.mean(residual_mag) if len(residual_mag) > 0 else 0.0,
        'median': np.median(residual_mag) if len(residual_mag) > 0 else 0.0,
        'max': np.max(residual_mag) if len(residual_mag) > 0 else 0.0
    }

    # % Improvement: (gt_mag - residual_mag) / gt_mag * 100
    # Filter to points with meaningful gt displacement to avoid division instability
    meaningful_mask = valid_gt_mag > 0.01
    if np.sum(meaningful_mask) > 0:
        meaningful_gt = valid_gt_mag[meaningful_mask]
        meaningful_resid = residual_mag[meaningful_mask]
        improvement_per_point = (meaningful_gt - meaningful_resid) / meaningful_gt * 100
        # Filter NaNs and use median for outlier robustness
        improvement_per_point = improvement_per_point[~np.isnan(improvement_per_point)]
        pct_improvement = np.median(improvement_per_point) if len(improvement_per_point) > 0 else 0.0
    else:
        pct_improvement = 0.0

    # Sample predicted field at conditioning point locations (should be ~0)
    cond_coords = np.argwhere(cond_3d > 0.5)  # [N_cond, 3] as (z, y, x)
    if len(cond_coords) > 0:
        cond_pred_mags = np.array([np.linalg.norm(disp_3d[:, z, y, x]) for z, y, x in cond_coords])
        cond_disp_stats = {
            'mean': np.mean(cond_pred_mags),
            'max': np.max(cond_pred_mags),
            'n_points': len(cond_coords)
        }
    else:
        cond_disp_stats = {'mean': 0.0, 'max': 0.0, 'n_points': 0}

    # Compute shared colormap ranges
    disp_vmax = np.percentile(disp_mag_3d, 99)
    gt_vmax = max(disp_vmax, gt_disp_mag[valid_np].max() if valid_np.any() else 1.0)
    disp_vmax_comp = np.percentile(np.abs(disp_3d), 99)

    # Optional 3D arrays
    sdt_pred_3d = sdt_pred[b, 0].cpu().numpy() if sdt_pred is not None else None
    sdt_gt_3d = sdt_target[b, 0].cpu().numpy() if sdt_target is not None else None
    sdt_vmax = max(np.abs(sdt_pred_3d).max(), np.abs(sdt_gt_3d).max()) if sdt_pred_3d is not None else 1.0
    hm_pred_3d = torch.sigmoid(heatmap_pred[b, 0]).cpu().numpy() if heatmap_pred is not None else None
    hm_gt_3d = heatmap_target[b, 0].cpu().numpy() if heatmap_target is not None else None

    # Segmentation: pred is [B, 2, D, H, W], target is [B, 1, D, H, W]
    has_seg = seg_pred is not None and seg_target is not None
    seg_pred_3d = seg_pred[b].argmax(dim=0).cpu().numpy() if seg_pred is not None else None  # [D, H, W]
    seg_gt_3d = seg_target[b, 0].cpu().numpy() if seg_target is not None else None  # [D, H, W]

    # Setup figure: 6 rows (2 per slice orientation), variable columns + text panel
    from matplotlib.gridspec import GridSpec
    n_cols = 5
    if sdt_pred is not None:
        n_cols += 1
    if heatmap_pred is not None:
        n_cols += 1
    if has_seg:
        n_cols += 1

    # Create figure with extra column for stats text panel
    fig = plt.figure(figsize=(4 * n_cols + 4, 24))
    gs = GridSpec(6, n_cols + 1, figure=fig, width_ratios=[1]*n_cols + [1.2], wspace=0.3)

    # Create axes for the visualization columns
    axes = np.empty((6, n_cols), dtype=object)
    for row in range(6):
        for col in range(n_cols):
            axes[row, col] = fig.add_subplot(gs[row, col])

    # Text panel spanning all rows on the right
    ax_text = fig.add_subplot(gs[:, n_cols])
    ax_text.axis('off')

    # Slice indices
    z0, y0, x0 = D // 2, H // 2, W // 2

    def plot_slice_pair(row_base, vol_slice, cond_slice, extrap_slice, other_slice,
                        disp_slice, disp_comps, gt_raster, pred_sampled_raster,
                        sdt_pred_slice, sdt_gt_slice, hm_pred_slice, hm_gt_slice,
                        seg_pred_slice, seg_gt_slice,
                        extent, slice_label, xlabel, ylabel):
        """Plot a pair of rows for one slice orientation."""
        ax0 = axes[row_base]
        ax1 = axes[row_base + 1]

        # Normalize volume for overlay
        vol_norm = (vol_slice - vol_slice.min()) / (vol_slice.max() - vol_slice.min() + 1e-8)

        # Row 0: Volume, Cond, Extrap, dense pred disp mag, sparse GT disp mag at extrap coords
        ax0[0].imshow(vol_slice, cmap='gray', extent=extent)
        ax0[0].set_title(f'Volume ({slice_label})')
        ax0[0].set_ylabel(ylabel)

        ax0[1].imshow(cond_slice, cmap='gray', extent=extent)
        ax0[1].set_title('Conditioning')
        ax0[1].set_yticks([])

        ax0[2].imshow(extrap_slice, cmap='gray', extent=extent)
        ax0[2].set_title('Extrap Surface')
        ax0[2].set_yticks([])

        ax0[3].imshow(disp_slice, cmap='hot', vmin=0, vmax=disp_vmax, extent=extent)
        ax0[3].set_title('Pred Disp Mag (dense)')
        ax0[3].set_yticks([])

        ax0[4].imshow(gt_raster, cmap='hot', vmin=0, vmax=gt_vmax, extent=extent)
        ax0[4].set_title('GT Disp Mag @ Extrap')
        ax0[4].set_yticks([])

        # Row 1: dz, dy, dx, Overlay, sparse pred disp mag sampled at extrap coords
        ax1[0].imshow(disp_comps[0], cmap='RdBu', vmin=-disp_vmax_comp, vmax=disp_vmax_comp, extent=extent)
        ax1[0].set_title('dz (pred)')
        ax1[0].set_xlabel(xlabel)
        ax1[0].set_ylabel(ylabel)

        ax1[1].imshow(disp_comps[1], cmap='RdBu', vmin=-disp_vmax_comp, vmax=disp_vmax_comp, extent=extent)
        ax1[1].set_title('dy (pred)')
        ax1[1].set_xlabel(xlabel)
        ax1[1].set_yticks([])

        ax1[2].imshow(disp_comps[2], cmap='RdBu', vmin=-disp_vmax_comp, vmax=disp_vmax_comp, extent=extent)
        ax1[2].set_title('dx (pred)')
        ax1[2].set_xlabel(xlabel)
        ax1[2].set_yticks([])

        # Overlay
        overlay = np.stack([vol_norm, vol_norm, vol_norm], axis=-1)
        overlay[cond_slice > 0.5, 1] = 1.0  # green
        overlay[extrap_slice > 0.5, 0] = 1.0  # red
        if other_slice is not None:
            overlay[other_slice > 0.5, 2] = 1.0  # blue
        ax1[3].imshow(overlay, extent=extent)
        title = 'Cond(G)+Extrap(R)' + ('+Other(B)' if other_slice is not None else '')
        ax1[3].set_title(title)
        ax1[3].set_xlabel(xlabel)
        ax1[3].set_yticks([])

        ax1[4].imshow(pred_sampled_raster, cmap='hot', vmin=0, vmax=gt_vmax, extent=extent)
        ax1[4].set_title('Pred Disp Mag @ Extrap')
        ax1[4].set_xlabel(xlabel)
        ax1[4].set_yticks([])

        # Optional columns
        col_idx = 5
        if sdt_pred_slice is not None:
            ax0[col_idx].imshow(sdt_pred_slice, cmap='RdBu', vmin=-sdt_vmax, vmax=sdt_vmax, extent=extent)
            ax0[col_idx].set_title('SDT Pred')
            ax0[col_idx].set_yticks([])
            ax1[col_idx].imshow(sdt_gt_slice if sdt_gt_slice is not None else np.zeros_like(sdt_pred_slice),
                                cmap='RdBu', vmin=-sdt_vmax, vmax=sdt_vmax, extent=extent)
            ax1[col_idx].set_title('SDT GT')
            ax1[col_idx].set_xlabel(xlabel)
            ax1[col_idx].set_yticks([])
            col_idx += 1

        if hm_pred_slice is not None:
            ax0[col_idx].imshow(hm_pred_slice, cmap='hot', vmin=0, vmax=1, extent=extent)
            ax0[col_idx].set_title('Heatmap Pred')
            ax0[col_idx].set_yticks([])
            ax1[col_idx].imshow(hm_gt_slice if hm_gt_slice is not None else np.zeros_like(hm_pred_slice),
                                cmap='hot', vmin=0, vmax=1, extent=extent)
            ax1[col_idx].set_title('Heatmap GT')
            ax1[col_idx].set_xlabel(xlabel)
            ax1[col_idx].set_yticks([])
            col_idx += 1

        if seg_pred_slice is not None and seg_gt_slice is not None:
            # Create overlay: green=target, red=pred, yellow=both agree
            seg_overlay = np.zeros((*seg_pred_slice.shape, 3), dtype=np.float32)
            pred_mask = seg_pred_slice > 0.5
            gt_mask = seg_gt_slice > 0.5 if seg_gt_slice is not None else np.zeros_like(pred_mask)
            # Red channel: prediction
            seg_overlay[..., 0] = pred_mask.astype(np.float32)
            # Green channel: target
            seg_overlay[..., 1] = gt_mask.astype(np.float32)
            # Where both agree (yellow), both R and G are 1
            ax0[col_idx].imshow(seg_overlay, extent=extent)
            ax0[col_idx].set_title('Seg Pred(R) GT(G)')
            ax0[col_idx].set_yticks([])
            # Show volume with seg overlay for context
            vol_norm_local = (vol_slice - vol_slice.min()) / (vol_slice.max() - vol_slice.min() + 1e-8)
            vol_rgb = np.stack([vol_norm_local, vol_norm_local, vol_norm_local], axis=-1)
            vol_rgb[pred_mask, 0] = np.clip(vol_rgb[pred_mask, 0] + 0.5, 0, 1)
            vol_rgb[gt_mask, 1] = np.clip(vol_rgb[gt_mask, 1] + 0.5, 0, 1)
            ax1[col_idx].imshow(vol_rgb, extent=extent)
            ax1[col_idx].set_title('Vol + Seg Overlay')
            ax1[col_idx].set_xlabel(xlabel)
            ax1[col_idx].set_yticks([])

    # --- Z-slice (XY plane) ---
    z_extent = [-W/2, W/2, H/2, -H/2]
    gt_z = rasterize_sparse_to_slice(coords_np, gt_disp_mag, valid_np, z0, (H, W), axis='z')
    pred_z = rasterize_sparse_to_slice(coords_np, pred_sampled_mag, valid_np, z0, (H, W), axis='z')
    plot_slice_pair(
        row_base=0,
        vol_slice=vol_3d[z0], cond_slice=cond_3d[z0], extrap_slice=extrap_surf_3d[z0],
        other_slice=other_wraps_3d[z0] if other_wraps_3d is not None else None,
        disp_slice=disp_mag_3d[z0], disp_comps=[disp_3d[0, z0], disp_3d[1, z0], disp_3d[2, z0]],
        gt_raster=gt_z, pred_sampled_raster=pred_z,
        sdt_pred_slice=sdt_pred_3d[z0] if sdt_pred_3d is not None else None,
        sdt_gt_slice=sdt_gt_3d[z0] if sdt_gt_3d is not None else None,
        hm_pred_slice=hm_pred_3d[z0] if hm_pred_3d is not None else None,
        hm_gt_slice=hm_gt_3d[z0] if hm_gt_3d is not None else None,
        seg_pred_slice=seg_pred_3d[z0] if has_seg else None,
        seg_gt_slice=seg_gt_3d[z0] if has_seg else None,
        extent=z_extent, slice_label=f'z={z0}', xlabel='x', ylabel='y'
    )

    # --- Y-slice (XZ plane) ---
    y_extent = [-W/2, W/2, D/2, -D/2]
    gt_y = rasterize_sparse_to_slice(coords_np, gt_disp_mag, valid_np, y0, (D, W), axis='y')
    pred_y = rasterize_sparse_to_slice(coords_np, pred_sampled_mag, valid_np, y0, (D, W), axis='y')
    plot_slice_pair(
        row_base=2,
        vol_slice=vol_3d[:, y0, :], cond_slice=cond_3d[:, y0, :], extrap_slice=extrap_surf_3d[:, y0, :],
        other_slice=other_wraps_3d[:, y0, :] if other_wraps_3d is not None else None,
        disp_slice=disp_mag_3d[:, y0, :], disp_comps=[disp_3d[0, :, y0, :], disp_3d[1, :, y0, :], disp_3d[2, :, y0, :]],
        gt_raster=gt_y, pred_sampled_raster=pred_y,
        sdt_pred_slice=sdt_pred_3d[:, y0, :] if sdt_pred_3d is not None else None,
        sdt_gt_slice=sdt_gt_3d[:, y0, :] if sdt_gt_3d is not None else None,
        hm_pred_slice=hm_pred_3d[:, y0, :] if hm_pred_3d is not None else None,
        hm_gt_slice=hm_gt_3d[:, y0, :] if hm_gt_3d is not None else None,
        seg_pred_slice=seg_pred_3d[:, y0, :] if has_seg else None,
        seg_gt_slice=seg_gt_3d[:, y0, :] if has_seg else None,
        extent=y_extent, slice_label=f'y={y0}', xlabel='x', ylabel='z'
    )

    # --- X-slice (YZ plane) ---
    x_extent = [-H/2, H/2, D/2, -D/2]
    gt_x = rasterize_sparse_to_slice(coords_np, gt_disp_mag, valid_np, x0, (D, H), axis='x')
    pred_x = rasterize_sparse_to_slice(coords_np, pred_sampled_mag, valid_np, x0, (D, H), axis='x')
    plot_slice_pair(
        row_base=4,
        vol_slice=vol_3d[:, :, x0], cond_slice=cond_3d[:, :, x0], extrap_slice=extrap_surf_3d[:, :, x0],
        other_slice=other_wraps_3d[:, :, x0] if other_wraps_3d is not None else None,
        disp_slice=disp_mag_3d[:, :, x0], disp_comps=[disp_3d[0, :, :, x0], disp_3d[1, :, :, x0], disp_3d[2, :, :, x0]],
        gt_raster=gt_x, pred_sampled_raster=pred_x,
        sdt_pred_slice=sdt_pred_3d[:, :, x0] if sdt_pred_3d is not None else None,
        sdt_gt_slice=sdt_gt_3d[:, :, x0] if sdt_gt_3d is not None else None,
        hm_pred_slice=hm_pred_3d[:, :, x0] if hm_pred_3d is not None else None,
        hm_gt_slice=hm_gt_3d[:, :, x0] if hm_gt_3d is not None else None,
        seg_pred_slice=seg_pred_3d[:, :, x0] if has_seg else None,
        seg_gt_slice=seg_gt_3d[:, :, x0] if has_seg else None,
        extent=x_extent, slice_label=f'x={x0}', xlabel='y', ylabel='z'
    )

    # === Build and display statistics text panel ===
    stats_lines = []
    stats_lines.append("=" * 40)
    stats_lines.append("DISPLACEMENT STATISTICS")
    stats_lines.append("=" * 40)
    stats_lines.append(f"Valid extrap points: {np.sum(valid_np)}")
    stats_lines.append("")

    # Per-component stats table
    stats_lines.append("--- Per-Component (at extrap coords) ---")
    stats_lines.append(f"{'':>6} {'GT mean':>9} {'GT med':>8} {'GT max':>8}")
    stats_lines.append(f"{'':>6} {'Pr mean':>9} {'Pr med':>8} {'Pr max':>8}")
    stats_lines.append("-" * 40)
    for name in component_names:
        gt = gt_comp_stats[name]
        pr = pred_comp_stats[name]
        stats_lines.append(f"{name:>6} {gt['mean']:>9.3f} {gt['median']:>8.3f} {gt['max']:>8.3f}")
        stats_lines.append(f"{'':>6} {pr['mean']:>9.3f} {pr['median']:>8.3f} {pr['max']:>8.3f}")
        stats_lines.append("")

    # Magnitude stats
    stats_lines.append("--- Magnitude (at extrap coords) ---")
    stats_lines.append(f"{'':>6} {'mean':>9} {'median':>8} {'max':>8}")
    stats_lines.append("-" * 40)
    stats_lines.append(f"{'GT':>6} {gt_mag_stats['mean']:>9.3f} {gt_mag_stats['median']:>8.3f} {gt_mag_stats['max']:>8.3f}")
    stats_lines.append(f"{'Pred':>6} {pred_mag_stats['mean']:>9.3f} {pred_mag_stats['median']:>8.3f} {pred_mag_stats['max']:>8.3f}")
    stats_lines.append(f"{'Resid':>6} {residual_stats['mean']:>9.3f} {residual_stats['median']:>8.3f} {residual_stats['max']:>8.3f}")
    stats_lines.append("")

    # Improvement
    stats_lines.append("--- Improvement ---")
    stats_lines.append(f"% Improvement: {pct_improvement:.1f}%")
    stats_lines.append("  (gt_mag - residual) / gt_mag * 100")
    stats_lines.append("")

    # Conditioning point displacement (should be ~0)
    stats_lines.append("--- Conditioning Points ---")
    stats_lines.append(f"N cond points: {cond_disp_stats['n_points']}")
    stats_lines.append(f"Pred disp @ cond (mean): {cond_disp_stats['mean']:.4f}")
    stats_lines.append(f"Pred disp @ cond (max):  {cond_disp_stats['max']:.4f}")
    stats_lines.append("  (should be ~0 if model learns anchoring)")
    stats_lines.append("")
    stats_lines.append("=" * 40)

    # Render text to the panel
    stats_text = "\n".join(stats_lines)
    ax_text.text(0.05, 0.95, stats_text, transform=ax_text.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def train(config_path):
    """Train a displacement field prediction model with optional SDT."""

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Defaults
    config.setdefault('in_channels', 3)  # vol + cond + extrap_surface
    config.setdefault('step_count', 1)  # Required by make_model
    config.setdefault('num_iterations', 250000)
    config.setdefault('log_frequency', 100)
    config.setdefault('ckpt_frequency', 5000)
    config.setdefault('grad_clip', 5)
    config.setdefault('learning_rate', 0.01)
    config.setdefault('weight_decay', 3e-5)
    config.setdefault('batch_size', 4)
    config.setdefault('num_workers', 4)
    config.setdefault('seed', 0)
    config.setdefault('use_sdt', False)
    config.setdefault('lambda_sdt', 1.0)
    config.setdefault('use_heatmap_targets', False)
    config.setdefault('lambda_heatmap', 1.0)
    config.setdefault('use_segmentation', False)
    config.setdefault('lambda_segmentation', 1.0)
    config.setdefault('segmentation_loss', {})
    config.setdefault('supervise_conditioning', False)
    config.setdefault('cond_supervision_weight', 0.1)
    config.setdefault('lambda_cond_disp', 0.0)
    config.setdefault('displacement_loss_type', 'vector_l2')
    config.setdefault('displacement_huber_beta', 5.0)
    config.setdefault('lambda_smooth', 0.0)

    # Build targets dict based on config
    targets = {
        'displacement': {'out_channels': 3, 'activation': 'none'}
    }
    use_sdt = config.get('use_sdt', False)
    if use_sdt:
        targets['sdt'] = {'out_channels': 1, 'activation': 'none'}
    use_heatmap = config.get('use_heatmap_targets', False)
    if use_heatmap:
        targets['heatmap'] = {'out_channels': 1, 'activation': 'none'}
    use_segmentation = config.get('use_segmentation', False)
    if use_segmentation:
        targets['segmentation'] = {'out_channels': 2, 'activation': 'none'}
    config['targets'] = targets

    out_dir = config['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    dynamo_plugin = TorchDynamoPlugin(
            backend="inductor",  # Options: "inductor", "aot_eager", "aot_nvfuser", etc.
            mode="default",      # Options: "default", "reduce-overhead", "max-autotune"
            fullgraph=False,
            dynamic=False,
            use_regional_compilation=False
        )
    

    accelerator = accelerate.Accelerator(
        mixed_precision=config.get('mixed_precision', 'no'),
        gradient_accumulation_steps=config.get('grad_acc_steps', 1),
        dynamo_plugin=dynamo_plugin
    )

    if 'wandb_project' in config and accelerator.is_main_process:
        wandb.init(
            project=config['wandb_project'],
            entity=config.get('wandb_entity', None),
            config=config
        )

    # Setup SDT loss if enabled
    sdt_loss_fn = None
    if use_sdt:
        from vesuvius.models.training.loss.losses import SignedDistanceLoss
        sdt_loss_fn = SignedDistanceLoss(
            beta=config.get('sdt_beta', 1.0),
            eikonal=config.get('sdt_eikonal', True),
            eikonal_weight=config.get('sdt_eikonal_weight', 0.01),
            laplacian=config.get('sdt_laplacian', True),
            laplacian_weight=config.get('sdt_laplacian_weight', 0.01),
            surface_sigma=config.get('sdt_surface_sigma', 3.0),
            reduction='mean',
        )

    lambda_sdt = config.get('lambda_sdt', 1.0)
    lambda_heatmap = config.get('lambda_heatmap', 1.0)
    lambda_segmentation = config.get('lambda_segmentation', 1.0)
    lambda_cond_disp = config.get('lambda_cond_disp', 0.0)
    lambda_smooth = config.get('lambda_smooth', 0.0)
    if config.get('supervise_conditioning', False) and lambda_cond_disp > 0.0:
        raise ValueError(
            "supervise_conditioning=True adds nonzero conditioning displacement targets, "
            "which conflicts with lambda_cond_disp > 0 (zero-displacement penalty on conditioning voxels). "
            "Set lambda_cond_disp to 0 when supervise_conditioning is enabled."
        )
    mask_cond_from_seg_loss = config.get('mask_cond_from_seg_loss', False)
    disp_loss_type = config.get('displacement_loss_type', 'vector_l2')
    disp_huber_beta = config.get('displacement_huber_beta', 5.0)

    # Setup heatmap loss if enabled (BCE + Dice)
    heatmap_loss_fn = None
    if use_heatmap:
        heatmap_loss_fn = DC_and_BCE_loss(
            bce_kwargs={},
            soft_dice_kwargs={'batch_dice': False, 'ddp': False},
            weight_ce=1.0,
            weight_dice=1.0
        )

    # Setup segmentation loss if enabled (MedialSurfaceRecall)
    seg_loss_fn = None
    if use_segmentation:
        seg_loss_cfg = config.get('segmentation_loss', {})
        soft_dice_kwargs = {
            'batch_dice': seg_loss_cfg.get('batch_dice', False),
            'smooth': seg_loss_cfg.get('smooth', 1e-5),
            'do_bg': seg_loss_cfg.get('do_bg', False),
            'ddp': seg_loss_cfg.get('ddp', False),
        }
        if 'soft_dice_kwargs' in seg_loss_cfg:
            soft_dice_kwargs.update(seg_loss_cfg['soft_dice_kwargs'])
        seg_loss_fn = DC_SkelREC_and_CE_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            soft_skelrec_kwargs={
                'batch_dice': soft_dice_kwargs.get('batch_dice'),
                'smooth': soft_dice_kwargs.get('smooth'),
                'do_bg': soft_dice_kwargs.get('do_bg'),
                'ddp': soft_dice_kwargs.get('ddp'),
            },
            ce_kwargs=seg_loss_cfg.get('ce_kwargs', {}),
            weight_ce=seg_loss_cfg.get('weight_ce', 1),
            weight_dice=seg_loss_cfg.get('weight_dice', 1),
            weight_srec=seg_loss_cfg.get('weight_srec', 1),
            ignore_label=seg_loss_cfg.get('ignore_label', None),
        )

    def make_generator(offset=0):
        gen = torch.Generator()
        gen.manual_seed(config['seed'] + accelerator.process_index * 1000 + offset)
        return gen

    # Train with augmentation, val without
    train_dataset = EdtSegDataset(config, apply_augmentation=True)
    val_dataset = EdtSegDataset(config, apply_augmentation=False)

    # Train/val split by indices
    num_patches = len(train_dataset)
    num_val = max(1, int(num_patches * config.get('val_fraction', 0.1)))
    num_train = num_patches - num_val

    indices = torch.randperm(num_patches, generator=torch.Generator().manual_seed(config['seed'])).tolist()
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        worker_init_fn=seed_worker,
        generator=make_generator(0),
        drop_last=True,
        collate_fn=collate_with_padding,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=1,
        worker_init_fn=seed_worker,
        generator=make_generator(1),
        collate_fn=collate_with_padding,
    )

    model = make_model(config)

    if config.get('compile_model', True):
        model = torch.compile(model)
        if accelerator.is_main_process:
            accelerator.print("Model compiled with torch.compile")

    optimizer = create_optimizer({
        'name': 'adamw',
        'learning_rate': config.get('learning_rate', 1e-3),
        'weight_decay': config.get('weight_decay', 1e-4),
    }, model)

    lr_scheduler = get_scheduler(
        scheduler_type='diffusers_cosine_warmup',
        optimizer=optimizer,
        initial_lr=config.get('learning_rate', 1e-3),
        max_steps=config['num_iterations'],
        warmup_steps=config.get('warmup_steps', 5000),
    )

    start_iteration = 0
    if 'load_ckpt' in config:
        print(f'Loading checkpoint {config["load_ckpt"]}')
        ckpt = torch.load(config['load_ckpt'], map_location='cpu', weights_only=False)
        state_dict = ckpt['model']
        # Handle compiled model state dict
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            model_keys = set(model.state_dict().keys())
            if not any(k.startswith('_orig_mod.') for k in model_keys):
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                print('Stripped _orig_mod. prefix from checkpoint state dict')
        model.load_state_dict(state_dict)

        if not config.get('load_weights_only', False):
            start_iteration = ckpt.get('step', 0)
            # Load optimizer state if optimizer type matches (SGD vs Adam check via betas)
            ckpt_optim_type = type(ckpt['optimizer']['param_groups'][0].get('betas', None))
            curr_optim_type = type(optimizer.param_groups[0].get('betas', None))
            if ckpt_optim_type == curr_optim_type:
                optimizer.load_state_dict(ckpt['optimizer'])
                print('Loaded optimizer state (momentum preserved)')
            else:
                print('Skipping optimizer state load (optimizer type changed)')

    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        accelerator.print("\n=== Displacement Field Training Configuration ===")
        accelerator.print(f"Input channels: {config['in_channels']}")
        accelerator.print(f"Displacement loss: {disp_loss_type}")
        output_str = "Output: displacement (3ch)"
        if use_sdt:
            output_str += " + SDT (1ch)"
        if use_heatmap:
            output_str += " + heatmap (1ch)"
        if use_segmentation:
            output_str += " + segmentation (2ch)"
        accelerator.print(output_str)
        if use_sdt:
            accelerator.print(f"Lambda SDT: {lambda_sdt}")
        if use_heatmap:
            accelerator.print(f"Lambda heatmap: {lambda_heatmap}")
        if use_segmentation:
            accelerator.print(f"Lambda segmentation: {lambda_segmentation}")
        if lambda_cond_disp > 0.0:
            accelerator.print(f"Lambda cond disp: {lambda_cond_disp}")
        accelerator.print(f"Supervise conditioning: {config.get('supervise_conditioning', False)}")
        if config.get('supervise_conditioning', False):
            accelerator.print(f"Cond supervision weight: {config.get('cond_supervision_weight', 0.1)}")
        accelerator.print(f"Optimizer: AdamW (lr={config.get('learning_rate', 1e-3)})")
        accelerator.print(f"Scheduler: diffusers_cosine_warmup (warmup={config.get('warmup_steps', 5000)})")
        accelerator.print(f"Train samples: {num_train}, Val samples: {num_val}")
        accelerator.print("=================================================\n")

    if config['verbose']:
            print("creating iterators...")
    val_iterator = iter(val_dataloader)
    train_iterator = iter(train_dataloader)
    grad_clip = config['grad_clip']

    progress_bar = tqdm(
        total=config['num_iterations'],
        initial=start_iteration,
        disable=not accelerator.is_local_main_process
    )

    for iteration in range(start_iteration, config['num_iterations']):
        if config['verbose']:
            print(f"starting iteration {iteration}")
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)

        if config['verbose']:
            print(f"got batch, keys: {batch.keys()}")

        inputs, extrap_coords, gt_displacement, valid_mask, point_weights, sdt_target, heatmap_target, seg_target, seg_skel = prepare_batch(
            batch, use_sdt, use_heatmap, use_segmentation
        )

        wandb_log = {}

        with accelerator.accumulate(model):
            # Forward pass
            output = model(inputs)
            disp_pred = output['displacement']  # [B, 3, D, H, W]

            # Displacement loss
            surf_loss = surface_sampled_loss(disp_pred, extrap_coords, gt_displacement, valid_mask,
                                             loss_type=disp_loss_type, beta=disp_huber_beta,
                                             sample_weights=point_weights)
            total_loss = surf_loss

            wandb_log['surf_loss'] = surf_loss.detach().item()

            # Smoothness loss on displacement field
            if lambda_smooth > 0:
                smooth_loss = smoothness_loss(disp_pred)
                weighted_smooth_loss = lambda_smooth * smooth_loss
                total_loss = total_loss + weighted_smooth_loss
                wandb_log['smooth_loss'] = weighted_smooth_loss.detach().item()

            # Optional SDT loss
            if use_sdt:
                sdt_pred = output['sdt']  # [B, 1, D, H, W]
                sdt_loss = sdt_loss_fn(sdt_pred, sdt_target)
                weighted_sdt_loss = lambda_sdt * sdt_loss
                total_loss = total_loss + weighted_sdt_loss
                wandb_log['sdt_loss'] = weighted_sdt_loss.detach().item()

            # Optional heatmap loss (BCE + Dice)
            heatmap_pred = None
            if use_heatmap:
                heatmap_pred = output['heatmap']  # [B, 1, D, H, W]
                heatmap_target_binary = (heatmap_target > 0.5).float()
                heatmap_loss = heatmap_loss_fn(heatmap_pred, heatmap_target_binary)
                weighted_heatmap_loss = lambda_heatmap * heatmap_loss
                total_loss = total_loss + weighted_heatmap_loss
                wandb_log['heatmap_loss'] = weighted_heatmap_loss.detach().item()

            # Optional segmentation loss (MedialSurfaceRecall)
            if use_segmentation:
                seg_pred = output['segmentation']  # [B, 2, D, H, W]

                # Optionally mask out conditioning region from seg loss
                seg_loss_mask = None
                if mask_cond_from_seg_loss:
                    cond_mask_seg = (inputs[:, 1:2] > 0.5).float()  # [B, 1, D, H, W]
                    seg_loss_mask = (cond_mask_seg < 0.5).float()   # 1 everywhere except cond

                seg_loss = seg_loss_fn(seg_pred, seg_target.long(), seg_skel.long(), loss_mask=seg_loss_mask)
                weighted_seg_loss = lambda_segmentation * seg_loss
                total_loss = total_loss + weighted_seg_loss
                wandb_log['seg_loss'] = weighted_seg_loss.detach().item()

            if lambda_cond_disp > 0.0:
                cond_mask = (inputs[:, 1:2] > 0.5).float()
                disp_mag_sq = (disp_pred ** 2).sum(dim=1, keepdim=True)
                cond_loss = (disp_mag_sq * cond_mask).sum() / cond_mask.sum().clamp(min=1.0)
                weighted_cond_loss = lambda_cond_disp * cond_loss
                total_loss = total_loss + weighted_cond_loss
                wandb_log['cond_disp_loss'] = weighted_cond_loss.detach().item()

            if torch.isnan(total_loss).any():
                raise ValueError('loss is NaN')

            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        wandb_log['loss'] = total_loss.detach().item()
        wandb_log['lr'] = optimizer.param_groups[0]['lr']

        postfix = {
            'loss': f"{wandb_log['loss']:.4f}",
            'surf': f"{wandb_log['surf_loss']:.4f}",
        }
        if lambda_smooth > 0:
            postfix['smooth'] = f"{wandb_log['smooth_loss']:.4f}"
        if use_sdt:
            postfix['sdt'] = f"{wandb_log['sdt_loss']:.4f}"
        if use_heatmap:
            postfix['hm'] = f"{wandb_log['heatmap_loss']:.4f}"
        if use_segmentation:
            postfix['seg'] = f"{wandb_log['seg_loss']:.4f}"
        if lambda_cond_disp > 0.0:
            postfix['cond'] = f"{wandb_log['cond_disp_loss']:.4f}"
        progress_bar.set_postfix(postfix)
        progress_bar.update(1)

        if iteration % config['log_frequency'] == 0 and accelerator.is_main_process:
            with torch.no_grad():
                model.eval()

                try:
                    val_batch = next(val_iterator)
                except StopIteration:
                    val_iterator = iter(val_dataloader)
                    val_batch = next(val_iterator)

                val_inputs, val_extrap_coords, val_gt_displacement, val_valid_mask, val_point_weights, val_sdt_target, val_heatmap_target, val_seg_target, val_seg_skel = prepare_batch(
                    val_batch, use_sdt, use_heatmap, use_segmentation
                )

                val_output = model(val_inputs)
                val_disp_pred = val_output['displacement']

                val_surf_loss = surface_sampled_loss(val_disp_pred, val_extrap_coords, val_gt_displacement, val_valid_mask,
                                                     loss_type=disp_loss_type, beta=disp_huber_beta,
                                                     sample_weights=val_point_weights)
                val_total_loss = val_surf_loss

                wandb_log['val_surf_loss'] = val_surf_loss.item()

                if lambda_smooth > 0:
                    val_smooth_loss = smoothness_loss(val_disp_pred)
                    val_weighted_smooth_loss = lambda_smooth * val_smooth_loss
                    val_total_loss = val_total_loss + val_weighted_smooth_loss
                    wandb_log['val_smooth_loss'] = val_weighted_smooth_loss.item()

                val_sdt_pred = None
                if use_sdt:
                    val_sdt_pred = val_output['sdt']
                    val_sdt_loss = sdt_loss_fn(val_sdt_pred, val_sdt_target)
                    val_weighted_sdt_loss = lambda_sdt * val_sdt_loss
                    val_total_loss = val_total_loss + val_weighted_sdt_loss
                    wandb_log['val_sdt_loss'] = val_weighted_sdt_loss.item()

                val_heatmap_pred = None
                if use_heatmap:
                    val_heatmap_pred = val_output['heatmap']
                    val_heatmap_target_binary = (val_heatmap_target > 0.5).float()
                    val_heatmap_loss = heatmap_loss_fn(val_heatmap_pred, val_heatmap_target_binary)
                    val_weighted_heatmap_loss = lambda_heatmap * val_heatmap_loss
                    val_total_loss = val_total_loss + val_weighted_heatmap_loss
                    wandb_log['val_heatmap_loss'] = val_weighted_heatmap_loss.item()

                if use_segmentation:
                    val_seg_pred = val_output['segmentation']
                    val_seg_loss_mask = None
                    if mask_cond_from_seg_loss:
                        val_cond_mask_seg = (val_inputs[:, 1:2] > 0.5).float()
                        val_seg_loss_mask = (val_cond_mask_seg < 0.5).float()
                    val_seg_loss = seg_loss_fn(
                        val_seg_pred, val_seg_target.long(), val_seg_skel.long(), loss_mask=val_seg_loss_mask
                    )
                    val_weighted_seg_loss = lambda_segmentation * val_seg_loss
                    val_total_loss = val_total_loss + val_weighted_seg_loss
                    wandb_log['val_seg_loss'] = val_weighted_seg_loss.item()

                if lambda_cond_disp > 0.0:
                    val_cond_mask = (val_inputs[:, 1:2] > 0.5).float()
                    val_disp_mag_sq = (val_disp_pred ** 2).sum(dim=1, keepdim=True)
                    val_cond_loss = (val_disp_mag_sq * val_cond_mask).sum() / val_cond_mask.sum().clamp(min=1.0)
                    val_weighted_cond_loss = lambda_cond_disp * val_cond_loss
                    val_total_loss = val_total_loss + val_weighted_cond_loss
                    wandb_log['val_cond_disp_loss'] = val_weighted_cond_loss.item()

                wandb_log['val_loss'] = val_total_loss.item()

                # Create visualization
                train_img_path = f'{out_dir}/{iteration:06}_train.png'
                val_img_path = f'{out_dir}/{iteration:06}_val.png'

                train_sdt_pred = output.get('sdt') if use_sdt else None
                train_heatmap_pred = heatmap_pred if use_heatmap else None
                train_seg_pred = output.get('segmentation') if use_segmentation else None
                make_visualization(
                    inputs, disp_pred, extrap_coords, gt_displacement, valid_mask,
                    sdt_pred=train_sdt_pred, sdt_target=sdt_target,
                    heatmap_pred=train_heatmap_pred, heatmap_target=heatmap_target,
                    seg_pred=train_seg_pred, seg_target=seg_target if use_segmentation else None,
                    save_path=train_img_path
                )
                make_visualization(
                    val_inputs, val_disp_pred, val_extrap_coords, val_gt_displacement, val_valid_mask,
                    sdt_pred=val_sdt_pred, sdt_target=val_sdt_target,
                    heatmap_pred=val_heatmap_pred, heatmap_target=val_heatmap_target,
                    seg_pred=val_output.get('segmentation') if use_segmentation else None,
                    seg_target=val_seg_target if use_segmentation else None,
                    save_path=val_img_path
                )

                if wandb.run is not None:
                    wandb_log['train_image'] = wandb.Image(train_img_path)
                    wandb_log['val_image'] = wandb.Image(val_img_path)

                model.train()

        if iteration % config['ckpt_frequency'] == 0 and accelerator.is_main_process:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'step': iteration,
            }, f'{out_dir}/ckpt_{iteration:06}.pth')

        if wandb.run is not None and accelerator.is_main_process:
            wandb.log(wandb_log)

    progress_bar.close()

    if accelerator.is_main_process:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'config': config,
            'step': config['num_iterations'],
        }, f'{out_dir}/ckpt_final.pth')


if __name__ == '__main__':
    train()
