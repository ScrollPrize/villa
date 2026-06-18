"""Visualization helpers for row/col conditioned trainer."""

import numpy as np
import torch
import torch.nn.functional as F


def _tensor_to_numpy(tensor):
    """Convert torch tensor to numpy, upcasting bfloat16 for compatibility."""
    tensor_cpu = tensor.detach().cpu()
    if tensor_cpu.dtype == torch.bfloat16:
        tensor_cpu = tensor_cpu.float()
    return tensor_cpu.numpy()


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
                       save_path=None,
                       **unused_trace_kwargs):
    """Create and save PNG visualization of Z, Y, and X slices."""
    import matplotlib.pyplot as plt

    b = 0
    D, H, W = inputs.shape[2], inputs.shape[3], inputs.shape[4]

    # Precompute 3D arrays
    vol_3d = _tensor_to_numpy(inputs[b, 0])
    cond_3d = _tensor_to_numpy(inputs[b, 1])
    extrap_surf_3d = _tensor_to_numpy(inputs[b, 2])
    other_wraps_3d = _tensor_to_numpy(inputs[b, 3]) if inputs.shape[1] > 3 else None

    # Displacement: [3, D, H, W] where components are (dz, dy, dx)
    disp_3d = _tensor_to_numpy(disp_pred[b])
    disp_mag_3d = np.linalg.norm(disp_3d, axis=0)

    # GT displacement processing
    gt_disp_np = _tensor_to_numpy(gt_displacement[b])  # (N, 3)
    gt_disp_mag = np.linalg.norm(gt_disp_np, axis=-1)
    coords_np = _tensor_to_numpy(extrap_coords[b])  # (N, 3) - z, y, x
    valid_np = _tensor_to_numpy(valid_mask[b]).astype(bool)

    # Sample predicted displacement vectors at extrap coords via trilinear interpolation.
    # This matches the training loss sampling path and reduces metric jitter from rounding.
    coords_t = extrap_coords[b:b + 1].to(device=disp_pred.device, dtype=disp_pred.dtype)  # [1, N, 3]
    coords_normalized = coords_t.clone()
    d_denom = max(D - 1, 1)
    h_denom = max(H - 1, 1)
    w_denom = max(W - 1, 1)
    coords_normalized[..., 0] = 2 * coords_normalized[..., 0] / d_denom - 1  # z
    coords_normalized[..., 1] = 2 * coords_normalized[..., 1] / h_denom - 1  # y
    coords_normalized[..., 2] = 2 * coords_normalized[..., 2] / w_denom - 1  # x
    grid = coords_normalized[..., [2, 1, 0]].view(1, -1, 1, 1, 3)  # grid_sample expects x,y,z
    sampled_pred = F.grid_sample(disp_pred[b:b + 1], grid, mode='bilinear', align_corners=True)
    pred_sampled_vectors = _tensor_to_numpy(sampled_pred.view(1, 3, -1).permute(0, 2, 1)[0])  # [N, 3]
    pred_sampled_mag = np.linalg.norm(pred_sampled_vectors, axis=-1)

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

    # Robust % improvement:
    #   100 * (1 - residual / gt_mag)
    # evaluated only on non-trivial gt magnitude and clipped to avoid huge unstable tails.
    improvement_gt_floor = 0.10
    improvement_clip = 100.0
    meaningful_mask = valid_gt_mag > improvement_gt_floor
    if np.sum(meaningful_mask) > 0:
        meaningful_gt = valid_gt_mag[meaningful_mask]
        meaningful_resid = residual_mag[meaningful_mask]
        raw_improvement_per_point = (meaningful_gt - meaningful_resid) / meaningful_gt * 100
        raw_improvement_per_point = raw_improvement_per_point[np.isfinite(raw_improvement_per_point)]
        improvement_per_point = np.clip(raw_improvement_per_point, -improvement_clip, improvement_clip)
        pct_improvement = np.median(improvement_per_point) if len(improvement_per_point) > 0 else 0.0
        n_improvement_points = int(len(improvement_per_point))
    else:
        pct_improvement = 0.0
        n_improvement_points = 0

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
    sdt_pred_3d = _tensor_to_numpy(sdt_pred[b, 0]) if sdt_pred is not None else None
    sdt_gt_3d = _tensor_to_numpy(sdt_target[b, 0]) if sdt_target is not None else None
    sdt_vmax = max(np.abs(sdt_pred_3d).max(), np.abs(sdt_gt_3d).max()) if sdt_pred_3d is not None else 1.0
    hm_pred_3d = _tensor_to_numpy(torch.sigmoid(heatmap_pred[b, 0])) if heatmap_pred is not None else None
    hm_gt_3d = _tensor_to_numpy(heatmap_target[b, 0]) if heatmap_target is not None else None

    # Segmentation: pred is [B, 2, D, H, W], target is [B, 1, D, H, W]
    has_seg = seg_pred is not None and seg_target is not None
    seg_pred_3d = _tensor_to_numpy(seg_pred[b].argmax(dim=0)) if seg_pred is not None else None  # [D, H, W]
    seg_gt_3d = _tensor_to_numpy(seg_target[b, 0]) if seg_target is not None else None  # [D, H, W]

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
    stats_lines.append(f"  gt_mag > {improvement_gt_floor:.2f}, clipped to +/-{improvement_clip:.0f}% (n={n_improvement_points})")
    stats_lines.append("  100 * (1 - residual / gt_mag)")
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


def _make_dense_triplet_visualization(
    inputs,
    disp_pred,
    dense_gt_displacement,
    dense_loss_weight=None,
    triplet_channel_order=None,
    save_path=None,
):
    """Triplet-mode dense visualization with swap-aware Channel A/B pairing."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec

    b = 0
    D, H, W = inputs.shape[2], inputs.shape[3], inputs.shape[4]

    vol_3d = _tensor_to_numpy(inputs[b, 0])
    cond_3d = _tensor_to_numpy(inputs[b, 1])
    pred_3d = _tensor_to_numpy(disp_pred[b])
    gt_3d = _tensor_to_numpy(dense_gt_displacement[b])

    if pred_3d.shape[0] != 6 or gt_3d.shape[0] != 6:
        raise ValueError(
            "Triplet visualization expects 6-channel displacement fields "
            f"(got pred={pred_3d.shape[0]}, gt={gt_3d.shape[0]})"
        )

    channel_order = np.array([0, 1], dtype=np.int64)
    if triplet_channel_order is not None:
        if isinstance(triplet_channel_order, torch.Tensor):
            order_arr = _tensor_to_numpy(triplet_channel_order)
        else:
            order_arr = np.asarray(triplet_channel_order)
        if order_arr.ndim == 2:
            if b < order_arr.shape[0]:
                order_arr = order_arr[b]
        if order_arr.ndim == 1 and order_arr.size == 2:
            candidate = order_arr.astype(np.int64, copy=False)
            if set(candidate.tolist()) == {0, 1}:
                channel_order = candidate

    slot_a = int(np.flatnonzero(channel_order == 0)[0])
    slot_b = int(np.flatnonzero(channel_order == 1)[0])

    def _branch_mag(field_3d, slot_idx):
        start = int(slot_idx) * 3
        return np.linalg.norm(field_3d[start:start + 3], axis=0)

    pred_channel_a_mag_3d = _branch_mag(pred_3d, slot_a)
    pred_channel_b_mag_3d = _branch_mag(pred_3d, slot_b)
    gt_channel_a_mag_3d = _branch_mag(gt_3d, slot_a)
    gt_channel_b_mag_3d = _branch_mag(gt_3d, slot_b)

    cond_mask_3d = cond_3d > 0.5
    other_wraps_union_3d = ((gt_channel_a_mag_3d < 0.5) | (gt_channel_b_mag_3d < 0.5))

    if dense_loss_weight is None:
        weight_3d = np.ones((D, H, W), dtype=np.float32)
    else:
        w = _tensor_to_numpy(dense_loss_weight[b])
        weight_3d = w[0] if w.ndim == 4 else w
    supervised_mask = weight_3d > 0
    cond_supervised = cond_mask_3d & supervised_mask

    zero_disp_tol = 0.5

    def _zero_disp_overlap_stats(pred_mag_3d: np.ndarray, gt_mag_3d: np.ndarray, mask_3d: np.ndarray):
        mask = np.asarray(mask_3d, dtype=bool)
        gt_zero = (gt_mag_3d <= zero_disp_tol) & mask
        pred_zero = (pred_mag_3d <= zero_disp_tol) & mask
        overlap = gt_zero & pred_zero
        gt_count = int(gt_zero.sum())
        pred_count = int(pred_zero.sum())
        overlap_count = int(overlap.sum())
        overlap_pct_gt = (100.0 * overlap_count / gt_count) if gt_count > 0 else 0.0
        return {
            "gt_count": gt_count,
            "pred_count": pred_count,
            "overlap_count": overlap_count,
            "overlap_pct_gt": float(overlap_pct_gt),
        }

    channel_a_zero_stats = _zero_disp_overlap_stats(
        pred_channel_a_mag_3d,
        gt_channel_a_mag_3d,
        supervised_mask,
    )
    channel_b_zero_stats = _zero_disp_overlap_stats(
        pred_channel_b_mag_3d,
        gt_channel_b_mag_3d,
        supervised_mask,
    )

    def _build_magnitude_norm(pred_vals, gt_vals):
        """Build a robust norm that keeps contrast when magnitudes cluster."""
        merged = np.concatenate([pred_vals.reshape(-1), gt_vals.reshape(-1)])
        merged = merged[np.isfinite(merged)]
        merged = merged[merged >= 0]
        if merged.size == 0:
            return mcolors.Normalize(vmin=0.0, vmax=1.0, clip=True), 0.0, 1.0, "linear"

        p01, p50, p99 = np.percentile(merged, [1, 50, 99])
        vmin = max(0.0, float(p01))
        vmax = float(max(p99, vmin + 1e-6))

        # If the robust window is nearly flat, expand around median for visibility.
        if (vmax - vmin) < max(1e-6, 1e-3 * max(float(p50), 1.0)):
            half_span = max(0.05 * max(float(p50), 1.0), 1e-3)
            vmin = max(0.0, float(p50 - half_span))
            vmax = float(p50 + half_span)
            return mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True), vmin, vmax, "linear-local"

        # Power-law scaling increases contrast in the low/mid range.
        return mcolors.PowerNorm(gamma=0.65, vmin=vmin, vmax=vmax, clip=True), vmin, vmax, "power"

    channel_a_vals_pred = pred_channel_a_mag_3d[supervised_mask] if np.any(supervised_mask) else pred_channel_a_mag_3d.reshape(-1)
    channel_a_vals_gt = gt_channel_a_mag_3d[supervised_mask] if np.any(supervised_mask) else gt_channel_a_mag_3d.reshape(-1)
    channel_b_vals_pred = pred_channel_b_mag_3d[supervised_mask] if np.any(supervised_mask) else pred_channel_b_mag_3d.reshape(-1)
    channel_b_vals_gt = gt_channel_b_mag_3d[supervised_mask] if np.any(supervised_mask) else gt_channel_b_mag_3d.reshape(-1)

    channel_a_norm, channel_a_vmin, channel_a_vmax, channel_a_norm_name = _build_magnitude_norm(
        channel_a_vals_pred, channel_a_vals_gt
    )
    channel_b_norm, channel_b_vmin, channel_b_vmax, channel_b_norm_name = _build_magnitude_norm(
        channel_b_vals_pred, channel_b_vals_gt
    )

    z0, y0, x0 = D // 2, H // 2, W // 2
    slices = [
        ("z", z0, f"z={z0}", "x", "y", [-W / 2, W / 2, H / 2, -H / 2]),
        ("y", y0, f"y={y0}", "x", "z", [-W / 2, W / 2, D / 2, -D / 2]),
        ("x", x0, f"x={x0}", "y", "z", [-H / 2, H / 2, D / 2, -D / 2]),
    ]

    def _slice(arr, axis, idx):
        if axis == "z":
            return arr[idx]
        if axis == "y":
            return arr[:, idx, :]
        return arr[:, :, idx]

    # Show only supervised band voxels in displacement panels.
    gt_channel_a_band = np.where(supervised_mask, gt_channel_a_mag_3d, np.nan)
    pred_channel_a_band = np.where(supervised_mask, pred_channel_a_mag_3d, np.nan)
    gt_channel_b_band = np.where(supervised_mask, gt_channel_b_mag_3d, np.nan)
    pred_channel_b_band = np.where(supervised_mask, pred_channel_b_mag_3d, np.nan)

    disp_cmap = plt.cm.cividis.copy()
    disp_cmap.set_bad(color="black")

    n_cols = 5
    fig = plt.figure(figsize=(4 * n_cols + 4, 14))
    gs = GridSpec(3, n_cols + 1, figure=fig, width_ratios=[1] * n_cols + [1.2], wspace=0.3)
    axes = np.empty((3, n_cols), dtype=object)
    for r in range(3):
        for c in range(n_cols):
            axes[r, c] = fig.add_subplot(gs[r, c])
    ax_text = fig.add_subplot(gs[:, n_cols])
    ax_text.axis("off")

    for row, (axis, idx, label, xlabel, ylabel, extent) in enumerate(slices):
        vol_slice = _slice(vol_3d, axis, idx)
        cond_slice = _slice(cond_mask_3d, axis, idx)
        other_slice = _slice(other_wraps_union_3d, axis, idx)
        gt_channel_a_slice = _slice(gt_channel_a_band, axis, idx)
        pred_channel_a_slice = _slice(pred_channel_a_band, axis, idx)
        gt_channel_b_slice = _slice(gt_channel_b_band, axis, idx)
        pred_channel_b_slice = _slice(pred_channel_b_band, axis, idx)

        vol_norm = (vol_slice - vol_slice.min()) / (vol_slice.max() - vol_slice.min() + 1e-8)
        overlay = np.stack([vol_norm, vol_norm, vol_norm], axis=-1)

        other_color = np.array([0.6, 1.0, 0.6], dtype=np.float32)   # light green
        cond_color = np.array([1.0, 0.0, 1.0], dtype=np.float32)    # magenta
        overlay[other_slice] = 0.40 * overlay[other_slice] + 0.60 * other_color
        overlay[cond_slice] = 0.20 * overlay[cond_slice] + 0.80 * cond_color

        axes[row, 0].imshow(overlay, extent=extent)
        axes[row, 0].set_title(f"Overlay ({label})")
        axes[row, 0].set_ylabel(ylabel)

        axes[row, 1].imshow(gt_channel_a_slice, cmap=disp_cmap, norm=channel_a_norm, extent=extent)
        axes[row, 1].set_title("GT Channel A |disp| (band)")
        axes[row, 1].set_yticks([])

        axes[row, 2].imshow(pred_channel_a_slice, cmap=disp_cmap, norm=channel_a_norm, extent=extent)
        axes[row, 2].set_title(
            "Pred Channel A |disp| (band)\n"
            f"0-disp overlap vs GT: {channel_a_zero_stats['overlap_pct_gt']:.1f}%"
        )
        axes[row, 2].set_yticks([])

        axes[row, 3].imshow(gt_channel_b_slice, cmap=disp_cmap, norm=channel_b_norm, extent=extent)
        axes[row, 3].set_title("GT Channel B |disp| (band)")
        axes[row, 3].set_yticks([])

        axes[row, 4].imshow(pred_channel_b_slice, cmap=disp_cmap, norm=channel_b_norm, extent=extent)
        axes[row, 4].set_title(
            "Pred Channel B |disp| (band)\n"
            f"0-disp overlap vs GT: {channel_b_zero_stats['overlap_pct_gt']:.1f}%"
        )
        axes[row, 4].set_yticks([])

        for c in range(n_cols):
            axes[row, c].set_xlabel(xlabel)

    band_channel_a_gt_vals = gt_channel_a_mag_3d[supervised_mask]
    band_channel_a_pred_vals = pred_channel_a_mag_3d[supervised_mask]
    band_channel_b_gt_vals = gt_channel_b_mag_3d[supervised_mask]
    band_channel_b_pred_vals = pred_channel_b_mag_3d[supervised_mask]

    stats_lines = [
        "=" * 42,
        "TRIPLET DENSE DISPLACEMENT",
        "=" * 42,
        f"Supervised voxels: {int(supervised_mask.sum())}",
        f"Conditioning voxels: {int(cond_mask_3d.sum())}",
        f"Cond&supervised voxels: {int(cond_supervised.sum())}",
        f"Slot mapping: slot0->Channel {'A' if int(channel_order[0]) == 0 else 'B'}, "
        f"slot1->Channel {'A' if int(channel_order[1]) == 0 else 'B'}",
        "",
        "--- Band (supervised) -> Channel A ---",
        f"GT   mean |disp|: {float(band_channel_a_gt_vals.mean()) if band_channel_a_gt_vals.size > 0 else 0.0:.4f}",
        f"Pred mean |disp|: {float(band_channel_a_pred_vals.mean()) if band_channel_a_pred_vals.size > 0 else 0.0:.4f}",
        (
            f"0-disp overlap vs GT (<= {zero_disp_tol:.2f} vx): "
            f"{channel_a_zero_stats['overlap_pct_gt']:.1f}% "
            f"[{channel_a_zero_stats['overlap_count']}/{channel_a_zero_stats['gt_count']}]"
        ),
        f"Display scale: {channel_a_norm_name}, [{channel_a_vmin:.3f}, {channel_a_vmax:.3f}]",
        "",
        "--- Band (supervised) -> Channel B ---",
        f"GT   mean |disp|: {float(band_channel_b_gt_vals.mean()) if band_channel_b_gt_vals.size > 0 else 0.0:.4f}",
        f"Pred mean |disp|: {float(band_channel_b_pred_vals.mean()) if band_channel_b_pred_vals.size > 0 else 0.0:.4f}",
        (
            f"0-disp overlap vs GT (<= {zero_disp_tol:.2f} vx): "
            f"{channel_b_zero_stats['overlap_pct_gt']:.1f}% "
            f"[{channel_b_zero_stats['overlap_count']}/{channel_b_zero_stats['gt_count']}]"
        ),
        f"Display scale: {channel_b_norm_name}, [{channel_b_vmin:.3f}, {channel_b_vmax:.3f}]",
        "",
        "Overlay colors:",
        "  Neighbor wraps (A/B) = light green",
        "  Conditioning = magenta",
        "=" * 42,
    ]
    ax_text.text(
        0.05, 0.95, "\n".join(stats_lines),
        transform=ax_text.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
    )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def make_trace_visualization(vis_data, save_path):
    """Create a dense trace-only visualization from a trainer payload."""
    make_dense_visualization(
        vis_data['inputs'], None, None, None,
        velocity_dir_pred=vis_data.get('velocity_dir_pred'),
        velocity_dir_target=vis_data.get('velocity_dir_target'),
        velocity_loss_weight=vis_data.get('velocity_loss_weight'),
        trace_loss_weight=vis_data.get('trace_loss_weight'),
        trace_validity_pred=vis_data.get('trace_validity_pred'),
        trace_validity_target=vis_data.get('trace_validity_target'),
        trace_validity_weight=vis_data.get('trace_validity_weight'),
        surface_attract_pred=vis_data.get('surface_attract_pred'),
        surface_attract_target=vis_data.get('surface_attract_target'),
        surface_attract_weight=vis_data.get('surface_attract_weight'),
        save_path=save_path,
    )


def make_copy_neighbor_visualization(vis_data, save_path):
    """Create a copy-neighbor trace/ODE visualization from a trainer payload."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    inputs = vis_data["inputs"]
    b = 0
    _, _, D, H, W = inputs.shape

    vol = _tensor_to_numpy(inputs[b, 0])
    cond = _tensor_to_numpy(inputs[b, 1])
    side_hint = _tensor_to_numpy(inputs[b, 2:5])
    target_seg = _tensor_to_numpy(vis_data["target_seg"][b]) if vis_data.get("target_seg") is not None else None
    domain = _tensor_to_numpy(vis_data["domain"][b]) if vis_data.get("domain") is not None else None
    if domain is None and vis_data.get("progress_phi_weight") is not None:
        domain = _tensor_to_numpy(vis_data["progress_phi_weight"][b, 0])

    progress_target = (
        _tensor_to_numpy(vis_data["progress_phi_target"][b, 0])
        if vis_data.get("progress_phi_target") is not None else None
    )
    progress_pred = (
        _tensor_to_numpy(vis_data["progress_phi_pred"][b, 0].float())
        if vis_data.get("progress_phi_pred") is not None else None
    )
    stop_target = _tensor_to_numpy(vis_data["stop_target"][b, 0]) if vis_data.get("stop_target") is not None else None
    stop_pred = (
        _tensor_to_numpy(torch.sigmoid(vis_data["stop_pred"][b, 0].float()))
        if vis_data.get("stop_pred") is not None else None
    )
    target_edt = _tensor_to_numpy(vis_data["target_edt"][b, 0]) if vis_data.get("target_edt") is not None else None

    velocity_align = None
    if vis_data.get("velocity_dir_pred") is not None and vis_data.get("velocity_dir_target") is not None:
        pred_vel = F.normalize(vis_data["velocity_dir_pred"][b:b + 1].float(), dim=1, eps=1e-6)
        target_vel = F.normalize(vis_data["velocity_dir_target"][b:b + 1].float(), dim=1, eps=1e-6)
        velocity_align = _tensor_to_numpy((pred_vel * target_vel).sum(dim=1)[0].clamp(-1.0, 1.0))
        if vis_data.get("velocity_loss_weight") is not None:
            vel_mask = _tensor_to_numpy(vis_data["velocity_loss_weight"][b, 0]) > 0.0
            velocity_align = np.where(vel_mask, velocity_align, np.nan)

    attract_err = None
    if vis_data.get("surface_attract_pred") is not None and vis_data.get("surface_attract_target") is not None:
        pred = _tensor_to_numpy(vis_data["surface_attract_pred"][b])
        target = _tensor_to_numpy(vis_data["surface_attract_target"][b])
        attract_err = np.linalg.norm(pred - target, axis=0)
        if vis_data.get("surface_attract_weight") is not None:
            attract_mask = _tensor_to_numpy(vis_data["surface_attract_weight"][b, 0]) > 0.0
            attract_err = np.where(attract_mask, attract_err, np.nan)

    seeds = _tensor_to_numpy(vis_data["endpoint_seed_points"][b]) if vis_data.get("endpoint_seed_points") is not None else None
    seed_mask = _tensor_to_numpy(vis_data["endpoint_seed_mask"][b]) > 0.0 if vis_data.get("endpoint_seed_mask") is not None else None

    focus_mask = domain > 0.0 if domain is not None else cond > 0.0
    if np.any(focus_mask):
        z0, y0, x0 = np.median(np.argwhere(focus_mask), axis=0).astype(int).tolist()
    else:
        z0, y0, x0 = D // 2, H // 2, W // 2
    slices = [("z", z0), ("y", y0), ("x", x0)]

    def _slice(arr, axis, idx):
        if arr is None:
            return None
        if axis == "z":
            return arr[idx]
        if axis == "y":
            return arr[:, idx, :]
        return arr[:, :, idx]

    def _finite_vmax(arr, fallback=1.0, percentile=99):
        if arr is None:
            return fallback
        vals = arr[np.isfinite(arr)]
        if vals.size == 0:
            return fallback
        value = float(np.percentile(np.abs(vals), percentile))
        return value if np.isfinite(value) and value > 1e-8 else fallback

    progress_error = None
    if progress_pred is not None and progress_target is not None:
        progress_error = np.abs(progress_pred - progress_target)
        if domain is not None:
            progress_error = np.where(domain > 0.0, progress_error, np.nan)
    progress_pred_vmin = 0.0
    progress_pred_vmax = 1.0
    if progress_pred is not None:
        vals = progress_pred[np.isfinite(progress_pred)]
        if vals.size > 0:
            progress_pred_vmin = min(0.0, float(np.percentile(vals, 1)))
            progress_pred_vmax = max(1.0, float(np.percentile(vals, 99)))

    cols = [
        "Source/Target",
        "Domain",
        "Velocity Cos",
        "Phi Target",
        "Phi Pred",
        "Phi |Err|",
        "Stop Target",
        "Stop Pred",
        "Target EDT + Seeds",
        "Attract |Err|",
    ]
    fig = plt.figure(figsize=(3.4 * len(cols), 11))
    gs = GridSpec(3, len(cols) + 1, figure=fig, width_ratios=[1] * len(cols) + [1.25], wspace=0.28)
    axes = np.empty((3, len(cols)), dtype=object)
    for r in range(3):
        for c in range(len(cols)):
            axes[r, c] = fig.add_subplot(gs[r, c])
    ax_text = fig.add_subplot(gs[:, len(cols)])
    ax_text.axis("off")

    edt_vmax = _finite_vmax(target_edt, fallback=8.0)
    attract_vmax = _finite_vmax(attract_err, fallback=1.0)
    phi_err_vmax = _finite_vmax(progress_error, fallback=0.25)

    for row, (axis, idx) in enumerate(slices):
        vol_slice = _slice(vol, axis, idx)
        cond_slice = _slice(cond, axis, idx) > 0.5
        target_slice = _slice(target_seg, axis, idx) > 0.5 if target_seg is not None else np.zeros_like(cond_slice)
        overlay = np.stack([vol_slice, vol_slice, vol_slice], axis=-1)
        overlay = (overlay - np.nanmin(overlay)) / max(float(np.nanmax(overlay) - np.nanmin(overlay)), 1e-6)
        overlay[..., 1] = np.maximum(overlay[..., 1], cond_slice.astype(np.float32))
        overlay[..., 0] = np.maximum(overlay[..., 0], target_slice.astype(np.float32))

        panels = [
            (overlay, None, None, None),
            (_slice(domain, axis, idx), "gray", 0, 1),
            (_slice(velocity_align, axis, idx), "coolwarm", -1, 1),
            (_slice(progress_target, axis, idx), "viridis", 0, 1),
            (_slice(progress_pred, axis, idx), "viridis", progress_pred_vmin, progress_pred_vmax),
            (_slice(progress_error, axis, idx), "magma", 0, phi_err_vmax),
            (_slice(stop_target, axis, idx), "viridis", 0, 1),
            (_slice(stop_pred, axis, idx), "viridis", 0, 1),
            (_slice(target_edt, axis, idx), "magma", 0, edt_vmax),
            (_slice(attract_err, axis, idx), "magma", 0, attract_vmax),
        ]
        for col, (panel, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row, col]
            if panel is None:
                panel = np.zeros_like(vol_slice)
            ax.imshow(panel, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{cols[col]} ({axis}={idx})")
            ax.set_xticks([])
            ax.set_yticks([])

        if seeds is not None and seed_mask is not None:
            active = seeds[seed_mask]
            if active.size > 0:
                if axis == "z":
                    near = np.abs(active[:, 0] - idx) <= 1.5
                    xy = active[near][:, [2, 1]]
                elif axis == "y":
                    near = np.abs(active[:, 1] - idx) <= 1.5
                    xy = active[near][:, [2, 0]]
                else:
                    near = np.abs(active[:, 2] - idx) <= 1.5
                    xy = active[near][:, [1, 0]]
                if xy.size > 0:
                    axes[row, 8].scatter(xy[:, 0], xy[:, 1], s=8, c="cyan", edgecolors="black", linewidths=0.2)

    def _finite_mean(arr, mask=None):
        if arr is None:
            return 0.0
        vals = arr[mask] if mask is not None and np.any(mask) else arr.reshape(-1)
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()) if vals.size else 0.0

    domain_mask = domain > 0.0 if domain is not None else None
    side = np.nanmean(side_hint.reshape(3, -1), axis=1)
    stats_lines = [
        "=" * 40,
        "COPY NEIGHBOR TRACE SUMMARY",
        "=" * 40,
        f"Input shape: {D}x{H}x{W}",
        f"Conditioning voxels: {int((cond > 0.5).sum())}",
        f"Target voxels: {int((target_seg > 0.5).sum()) if target_seg is not None else 0}",
        f"Domain voxels: {int(domain_mask.sum()) if domain_mask is not None else 0}",
        f"Side hint mean ZYX: {side[0]:.3f}, {side[1]:.3f}, {side[2]:.3f}",
        "",
        "--- Velocity ---",
        f"Cosine mean: {_finite_mean(velocity_align, domain_mask):.4f}",
        "",
        "--- Progress ---",
        f"Phi target mean: {_finite_mean(progress_target, domain_mask):.4f}",
        f"Phi pred mean:   {_finite_mean(progress_pred, domain_mask):.4f}",
        f"Phi abs err:     {_finite_mean(progress_error, domain_mask):.4f}",
        "",
        "--- Stop / Attract ---",
        f"Stop target voxels: {int((stop_target > 0.5).sum()) if stop_target is not None else 0}",
        f"Stop pred mean:     {_finite_mean(stop_pred, domain_mask):.4f}",
        f"Attract err mean:   {_finite_mean(attract_err):.4f}",
    ]
    if vis_data.get("endpoint_step_count") is not None:
        steps = _tensor_to_numpy(vis_data["endpoint_step_count"])
        stats_lines.append(f"Endpoint steps:     {int(steps[b])}")
    if seed_mask is not None:
        stats_lines.append(f"Endpoint seeds:     {int(seed_mask.sum())}")
    stats_lines.append("=" * 40)
    ax_text.text(
        0.04,
        0.96,
        "\n".join(stats_lines),
        transform=ax_text.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
    )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def make_dense_visualization(
    inputs,
    disp_pred,
    dense_gt_displacement,
    dense_loss_weight=None,
    triplet_channel_order=None,
    sdt_pred=None,
    sdt_target=None,
    heatmap_pred=None,
    heatmap_target=None,
    seg_pred=None,
    seg_target=None,
    velocity_dir_pred=None,
    velocity_dir_target=None,
    velocity_loss_weight=None,
    trace_loss_weight=None,
    trace_validity_pred=None,
    trace_validity_target=None,
    trace_validity_weight=None,
    surface_attract_pred=None,
    surface_attract_target=None,
    surface_attract_weight=None,
    save_path=None,
):
    """Create and save PNG visualization for dense displacement supervision."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    b = 0
    D, H, W = inputs.shape[2], inputs.shape[3], inputs.shape[4]

    vol_3d = _tensor_to_numpy(inputs[b, 0])
    cond_3d = _tensor_to_numpy(inputs[b, 1])
    aux_3d = _tensor_to_numpy(inputs[b, 2]) if inputs.shape[1] > 2 else None

    gt_3d = _tensor_to_numpy(dense_gt_displacement[b]) if dense_gt_displacement is not None else None
    if disp_pred is None:
        if gt_3d is not None:
            raise ValueError("dense displacement visualization requires disp_pred when dense GT is provided")
        pred_3d = np.zeros((3, D, H, W), dtype=np.float32)
    else:
        pred_3d = _tensor_to_numpy(disp_pred[b])
    trace_only_mode = gt_3d is None
    triplet_mode = gt_3d is not None and pred_3d.shape[0] == 6 and gt_3d.shape[0] == 6
    if triplet_mode:
        _make_dense_triplet_visualization(
            inputs=inputs,
            disp_pred=disp_pred,
            dense_gt_displacement=dense_gt_displacement,
            dense_loss_weight=dense_loss_weight,
            triplet_channel_order=triplet_channel_order,
            save_path=save_path,
        )
        return

    pred_back_mag_3d = None
    pred_front_mag_3d = None
    gt_back_mag_3d = None
    gt_front_mag_3d = None
    other_wraps_union_3d = None

    if triplet_mode:
        pred_back_mag_3d = np.linalg.norm(pred_3d[:3], axis=0)
        pred_front_mag_3d = np.linalg.norm(pred_3d[3:6], axis=0)
        gt_back_mag_3d = np.linalg.norm(gt_3d[:3], axis=0)
        gt_front_mag_3d = np.linalg.norm(gt_3d[3:6], axis=0)
        other_wraps_union_3d = ((gt_back_mag_3d < 0.5) | (gt_front_mag_3d < 0.5)).astype(np.float32)

        # Aggregate maps for global stats columns.
        pred_mag_3d = 0.5 * (pred_back_mag_3d + pred_front_mag_3d)
        gt_mag_3d = 0.5 * (gt_back_mag_3d + gt_front_mag_3d)
        gt_surface_3d = other_wraps_union_3d
        resid_back_mag_3d = np.linalg.norm(pred_3d[:3] - gt_3d[:3], axis=0)
        resid_front_mag_3d = np.linalg.norm(pred_3d[3:6] - gt_3d[3:6], axis=0)
        resid_mag_3d = 0.5 * (resid_back_mag_3d + resid_front_mag_3d)
    else:
        pred_mag_3d = np.linalg.norm(pred_3d, axis=0)
        if gt_3d is not None:
            gt_mag_3d = np.linalg.norm(gt_3d, axis=0)
            gt_surface_3d = (gt_mag_3d < 0.5).astype(np.float32)
            resid_mag_3d = np.linalg.norm(pred_3d - gt_3d, axis=0)
        else:
            gt_mag_3d = np.zeros_like(pred_mag_3d)
            gt_surface_3d = np.zeros_like(pred_mag_3d)
            resid_mag_3d = np.zeros_like(pred_mag_3d)

    if dense_loss_weight is None:
        weight_3d = np.ones((D, H, W), dtype=np.float32)
    else:
        w = _tensor_to_numpy(dense_loss_weight[b])
        weight_3d = w[0] if w.ndim == 4 else w
    weight_3d = weight_3d.astype(np.float32, copy=False)
    supervised_mask = weight_3d > 0

    def _safe_percentile(arr, p, fallback=1.0):
        if arr.size == 0:
            return fallback
        val = np.percentile(arr, p)
        return float(val if np.isfinite(val) and val > 1e-8 else fallback)

    def _weighted_mean(arr, wts):
        den = float(wts.sum())
        if den <= 1e-8:
            return float(arr.mean())
        return float((arr * wts).sum() / den)

    def _finite_mean(arr, mask=None):
        if arr is None:
            return 0.0
        if mask is not None:
            if not np.any(mask):
                return 0.0
            vals = arr[mask]
        else:
            vals = arr.reshape(-1)
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()) if vals.size > 0 else 0.0

    pred_for_scale = pred_mag_3d[supervised_mask] if np.any(supervised_mask) else pred_mag_3d.reshape(-1)
    gt_for_scale = gt_mag_3d[supervised_mask] if np.any(supervised_mask) else gt_mag_3d.reshape(-1)
    resid_for_scale = resid_mag_3d[supervised_mask] if np.any(supervised_mask) else resid_mag_3d.reshape(-1)
    disp_vmax = max(_safe_percentile(pred_for_scale, 99), _safe_percentile(gt_for_scale, 99))
    resid_vmax = _safe_percentile(resid_for_scale, 99)
    back_disp_vmax = None
    front_disp_vmax = None
    if triplet_mode:
        pred_back_for_scale = pred_back_mag_3d[supervised_mask] if np.any(supervised_mask) else pred_back_mag_3d.reshape(-1)
        gt_back_for_scale = gt_back_mag_3d[supervised_mask] if np.any(supervised_mask) else gt_back_mag_3d.reshape(-1)
        pred_front_for_scale = pred_front_mag_3d[supervised_mask] if np.any(supervised_mask) else pred_front_mag_3d.reshape(-1)
        gt_front_for_scale = gt_front_mag_3d[supervised_mask] if np.any(supervised_mask) else gt_front_mag_3d.reshape(-1)
        back_disp_vmax = max(_safe_percentile(pred_back_for_scale, 99), _safe_percentile(gt_back_for_scale, 99))
        front_disp_vmax = max(_safe_percentile(pred_front_for_scale, 99), _safe_percentile(gt_front_for_scale, 99))

    sdt_pred_3d = _tensor_to_numpy(sdt_pred[b, 0]) if sdt_pred is not None else None
    sdt_gt_3d = _tensor_to_numpy(sdt_target[b, 0]) if sdt_target is not None else None
    sdt_vmax = 1.0
    if sdt_pred_3d is not None:
        sdt_vmax = max(float(np.abs(sdt_pred_3d).max()), float(np.abs(sdt_gt_3d).max()) if sdt_gt_3d is not None else 0.0, 1e-6)
    hm_pred_3d = _tensor_to_numpy(torch.sigmoid(heatmap_pred[b, 0])) if heatmap_pred is not None else None
    hm_gt_3d = _tensor_to_numpy(heatmap_target[b, 0]) if heatmap_target is not None else None
    has_seg = seg_pred is not None and seg_target is not None
    seg_pred_3d = _tensor_to_numpy(seg_pred[b].argmax(dim=0)) if seg_pred is not None else None
    seg_gt_3d = _tensor_to_numpy(seg_target[b, 0]) if seg_target is not None else None

    has_trace = (
        velocity_dir_pred is not None
        or trace_validity_pred is not None
        or surface_attract_pred is not None
    )
    trace_weight_3d = _tensor_to_numpy(trace_loss_weight[b, 0]) if trace_loss_weight is not None else None
    velocity_weight_3d = _tensor_to_numpy(velocity_loss_weight[b, 0]) if velocity_loss_weight is not None else trace_weight_3d
    attract_weight_3d = (
        _tensor_to_numpy(surface_attract_weight[b, 0])
        if surface_attract_weight is not None else None
    )
    trace_mask_3d = trace_weight_3d > 0.0 if trace_weight_3d is not None else None
    velocity_mask_3d = velocity_weight_3d > 0.0 if velocity_weight_3d is not None else trace_mask_3d
    attract_mask_3d = attract_weight_3d > 0.0 if attract_weight_3d is not None else None
    vel_align_3d = None
    if velocity_dir_pred is not None and velocity_dir_target is not None:
        pred_vel = F.normalize(velocity_dir_pred[b:b + 1].float(), dim=1, eps=1e-6)
        target_vel = F.normalize(velocity_dir_target[b:b + 1].float(), dim=1, eps=1e-6)
        vel_align_3d = _tensor_to_numpy((pred_vel * target_vel).sum(dim=1)[0].clamp(-1.0, 1.0))
        if velocity_mask_3d is not None:
            vel_align_3d = np.where(velocity_mask_3d, vel_align_3d, np.nan)
    trace_validity_pred_3d = (
        _tensor_to_numpy(torch.sigmoid(trace_validity_pred[b, 0].float()))
        if trace_validity_pred is not None else None
    )
    trace_validity_gt_3d = _tensor_to_numpy(trace_validity_target[b, 0]) if trace_validity_target is not None else None
    trace_validity_weight_3d = (
        _tensor_to_numpy(trace_validity_weight[b, 0])
        if trace_validity_weight is not None else None
    )
    trace_validity_mask_3d = trace_validity_weight_3d > 0.0 if trace_validity_weight_3d is not None else None
    attract_pred_mag_3d = (
        np.linalg.norm(_tensor_to_numpy(surface_attract_pred[b]), axis=0)
        if surface_attract_pred is not None else None
    )
    attract_pred_vec_3d = _tensor_to_numpy(surface_attract_pred[b]) if surface_attract_pred is not None else None
    attract_gt_vec_3d = _tensor_to_numpy(surface_attract_target[b]) if surface_attract_target is not None else None
    attract_gt_mag_3d = (
        np.linalg.norm(attract_gt_vec_3d, axis=0)
        if surface_attract_target is not None else None
    )
    attract_err_mag_3d = (
        np.linalg.norm(attract_pred_vec_3d - attract_gt_vec_3d, axis=0)
        if attract_pred_vec_3d is not None and attract_gt_vec_3d is not None else None
    )
    if attract_mask_3d is not None:
        if attract_pred_mag_3d is not None:
            attract_pred_mag_3d = np.where(attract_mask_3d, attract_pred_mag_3d, np.nan)
        if attract_gt_mag_3d is not None:
            attract_gt_mag_3d = np.where(attract_mask_3d, attract_gt_mag_3d, np.nan)
        if attract_err_mag_3d is not None:
            attract_err_mag_3d = np.where(attract_mask_3d, attract_err_mag_3d, np.nan)
    attract_vmax = 1.0
    if attract_pred_mag_3d is not None or attract_gt_mag_3d is not None:
        attract_vals = [
            arr[np.isfinite(arr)]
            for arr in (attract_pred_mag_3d, attract_gt_mag_3d, attract_err_mag_3d)
            if arr is not None
        ]
        attract_vals = [vals for vals in attract_vals if vals.size > 0]
        attract_vmax = _safe_percentile(np.concatenate(attract_vals), 99, fallback=1.0) if attract_vals else 1.0

    use_aux = aux_3d is not None and not trace_only_mode
    use_sdt = sdt_pred_3d is not None
    use_hm = hm_pred_3d is not None
    n_trace_cols = 0
    n_trace_cols += int(vel_align_3d is not None)
    n_trace_cols += int(trace_validity_pred_3d is not None)
    n_trace_cols += int(trace_validity_gt_3d is not None)
    n_trace_cols += int(trace_validity_weight_3d is not None)
    n_trace_cols += int(trace_mask_3d is not None)
    n_trace_cols += int(attract_pred_mag_3d is not None)
    n_trace_cols += int(attract_gt_mag_3d is not None)
    n_trace_cols += int(attract_mask_3d is not None)
    n_trace_cols += int(attract_err_mag_3d is not None)
    base_cols = 2 if trace_only_mode else 6
    n_cols = base_cols + int(use_aux) + int(triplet_mode) * 3 + int(use_sdt) + int(use_hm) + int(has_seg) + n_trace_cols

    fig = plt.figure(figsize=(3.6 * n_cols + 4, 14))
    gs = GridSpec(3, n_cols + 1, figure=fig, width_ratios=[1] * n_cols + [1.2], wspace=0.3)
    axes = np.empty((3, n_cols), dtype=object)
    for r in range(3):
        for c in range(n_cols):
            axes[r, c] = fig.add_subplot(gs[r, c])
    ax_text = fig.add_subplot(gs[:, n_cols])
    ax_text.axis('off')

    z0, y0, x0 = D // 2, H // 2, W // 2
    slices = [
        ("z", z0, f"z={z0}", "x", "y", [-W / 2, W / 2, H / 2, -H / 2]),
        ("y", y0, f"y={y0}", "x", "z", [-W / 2, W / 2, D / 2, -D / 2]),
        ("x", x0, f"x={x0}", "y", "z", [-H / 2, H / 2, D / 2, -D / 2]),
    ]

    def _slice(arr, axis, idx):
        if axis == "z":
            return arr[idx]
        if axis == "y":
            return arr[:, idx, :]
        return arr[:, :, idx]

    for row, (axis, idx, label, xlabel, ylabel, extent) in enumerate(slices):
        vol_slice = _slice(vol_3d, axis, idx)
        cond_slice = _slice(cond_3d, axis, idx)
        aux_slice = _slice(aux_3d, axis, idx) if use_aux else None
        pred_slice = _slice(pred_mag_3d, axis, idx)
        gt_surface_slice = _slice(gt_surface_3d, axis, idx)
        gt_slice = _slice(gt_mag_3d, axis, idx)
        resid_slice = _slice(resid_mag_3d, axis, idx)

        col = 0
        axes[row, col].imshow(vol_slice, cmap="gray", extent=extent)
        axes[row, col].set_title(f"Volume ({label})")
        axes[row, col].set_ylabel(ylabel)
        col += 1

        axes[row, col].imshow(cond_slice, cmap="gray", extent=extent)
        axes[row, col].set_title("Conditioning")
        axes[row, col].set_yticks([])
        col += 1

        if triplet_mode:
            other_wraps_slice = _slice(other_wraps_union_3d, axis, idx)
            back_pred_slice = _slice(pred_back_mag_3d, axis, idx)
            back_gt_slice = _slice(gt_back_mag_3d, axis, idx)
            front_pred_slice = _slice(pred_front_mag_3d, axis, idx)
            front_gt_slice = _slice(gt_front_mag_3d, axis, idx)

            axes[row, col].imshow(other_wraps_slice, cmap="gray", vmin=0, vmax=1, extent=extent)
            axes[row, col].set_title("Other Wraps (Both)")
            axes[row, col].set_yticks([])
            col += 1

            pair_extent = [extent[0], extent[0] + 2 * (extent[1] - extent[0]), extent[2], extent[3]]
            back_pair = np.concatenate([back_pred_slice, back_gt_slice], axis=1)
            front_pair = np.concatenate([front_pred_slice, front_gt_slice], axis=1)

            axes[row, col].imshow(back_pair, cmap="hot", vmin=0, vmax=back_disp_vmax, extent=pair_extent)
            axes[row, col].set_title("Behind Disp (L:Pred R:GT)")
            axes[row, col].set_yticks([])
            col += 1

            axes[row, col].imshow(front_pair, cmap="hot", vmin=0, vmax=front_disp_vmax, extent=pair_extent)
            axes[row, col].set_title("Front Disp (L:Pred R:GT)")
            axes[row, col].set_yticks([])
            col += 1

        if use_aux:
            axes[row, col].imshow(aux_slice, cmap="gray", extent=extent)
            axes[row, col].set_title("Input Ch2")
            axes[row, col].set_yticks([])
            col += 1

        if not trace_only_mode:
            axes[row, col].imshow(pred_slice, cmap="hot", vmin=0, vmax=disp_vmax, extent=extent)
            axes[row, col].set_title("Pred Disp Mag")
            axes[row, col].set_yticks([])
            col += 1

            axes[row, col].imshow(gt_surface_slice, cmap="gray", vmin=0, vmax=1, extent=extent)
            axes[row, col].set_title("GT Surface (full)")
            axes[row, col].set_yticks([])
            col += 1

            axes[row, col].imshow(gt_slice, cmap="hot", vmin=0, vmax=disp_vmax, extent=extent)
            axes[row, col].set_title("GT Disp Mag")
            axes[row, col].set_yticks([])
            col += 1

            axes[row, col].imshow(resid_slice, cmap="hot", vmin=0, vmax=resid_vmax, extent=extent)
            axes[row, col].set_title("|Pred-GT|")
            axes[row, col].set_yticks([])
            col += 1

        if use_sdt:
            sdt_pred_slice = _slice(sdt_pred_3d, axis, idx)
            axes[row, col].imshow(sdt_pred_slice, cmap="RdBu", vmin=-sdt_vmax, vmax=sdt_vmax, extent=extent)
            axes[row, col].set_title("SDT Pred")
            axes[row, col].set_yticks([])
            col += 1

        if use_hm:
            hm_pred_slice = _slice(hm_pred_3d, axis, idx)
            axes[row, col].imshow(hm_pred_slice, cmap="hot", vmin=0, vmax=1, extent=extent)
            axes[row, col].set_title("Heatmap Pred")
            axes[row, col].set_yticks([])
            col += 1

        if has_seg:
            seg_pred_slice = _slice(seg_pred_3d, axis, idx) > 0.5
            seg_gt_slice = _slice(seg_gt_3d, axis, idx) > 0.5
            overlay = np.zeros((*seg_pred_slice.shape, 3), dtype=np.float32)
            overlay[..., 0] = seg_pred_slice.astype(np.float32)
            overlay[..., 1] = seg_gt_slice.astype(np.float32)
            axes[row, col].imshow(overlay, extent=extent)
            axes[row, col].set_title("Seg Pred(R) GT(G)")
            axes[row, col].set_yticks([])
            col += 1

        if vel_align_3d is not None:
            axes[row, col].imshow(_slice(vel_align_3d, axis, idx), cmap="coolwarm", vmin=-1, vmax=1, extent=extent)
            axes[row, col].set_title("Vel Cos(Pred,GT)")
            axes[row, col].set_yticks([])
            col += 1

        if trace_validity_pred_3d is not None:
            axes[row, col].imshow(_slice(trace_validity_pred_3d, axis, idx), cmap="viridis", vmin=0, vmax=1, extent=extent)
            axes[row, col].set_title("Validity Pred")
            axes[row, col].set_yticks([])
            col += 1

        if trace_validity_gt_3d is not None:
            axes[row, col].imshow(_slice(trace_validity_gt_3d, axis, idx), cmap="viridis", vmin=0, vmax=1, extent=extent)
            axes[row, col].set_title("Validity GT")
            axes[row, col].set_yticks([])
            col += 1

        if trace_validity_weight_3d is not None:
            axes[row, col].imshow(_slice(trace_validity_weight_3d, axis, idx), cmap="gray", vmin=0, vmax=1, extent=extent)
            axes[row, col].set_title("Validity Weight")
            axes[row, col].set_yticks([])
            col += 1

        if trace_mask_3d is not None:
            axes[row, col].imshow(_slice(trace_mask_3d.astype(np.float32), axis, idx), cmap="gray", vmin=0, vmax=1, extent=extent)
            axes[row, col].set_title("Trace Sup Mask")
            axes[row, col].set_yticks([])
            col += 1

        if attract_pred_mag_3d is not None:
            axes[row, col].imshow(_slice(attract_pred_mag_3d, axis, idx), cmap="hot", vmin=0, vmax=attract_vmax, extent=extent)
            axes[row, col].set_title("Attract Pred Mag")
            axes[row, col].set_yticks([])
            col += 1

        if attract_gt_mag_3d is not None:
            axes[row, col].imshow(_slice(attract_gt_mag_3d, axis, idx), cmap="hot", vmin=0, vmax=attract_vmax, extent=extent)
            axes[row, col].set_title("Attract GT Mag")
            axes[row, col].set_yticks([])
            col += 1

        if attract_mask_3d is not None:
            axes[row, col].imshow(_slice(attract_mask_3d.astype(np.float32), axis, idx), cmap="gray", vmin=0, vmax=1, extent=extent)
            axes[row, col].set_title("Attract Band")
            axes[row, col].set_yticks([])
            col += 1

        if attract_err_mag_3d is not None:
            axes[row, col].imshow(_slice(attract_err_mag_3d, axis, idx), cmap="hot", vmin=0, vmax=attract_vmax, extent=extent)
            axes[row, col].set_title("Attract |Pred-GT|")
            axes[row, col].set_yticks([])

        for c in range(n_cols):
            axes[row, c].set_xlabel(xlabel)

    cond_mask = cond_3d > 0.5
    cond_pred = pred_mag_3d[cond_mask]
    cond_gt = gt_mag_3d[cond_mask]
    cond_back_pred = pred_back_mag_3d[cond_mask] if triplet_mode else None
    cond_back_gt = gt_back_mag_3d[cond_mask] if triplet_mode else None
    cond_front_pred = pred_front_mag_3d[cond_mask] if triplet_mode else None
    cond_front_gt = gt_front_mag_3d[cond_mask] if triplet_mode else None

    gt_floor = 0.10
    improvement_clip = 100.0
    meaningful = (gt_mag_3d > gt_floor) & supervised_mask
    if np.any(meaningful):
        imp = (1.0 - resid_mag_3d[meaningful] / np.maximum(gt_mag_3d[meaningful], 1e-8)) * 100.0
        imp = np.clip(imp[np.isfinite(imp)], -improvement_clip, improvement_clip)
        improvement = float(np.median(imp)) if imp.size > 0 else 0.0
        n_imp = int(imp.size)
    else:
        improvement = 0.0
        n_imp = 0

    w = weight_3d
    if trace_only_mode:
        stats_lines = [
            "=" * 40,
            "TRACE FIELD SUMMARY",
            "=" * 40,
            f"Input shape: {D}x{H}x{W}",
            f"Conditioning voxels: {int(cond_mask.sum())}",
            "",
            "--- Velocity ---",
            f"Supervised voxels: {int(velocity_mask_3d.sum()) if velocity_mask_3d is not None else 0}",
            f"Cosine mean:       {_finite_mean(vel_align_3d, velocity_mask_3d):.4f}",
        ]
        if vel_align_3d is not None and velocity_mask_3d is not None and np.any(velocity_mask_3d):
            vals = vel_align_3d[velocity_mask_3d]
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                stats_lines.extend([
                    f"Cosine p10/p50:    {np.percentile(vals, 10):.4f} / {np.percentile(vals, 50):.4f}",
                    f"Cosine below 0.5:  {100.0 * np.mean(vals < 0.5):.1f}%",
                ])
        stats_lines.extend([
            "",
            "--- Validity ---",
            f"Weighted voxels:   {int(trace_validity_mask_3d.sum()) if trace_validity_mask_3d is not None else 0}",
        ])
        if trace_validity_gt_3d is not None and trace_validity_mask_3d is not None:
            valid_pos = trace_validity_mask_3d & (trace_validity_gt_3d > 0.5)
            valid_neg = trace_validity_mask_3d & (trace_validity_gt_3d <= 0.5)
            stats_lines.extend([
                f"Positive voxels:   {int(valid_pos.sum())}",
                f"Negative voxels:   {int(valid_neg.sum())}",
                f"Pred on positives: {_finite_mean(trace_validity_pred_3d, valid_pos):.4f}",
                f"Pred on negatives: {_finite_mean(trace_validity_pred_3d, valid_neg):.4f}",
            ])
        stats_lines.extend([
            "",
            "--- Surface attraction ---",
            f"Band voxels:       {int(attract_mask_3d.sum()) if attract_mask_3d is not None else 0}",
            f"Pred magnitude:    {_finite_mean(attract_pred_mag_3d, attract_mask_3d):.4f}",
            f"GT magnitude:      {_finite_mean(attract_gt_mag_3d, attract_mask_3d):.4f}",
            f"Error magnitude:   {_finite_mean(attract_err_mag_3d, attract_mask_3d):.4f}",
        ])
    else:
        stats_lines = [
            "=" * 40,
            "DENSE DISPLACEMENT STATS",
            "=" * 40,
            f"Triplet mode: {'yes' if triplet_mode else 'no'}",
            f"Supervised voxels: {int(supervised_mask.sum())}",
            f"Total voxels:      {int(weight_3d.size)}",
            "",
            "--- Weighted means ---",
            f"Pred |disp|: {_weighted_mean(pred_mag_3d, w):.4f}",
            f"GT   |disp|: {_weighted_mean(gt_mag_3d, w):.4f}",
            f"Resid|disp|: {_weighted_mean(resid_mag_3d, w):.4f}",
            "",
            "--- Improvement ---",
            f"Median % improvement: {improvement:.1f}%",
            f"  gt_mag > {gt_floor:.2f}, clipped to +/-{improvement_clip:.0f}% (n={n_imp})",
            "",
            "--- Conditioning voxels ---",
            f"N cond voxels: {int(cond_mask.sum())}",
            f"Pred |disp| @ cond mean: {float(cond_pred.mean()) if cond_pred.size > 0 else 0.0:.4f}",
            f"GT   |disp| @ cond mean: {float(cond_gt.mean()) if cond_gt.size > 0 else 0.0:.4f}",
        ]
    if has_trace and not trace_only_mode:
        stats_lines.extend([
            "",
            "--- Trace heads ---",
            f"Velocity supervised voxels: {int(velocity_mask_3d.sum()) if velocity_mask_3d is not None else 0}",
            f"Trace supervised voxels:    {int(trace_mask_3d.sum()) if trace_mask_3d is not None else 0}",
            f"Validity weighted voxels:   {int(trace_validity_mask_3d.sum()) if trace_validity_mask_3d is not None else 0}",
            f"Attract band voxels:        {int(attract_mask_3d.sum()) if attract_mask_3d is not None else 0}",
            f"Velocity cosine mean: {_finite_mean(vel_align_3d, velocity_mask_3d):.4f}",
            f"Validity pred mean:   {_finite_mean(trace_validity_pred_3d, trace_validity_mask_3d):.4f}",
            f"Attract pred mean:    {_finite_mean(attract_pred_mag_3d, attract_mask_3d):.4f}",
            f"Attract GT mean:      {_finite_mean(attract_gt_mag_3d, attract_mask_3d):.4f}",
            f"Attract error mean:   {_finite_mean(attract_err_mag_3d, attract_mask_3d):.4f}",
        ])
    if triplet_mode:
        stats_lines.extend([
            "",
            "--- Conditioning voxels (per neighbor) ---",
            f"Behind Pred|disp| @ cond mean: {float(cond_back_pred.mean()) if cond_back_pred.size > 0 else 0.0:.4f}",
            f"Behind GT  |disp| @ cond mean: {float(cond_back_gt.mean()) if cond_back_gt.size > 0 else 0.0:.4f}",
            f"Front  Pred|disp| @ cond mean: {float(cond_front_pred.mean()) if cond_front_pred.size > 0 else 0.0:.4f}",
            f"Front  GT  |disp| @ cond mean: {float(cond_front_gt.mean()) if cond_front_gt.size > 0 else 0.0:.4f}",
        ])
    stats_lines.append("=" * 40)
    ax_text.text(
        0.05, 0.95, "\n".join(stats_lines),
        transform=ax_text.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
    )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
