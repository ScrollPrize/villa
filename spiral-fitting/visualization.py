import hashlib
import os
import pickle

import kornia
import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw
from tqdm import tqdm

from sample_spiral import get_bounding_windings, get_theta, get_theta_and_radii
from tracks import render_spiral_on_tracks_for_slice


def _update_hash_with_array(hasher, array):
    array = np.ascontiguousarray(array)
    hasher.update(str(array.dtype).encode('ascii'))
    hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    hasher.update(array.tobytes())


def _patch_lines_cache_filename(patches, all_zs, cache_path, cache_version):
    patch_hasher = hashlib.sha256()
    for patch in patches:
        uuid = getattr(patch, 'uuid', None)
        if uuid is not None:
            patch_hasher.update(str(uuid).encode('utf-8'))
        patch_hasher.update(b'\0')
        _update_hash_with_array(patch_hasher, patch.zyxs.detach().cpu().numpy())
        _update_hash_with_array(patch_hasher, patch.valid_quad_mask.detach().cpu().numpy())
    hashed_patches = patch_hasher.hexdigest()[:16]

    slices_hasher = hashlib.sha256()
    _update_hash_with_array(slices_hasher, all_zs)
    hashed_slices = slices_hasher.hexdigest()[:16]

    return f'{cache_path}/patches-{hashed_patches}_lines_v3_{cache_version}_slices-{hashed_slices}.pkl'


def _compute_patch_lines_by_slice(patches, slice_zs, cache_path, cache_version='full_res_v1'):
    all_zs = np.array(slice_zs)
    cache_filename = _patch_lines_cache_filename(patches, all_zs, cache_path, cache_version)

    if os.path.exists(cache_filename):
        with open(cache_filename, 'rb') as fp:
            lines_by_slice = pickle.load(fp)
    else:
        lines_by_slice = [[] for _ in all_zs]
        for patch_idx, patch in enumerate(tqdm(patches, desc='slicing patches')):
            mesh = patch.to_trimesh()
            if mesh.faces.shape[0] == 0:
                continue
            patch_min_z, patch_max_z = mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max()
            relevant_zs_mask = (all_zs >= patch_min_z) & (all_zs <= patch_max_z)
            relevant_zs = all_zs[relevant_zs_mask]
            if len(relevant_zs) == 0:
                continue
            lines, to_3ds, face_indices = trimesh.intersections.mesh_multiplane(mesh, [0, 0, 0], [0, 0, 1], relevant_zs)
            # lines has one entry per slice; each entry is a tensor of lines represented by pairs of yx points, indexed [line-idx, start/end, yx]
            # face_indices[slice] gives the trimesh face index per line; we map back through the valid-quad filter to a quad index within the patch grid.
            assert all((to_3d[:3, :3] == np.eye(3)).all() for to_3d in to_3ds)
            assert (to_3ds[:, 2, 3] == relevant_zs).all()
            valid_quads_flat = patch.valid_quad_mask.flatten().numpy()
            num_quads_grid = valid_quads_flat.shape[0]
            face_mask = np.concatenate([valid_quads_flat, valid_quads_flat], axis=0)
            filtered_face_to_full = np.where(face_mask)[0]
            relevant_indices = np.where(relevant_zs_mask)[0]
            for local_idx, global_idx in enumerate(relevant_indices):
                if lines[local_idx] is not None and len(lines[local_idx]) > 0:
                    line_face_indices = np.asarray(face_indices[local_idx], dtype=np.int64)
                    line_quad_indices = (filtered_face_to_full[line_face_indices] % num_quads_grid).astype(np.int64)
                    lines_by_slice[global_idx].append((patch_idx, lines[local_idx], line_quad_indices))
        os.makedirs(cache_path, exist_ok=True)
        with open(cache_filename, 'wb') as fp:
            pickle.dump(lines_by_slice, fp)

    return lines_by_slice


def overlay_patches_on_slices(patches, slice_zs, slice_shape, cache_path, cache_version='full_res_v1', canvas_scale=1):
    color_slices = torch.zeros([len(slice_zs), *slice_shape, 3], dtype=torch.uint8)
    quad_label_slices = torch.zeros(color_slices.shape[:-1], dtype=torch.int32, device=color_slices.device)
    lines_by_slice = _compute_patch_lines_by_slice(patches, slice_zs, cache_path, cache_version)
    patch_colors = [tuple(np.random.randint(100, 256, size=3).tolist()) for _ in patches]

    # Each patch contributes (H-1)*(W-1) grid quads (row-major, including invalid ones - they
    # never appear as lines, but indexing this way matches the flat satisfied_quad_mask layout).
    quad_offsets = np.zeros(len(patches) + 1, dtype=np.int64)
    for patch_idx, patch in enumerate(patches):
        h, w = patch.zyxs.shape[:2]
        quad_offsets[patch_idx + 1] = quad_offsets[patch_idx] + (h - 1) * (w - 1)

    for slice_idx in tqdm(range(len(slice_zs)), desc='rasterising patch lines'):
        slice_lines = lines_by_slice[slice_idx]
        color_img = Image.fromarray(np.zeros(color_slices.shape[1:4], dtype=np.uint8))
        quad_label_img = Image.fromarray(np.zeros(quad_label_slices.shape[1:3], dtype=np.int32), mode='I')
        color_draw = ImageDraw.Draw(color_img)
        quad_label_draw = ImageDraw.Draw(quad_label_img)
        for patch_idx, lines, line_quad_indices in slice_lines:
            color = patch_colors[patch_idx] if patch_idx < len(patch_colors) else (255, 255, 255)
            offset = int(quad_offsets[patch_idx])
            for line_xy, quad_idx in zip(lines, line_quad_indices):
                line_points = (line_xy / canvas_scale).flatten().tolist()
                color_draw.line(line_points, fill=color)
                quad_label_draw.line(line_points, fill=int(offset + int(quad_idx) + 1))
        color_slices[slice_idx] = torch.from_numpy(np.array(color_img))
        quad_label_slices[slice_idx] = torch.from_numpy(np.array(quad_label_img))
    return color_slices, quad_label_slices, quad_offsets


def get_winding_positions_on_radials(slice_z, thetas, max_radius, slice_to_spiral_transform, dr_per_winding, z_to_umbilicus_yx):
    theta_slice, radius_slice = torch.meshgrid(thetas, torch.arange(1., max_radius), indexing='ij')
    radials_yx_slice = torch.from_numpy(z_to_umbilicus_yx(slice_z.cpu()).astype(np.float32)) + torch.stack([torch.sin(theta_slice), torch.cos(theta_slice)], dim=-1) * radius_slice[..., None]
    radials_zyx_slice = torch.cat([slice_z.expand(radials_yx_slice.shape[:2])[..., None], radials_yx_slice.to(device=slice_z.device)], dim=-1)
    radials_zyx_spiral = slice_to_spiral_transform(radials_zyx_slice)
    _, _, inner_winding_idx, _ = get_bounding_windings(radials_zyx_spiral[..., 1:], dr_per_winding)
    radii_by_radial = []
    yxs_by_radial = []
    winding_indices_by_radial = []
    for radial_idx in range(inner_winding_idx.shape[0]):
        winding_change_indices = torch.where(torch.diff(inner_winding_idx[radial_idx], prepend=inner_winding_idx[radial_idx, :1]))[0].cpu()
        radii_by_radial.append(radius_slice[radial_idx, winding_change_indices])
        yxs_by_radial.append(radials_yx_slice[radial_idx, winding_change_indices])
        winding_indices_by_radial.append(inner_winding_idx[radial_idx, winding_change_indices])
    return radii_by_radial, yxs_by_radial, winding_indices_by_radial


@torch.no_grad()
def _rasterize_spiral_for_slice(spiral_and_transform, slice_to_spiral_transform, slice_yx, slice_z, winding_range):
    # Transform a full slice from scroll into spiral space (in chunks to bound peak VRAM)
    # and evaluate the per-winding spiral density there. Returns (spiral_zyx, spiral_density).
    device = slice_yx.device
    slice_zyx = torch.cat([
        torch.full([*slice_yx.shape[:2], 1], float(slice_z), device=device),
        slice_yx,
    ], dim=-1)
    slice_zyx_flat = slice_zyx.reshape(-1, 3)
    chunk = 65536
    spiral_pieces = []
    for start in range(0, slice_zyx_flat.shape[0], chunk):
        spiral_pieces.append(slice_to_spiral_transform(slice_zyx_flat[start:start + chunk]))
    spiral_zyx = torch.cat(spiral_pieces, dim=0).reshape(*slice_zyx.shape)
    spiral_density = spiral_and_transform.get_spiral_density(spiral_zyx, winding_range=winding_range)
    return spiral_zyx, spiral_density


@torch.inference_mode()
def save_overlay(
    spiral_and_transform,
    flow_min_corner_spiral_zyx, flow_max_corner_spiral_zyx,
    zs_for_visualisation, slice_yx,
    scroll_slices_for_visualisation, prediction_slices_for_visualisation,
    quad_label_map, quad_status_flat,
    unattached_pcl_strips, unattached_pcl_per_point_satisfied, unattached_pcl_fully_satisfied,
    z_to_umbilicus_yx,
    winding_range,
    tracks,
    out_path, suffix,
    render_volume_scale=1,
):

    device = slice_yx.device
    slice_to_spiral_transform = spiral_and_transform.get_slice_to_spiral_transform()

    # TODO: maybe use the smoothed umbilicus here, to avoid weird swirls appearing
    flow_corners_zyx = slice_to_spiral_transform.inv(torch.stack([flow_min_corner_spiral_zyx, flow_max_corner_spiral_zyx], dim=0).to(torch.float32)).to(torch.int64)
    flow_min_corner_zyx = flow_corners_zyx.amin(dim=0)
    flow_max_corner_zyx = flow_corners_zyx.amax(dim=0)

    def draw_boxes(canvas):
        def draw_box(min_corner_yx, max_corner_yx):
            min_corner_yx = torch.clip(min_corner_yx, torch.zeros_like(min_corner_yx), torch.tensor(canvas.shape[:2], device=min_corner_yx.device) - 1)
            max_corner_yx = torch.clip(max_corner_yx, torch.zeros_like(max_corner_yx), torch.tensor(canvas.shape[:2], device=max_corner_yx.device) - 1)
            canvas[min_corner_yx[0] : max_corner_yx[0], min_corner_yx[1]: min_corner_yx[1] + 1] = 150
            canvas[min_corner_yx[0] : max_corner_yx[0], max_corner_yx[1]: max_corner_yx[1] + 1] = 150
            canvas[min_corner_yx[0] : min_corner_yx[0] + 1, min_corner_yx[1]: max_corner_yx[1]] = 150
            canvas[max_corner_yx[0] : max_corner_yx[0] + 1, min_corner_yx[1]: max_corner_yx[1]] = 150
        draw_box(flow_min_corner_zyx[1:], flow_max_corner_zyx[1:])

    def overlay_on_predictions(spiral, slice, mask, name):
        spiral_colour = torch.tensor([200, 20, 20], device=slice.device)
        canvas = slice.to(torch.float32) * (1 - spiral[..., None]) + spiral_colour * spiral[..., None]
        canvas = canvas.to(torch.uint8).cpu().numpy()
        Image.fromarray(canvas).save(f'{out_path}/spiral_on_{name}_{suffix}.png', compress_level=3)

    def overlay_on_patch_satisfaction(spiral, spiral_zyx, label_map, slice_z, name):
        # quad_status_flat: 0=patch-and-quad not satisfied, 1=quad-only satisfied, 2=patch overall satisfied
        status = quad_status_flat.to(device=spiral.device, dtype=torch.int64)
        label_map = label_map.to(device=spiral.device, dtype=torch.int64)
        num_labels = status.numel()
        status_palette = torch.tensor([
            [200, 0, 0],  # 0: red - quad not satisfied (patch not overall satisfied)
            [230, 140, 0],  # 1: orange - quad satisfied but patch not overall satisfied
            [255, 200, 0],  # 2: yellow - patch overall satisfied
        ], dtype=torch.uint8, device=spiral.device)
        colour_table = torch.zeros([num_labels + 1, 3], dtype=torch.uint8, device=spiral.device)
        colour_table[1:] = status_palette[status]
        # Per-winding spiral colour: cycle through `num_winding_hues` hues so each winding gets a constant colour.
        dr_per_winding = spiral_and_transform.get_dr_per_winding()
        _, _, shifted_radius = get_theta_and_radii(spiral_zyx[..., 1:], dr_per_winding)
        winding_idx = (shifted_radius / dr_per_winding).round().to(torch.int64).clamp_min(0)
        num_winding_hues = 6
        # Cycle hues over [yellow, pink] to avoid satisfaction colours.
        hue_min, hue_max = 1.5 / 6, 5.25 / 6
        hue_fraction = hue_min + (winding_idx % num_winding_hues).to(torch.float32) / num_winding_hues * (hue_max - hue_min)
        hue = hue_fraction * 2 * np.pi
        hsv = torch.stack([hue, torch.full_like(hue, 0.5), torch.ones_like(hue)])
        spiral_colours = kornia.color.hsv_to_rgb(hsv).permute(1, 2, 0) * 255
        canvas = spiral_colours * spiral[..., None]
        patch_mask = (label_map > 0)[..., None]
        canvas = torch.where(patch_mask, colour_table[label_map].to(torch.float32), canvas)
        canvas = canvas.to(torch.uint8).cpu().numpy()
        image = Image.fromarray(canvas)
        if unattached_pcl_strips:
            draw = ImageDraw.Draw(image)
            pcl_z_window = 16
            point_radius = 1
            pcl_palette = [(200, 0, 0), (230, 140, 0), (255, 200, 0)]  # matches status_palette[0:3]
            for k, strip in enumerate(unattached_pcl_strips):
                zyxs = strip['zyxs']
                in_slab = np.abs(zyxs[:, 0] - float(slice_z)) <= pcl_z_window
                if not in_slab.any():
                    continue
                fully = bool(unattached_pcl_fully_satisfied[k].item())
                per_point_sat = unattached_pcl_per_point_satisfied[k].numpy()
                for idx in np.nonzero(in_slab)[0]:
                    y = float(zyxs[idx, 1]) / render_volume_scale
                    x = float(zyxs[idx, 2]) / render_volume_scale
                    if fully:
                        colour = pcl_palette[2]
                    elif per_point_sat[idx]:
                        colour = pcl_palette[1]
                    else:
                        colour = pcl_palette[0]
                    draw.ellipse(
                        [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
                        fill=colour,
                    )
        image.save(f'{out_path}/spiral_on_{name}_{suffix}.png', compress_level=3)

    def visualise_field(spiral_zyx, name):
        # Show the slice->spiral field as RGB: spiral (z, y, x) normalised into [0, 1] per channel
        # using the value range across the slice.
        values = spiral_zyx.to(torch.float32).reshape(-1, 3)
        lo = values.amin(dim=0)
        hi = values.amax(dim=0)
        normalised = ((spiral_zyx.to(torch.float32) - lo) / (hi - lo).clamp_min(1e-6)).clamp(0., 1.)
        canvas = (normalised * 255).to(torch.uint8)
        Image.fromarray(canvas.cpu().numpy()).save(f'{out_path}/{name}_{suffix}.png', compress_level=3)

    def overlay_on_scroll(slice_zyx, spiral_zyx, spiral_density, slice, name):
        slice_min = slice[slice > 0].amin() if (slice > 0).any() else 0
        slice_normalised = (slice - slice_min) / (slice.amax() - slice_min) * (slice > 0)
        spiral_density_normalised = spiral_density / spiral_density.amax()
        theta, relative_yx = get_theta(spiral_zyx[..., 1:])
        # 1. Coloured spiral overlaid on scroll
        theta_colours = kornia.color.hsv_to_rgb(torch.stack([theta, *[torch.ones_like(theta)] * 2])).permute(1, 2, 0) * 0.5
        spiral_density_coloured = spiral_density_normalised[..., None].expand(-1, -1, 3) * theta_colours
        canvas = slice_normalised[..., None].expand(-1, -1, 3) * (1. - spiral_density_normalised[..., None]) + spiral_density_coloured
        canvas *= (slice > 0)[..., None]
        canvas = (canvas * 255).to(torch.uint8)
        draw_boxes(canvas)
        canvas = Image.fromarray(canvas.cpu().numpy())
        draw = ImageDraw.Draw(canvas)
        _, yxs_by_radial, winding_indices_by_radial = get_winding_positions_on_radials(
            slice_z=slice_zyx[0, 0, :1],
            thetas=torch.arange(torch.pi / 8, 2 * torch.pi, torch.pi / 4),
            max_radius=slice_zyx[..., 1:].amax(),
            slice_to_spiral_transform=slice_to_spiral_transform,
            dr_per_winding=spiral_and_transform.get_dr_per_winding(),
            z_to_umbilicus_yx=z_to_umbilicus_yx,
        )
        for radial_idx in range(len(yxs_by_radial)):
            for idx in range(winding_indices_by_radial[radial_idx].shape[0]):
                marker_yx = yxs_by_radial[radial_idx][idx]
                if (marker_yx > 0).all() and (marker_yx < torch.tensor(slice.shape)).all() and slice[*marker_yx.to(torch.int64)] > 0:
                    winding_idx = int(winding_indices_by_radial[radial_idx][idx].item())
                    if winding_idx > 0 and winding_idx % 5 == 0:
                        draw.point(tuple(marker_yx)[::-1])
                        draw.text(
                            tuple(marker_yx)[::-1],
                            str(winding_idx)
                        )
        canvas.save(f'{out_path}/spiral_on_{name}_{suffix}.png', compress_level=3)

    dr_per_winding = spiral_and_transform.get_dr_per_winding()
    for vis_slice_idx, slice_z in enumerate(tqdm(zs_for_visualisation, desc='visualising slices')):
        spiral_zyx, spiral_density = _rasterize_spiral_for_slice(
            spiral_and_transform, slice_to_spiral_transform, slice_yx, slice_z, winding_range,
        )
        slice = scroll_slices_for_visualisation[vis_slice_idx].to(device)
        # overlay_on_scroll(slice_zyx, spiral_zyx, spiral_density, slice, f'scroll_s{slice_z:05}')
        # overlay_on_predictions(spiral_density, prediction_slices_for_visualisation[vis_slice_idx].to(device), slice > 0., f'pred_s{slice_z:05}')
        overlay_on_patch_satisfaction(spiral_density, spiral_zyx, quad_label_map[vis_slice_idx], slice_z, f'patches_s{slice_z:05}')
        if tracks:
            render_spiral_on_tracks_for_slice(
                spiral_zyx, spiral_density, dr_per_winding,
                slice_z, tracks, [],
                out_path, suffix,
                render_volume_scale=render_volume_scale,
            )
        if os.environ.get('FIT_SPIRAL_SAVE_DISPLACEMENT') == '1':
            slice_zyx = torch.cat([torch.full([*slice_yx.shape[:2], 1], slice_z, device=device), slice_yx], dim=-1)
            visualise_field(spiral_zyx - slice_zyx, f'displacement_s{slice_z:05}')
