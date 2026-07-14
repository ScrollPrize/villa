
import os
import json
import torch
import numpy as np
from PIL import Image
from typing import Optional, Union, Literal
from einops import rearrange
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class Patch:
    zyxs: torch.Tensor
    scale: torch.Tensor
    overlapping_ids: Optional[list[str]]  # None implies no overlapping.json (hence unknown overlaps)
    winding: Optional[Union[Literal['single'], torch.Tensor]]  # None means no winding.tif found
    uuid: Optional[str] = None

    def _get_face_indices(self):
        h, w = self.zyxs.shape[:2]
        indices = torch.arange(h * w).view(h, w)
        top_left = indices[:-1, :-1].flatten()
        top_right = indices[:-1, 1:].flatten()
        bottom_left = indices[1:, :-1].flatten()
        bottom_right = indices[1:, 1:].flatten()
        return torch.cat([
            torch.stack([bottom_left, top_left, top_right], dim=1),
            torch.stack([bottom_left, top_right, bottom_right], dim=1)
        ], dim=0)

    def __post_init__(self):
        self.valid_vertex_mask = torch.any(self.zyxs != -1, dim=-1)
        self.valid_vertex_indices = torch.stack(torch.where(self.valid_vertex_mask), dim=-1)
        # In valid_quad_mask, the ij'th element says whether all four corners of the quad with min-corner at vertex ij are valid
        self.valid_quad_mask = self.valid_vertex_mask[:-1, :-1] & self.valid_vertex_mask[1:, :-1] & self.valid_vertex_mask[:-1, 1:] & self.valid_vertex_mask[1:, 1:]
        self.valid_quad_indices = torch.stack(torch.where(self.valid_quad_mask), dim=-1)
        assert len(self.valid_quad_indices) > 0
        self.area = (self.valid_quad_mask).sum() * (1 / self.scale).prod()
        self.valid_zyxs = self.zyxs[self.valid_vertex_mask]

    def ij_to_zyx(self, ij: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Bilinearly interpolate zyx coordinates at fractional (i, j) pixel locations.

        Args:
            ij: Tensor of shape (..., 2) containing (i, j) pixel coordinates in the
                patch grid (row, col). Coordinates outside the patch or on invalid
                quads are marked invalid in the returned mask.

        Returns:
            (zyx, valid_mask) where zyx has shape (..., 3) and invalid samples are
            filled with -1.
        """
        if ij.shape[-1] != 2:
            raise ValueError(f"ij must have shape (..., 2), got {ij.shape}")

        h, w, _ = self.zyxs.shape
        ij_f = ij.to(dtype=torch.float32)
        flat = ij_f.reshape(-1, 2)

        i0 = torch.floor(flat[:, 0]).long()
        j0 = torch.floor(flat[:, 1]).long()

        in_bounds = (i0 >= 0) & (j0 >= 0) & (i0 < h - 1) & (j0 < w - 1)
        valid = torch.zeros_like(in_bounds, dtype=torch.bool)
        if in_bounds.any():
            valid_in_bounds = self.valid_quad_mask.to(ij.device)[i0[in_bounds], j0[in_bounds]]
            valid[in_bounds] = valid_in_bounds

        result = torch.full(
            (flat.shape[0], 3),
            -1.0,
            dtype=torch.float32,
            device=ij.device,
        )

        valid_idx = valid.nonzero(as_tuple=True)[0]
        if len(valid_idx) > 0:
            di = (flat[valid_idx, 0] - i0[valid_idx]).unsqueeze(-1)
            dj = (flat[valid_idx, 1] - j0[valid_idx]).unsqueeze(-1)

            zyxs = self.zyxs.to(ij.device, dtype=torch.float32)
            tl = zyxs[i0[valid_idx], j0[valid_idx]]
            tr = zyxs[i0[valid_idx], j0[valid_idx] + 1]
            bl = zyxs[i0[valid_idx] + 1, j0[valid_idx]]
            br = zyxs[i0[valid_idx] + 1, j0[valid_idx] + 1]

            top = tl + (tr - tl) * dj
            bottom = bl + (br - bl) * dj
            interp = top + (bottom - top) * di
            result[valid_idx] = interp

        zyx = result.reshape(*ij.shape[:-1], 3)
        mask = valid.reshape(ij.shape[:-1])
        return zyx, mask

    def project(self, zyx: torch.Tensor, chunk_size: int = 64):
        """Project 3D points to the closest location on the valid patch surface.

        Returns:
            (ij_coords, dists) where ij_coords has shape (..., 2) with projected
            (i, j) coordinates in patch space (filled with -1 when invalid) and
            dists has shape (...) with the Euclidean distance from each input
            point to its closest surface point (inf when invalid).
        """
        if zyx.shape[-1] != 3:
            raise ValueError(f"zyx must have shape (..., 3), got {zyx.shape}")

        device = zyx.device
        h, w, _ = self.zyxs.shape
        vertices = self.zyxs.to(device=device, dtype=torch.float32).reshape(-1, 3)

        faces_all = self._get_face_indices().to(device)
        valid_quads = self.valid_quad_mask.flatten()
        face_mask = torch.cat([valid_quads, valid_quads], dim=0)
        faces = faces_all[face_mask]
        if faces.numel() == 0:
            out_shape = (*zyx.shape[:-1], 2)
            ij = torch.full(out_shape, -1.0, device=device)
            dist_shape = zyx.shape[:-1]
            dists = torch.full(dist_shape, float('inf'), device=device, dtype=torch.float32)
            return ij, dists

        base_grid = torch.stack(
            torch.meshgrid(
                torch.arange(h - 1, device=device),
                torch.arange(w - 1, device=device),
                indexing='ij',
            ),
            dim=-1,
        ).reshape(-1, 2)
        base_faces = torch.cat([base_grid, base_grid], dim=0)[face_mask]
        face_type = torch.cat([
            torch.zeros_like(valid_quads, dtype=torch.long),
            torch.ones_like(valid_quads, dtype=torch.long)
        ], dim=0)[face_mask]

        a = vertices[faces[:, 0]]
        b = vertices[faces[:, 1]]
        c = vertices[faces[:, 2]]
        ab = b - a
        ac = c - a

        flat = zyx.reshape(-1, 3).to(device=device, dtype=torch.float32)
        out_ij = torch.full((flat.shape[0], 2), -1.0, device=device, dtype=torch.float32)
        out_dist = torch.full((flat.shape[0],), float('inf'), device=device, dtype=torch.float32)
        eps = 1e-8

        for start in range(0, flat.shape[0], chunk_size):
            pts = flat[start:start + chunk_size]
            P = pts[:, None, :]

            AP = P - a
            BP = P - b
            CP = P - c

            d1 = (AP * ab).sum(-1)
            d2 = (AP * ac).sum(-1)
            d3 = (BP * ab).sum(-1)
            d4 = (BP * ac).sum(-1)
            d5 = (CP * ab).sum(-1)
            d6 = (CP * ac).sum(-1)

            vc = d1 * d4 - d3 * d2
            vb = d5 * d2 - d1 * d6
            va = d3 * d6 - d5 * d4

            mask_a = (d1 <= 0) & (d2 <= 0)
            mask_b = (d3 >= 0) & (d4 <= d3)
            mask_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
            mask_c = (d6 >= 0) & (d5 <= d6)
            mask_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
            mask_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
            mask_face = ~(mask_a | mask_b | mask_ab | mask_c | mask_ac | mask_bc)

            bary = torch.zeros((pts.shape[0], faces.shape[0], 3), device=device, dtype=torch.float32)
            bary[..., 0][mask_a] = 1.0
            bary[..., 1][mask_b] = 1.0

            v_ab = d1 / (d1 - d3 + eps)
            bary[..., 0][mask_ab] = 1 - v_ab[mask_ab]
            bary[..., 1][mask_ab] = v_ab[mask_ab]

            bary[..., 2][mask_c] = 1.0

            w_ac = d2 / (d2 - d6 + eps)
            bary[..., 0][mask_ac] = 1 - w_ac[mask_ac]
            bary[..., 2][mask_ac] = w_ac[mask_ac]

            w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6) + eps)
            bary[..., 1][mask_bc] = 1 - w_bc[mask_bc]
            bary[..., 2][mask_bc] = w_bc[mask_bc]

            denom = (va + vb + vc) + eps
            v_face = vb / denom
            w_face = vc / denom
            bary[..., 1][mask_face] = v_face[mask_face]
            bary[..., 2][mask_face] = w_face[mask_face]
            bary[..., 0][mask_face] = 1 - v_face[mask_face] - w_face[mask_face]

            closest = (
                bary[..., 0:1] * a
                + bary[..., 1:2] * b
                + bary[..., 2:3] * c
            )
            dist2 = ((closest - P) ** 2).sum(-1)
            idx = dist2.argmin(dim=1)

            batch_idx = torch.arange(pts.shape[0], device=device)
            best_bary = bary[batch_idx, idx]
            best_base = base_faces[idx]
            best_type = face_type[idx]

            i_offset = torch.where(best_type == 0, best_bary[:, 0], best_bary[:, 0] + best_bary[:, 2])
            j_offset = torch.where(best_type == 0, best_bary[:, 2], best_bary[:, 1] + best_bary[:, 2])

            out_ij[start:start + pts.shape[0], 0] = best_base[:, 0].float() + i_offset
            out_ij[start:start + pts.shape[0], 1] = best_base[:, 1].float() + j_offset
            out_dist[start:start + pts.shape[0]] = dist2[batch_idx, idx].sqrt()

        ij_shape = (*zyx.shape[:-1], 2)
        dist_shape = zyx.shape[:-1]
        return out_ij.reshape(ij_shape), out_dist.reshape(dist_shape)

    def to_trimesh(self):
        import trimesh
        h, w, _ = self.zyxs.shape
        vertices_zyx = self.zyxs.reshape(-1, 3).numpy()
        vertices_xyz = vertices_zyx[:, ::-1].copy()
        faces_all = self._get_face_indices().numpy()
        valid_quads = self.valid_quad_mask.flatten().numpy()
        face_mask = np.concatenate([valid_quads, valid_quads], axis=0)
        faces = faces_all[face_mask]
        return trimesh.Trimesh(vertices=vertices_xyz, faces=faces, process=False)

    def save_obj(self, path: str, scale: float = 1e-3) -> None:

        h, w, _ = self.zyxs.shape
        flat = self.zyxs.reshape(-1, 3)
        valid_mask = self.valid_vertex_mask.reshape(-1)
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
        if valid_indices.numel() == 0:
            raise ValueError("Patch has no valid vertices to export")

        mapping = torch.full((h * w,), -1, dtype=torch.long)
        mapping[valid_indices] = torch.arange(valid_indices.numel(), dtype=torch.long)

        with open(path, "w") as f:
            f.write("# Patch mesh\n")
            f.write(f"# Vertices: {valid_indices.numel()}\n")
            f.write(f"# Triangles: {self.valid_quad_indices.shape[0] * 2}\n\n")

            for idx in valid_indices.tolist():
                z, y, x = flat[idx].tolist()
                f.write(f"v {x * scale} {y * scale} {z * scale}\n")

            f.write("\n")

            for ij in self.valid_quad_indices.tolist():
                i, j = ij
                tl = mapping[i * w + j].item()
                tr = mapping[i * w + (j + 1)].item()
                bl = mapping[(i + 1) * w + j].item()
                br = mapping[(i + 1) * w + (j + 1)].item()
                if min(tl, tr, bl, br) < 0:
                    continue
                f.write(f"f {bl + 1} {tl + 1} {tr + 1}\n")
                f.write(f"f {bl + 1} {tr + 1} {br + 1}\n")


def load_tifxyz(path):

    with open(f'{path}/meta.json', 'r') as meta_json:
        metadata = json.load(meta_json)
        scale = torch.tensor(metadata['scale'])
        uuid = metadata.get('uuid')
    zyxs_np = np.stack([np.array(Image.open(f'{path}/{coord}.tif')) for coord in 'zyx'], axis=-1)

    # Some patches mark invalid vertices via mask.tif (mask == 0) rather than the -1 sentinel in
    # x/y/z, so masked-out vertices can carry real coordinates. Force those to -1 so the standard
    # validity logic (zyxs != -1) is correct. Patches without mask.tif are unaffected.
    mask_path = f'{path}/mask.tif'
    if os.path.exists(mask_path):
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        zyxs_np[mask == 0] = -1.0

    zyxs = torch.from_numpy(zyxs_np).to(torch.float32)

    if os.path.exists(f'{path}/overlapping.json'):
        with open(f'{path}/overlapping.json', 'r') as overlapping_json:
            overlapping_ids = json.load(overlapping_json)['overlapping']
    else:
        overlapping_ids = None

    winding_path = f'{path}/winding.tif'
    if os.path.exists(winding_path):
        winding_np = np.array(Image.open(winding_path))
        assert winding_np.shape[:2] == zyxs.shape[:2] and winding_np.ndim == 2 and winding_np.dtype == np.float32
        wt = torch.from_numpy(winding_np)
        if (torch.isnan(wt) | (wt == 0.)).all():  # all zero/NaN -> mark as single winding
            winding_value = 'single'
        else:
            winding_value = wt
    else:
        winding_value = None

    return Patch(zyxs, scale, overlapping_ids, winding_value, uuid)


def save_tifxyz(zyxs, path, uuid, step_size, voxel_size_um, source):
    path = f'{path}/{uuid}'
    os.makedirs(path, exist_ok=True)
    Image.fromarray(np.asarray(zyxs[..., 2], dtype=np.float32)).save(f'{path}/x.tif')
    Image.fromarray(np.asarray(zyxs[..., 1], dtype=np.float32)).save(f'{path}/y.tif')
    Image.fromarray(np.asarray(zyxs[..., 0], dtype=np.float32)).save(f'{path}/z.tif')
    valid_vertex = np.any(zyxs != -1, axis=-1)
    valid_quad = valid_vertex[:-1, :-1] & valid_vertex[1:, :-1] & valid_vertex[:-1, 1:] & valid_vertex[1:, 1:]
    area_vx2 = int(valid_quad.sum()) * step_size ** 2
    valid_zyxs = zyxs[valid_vertex]
    bbox = (
        [valid_zyxs.min(axis=0)[::-1].tolist(), valid_zyxs.max(axis=0)[::-1].tolist()]
        if valid_vertex.any()
        else [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]
    )
    with open(f'{path}/meta.json', 'w') as f:
        json.dump({
            'scale': [1 / step_size, 1 / step_size],
            'bbox': bbox,
            'area_vx2': area_vx2,
            'area_cm2': area_vx2 * voxel_size_um ** 2 / 1.e8,
            'format': 'tifxyz',
            'type': 'seg',
            'uuid': uuid,
            'source': source,
        }, f, indent=4)
    return True

