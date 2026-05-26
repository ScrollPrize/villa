import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from tifxyz import Patch


@dataclass(frozen=True)
class PointPatchLink:
    """Link between a point collection entry and a patch."""
    point_id: int
    collection_id: int
    collection_name: str
    point_zyx: List[float]
    ij_coords: List[float]
    distance: float
    winding_annotation: float


def load_point_collection(filename: str) -> Optional[Dict[int, Dict[str, Any]]]:
    """Load point collection from JSON file and return as dictionary.

    `kind` must be one of PCL_KINDS and is stamped onto each loaded pcl; it
    determines whether the pcl partakes in patch attachment (cross_patch) or is
    consumed as a free-floating ordered strip (unattached).
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)

        # Check version
        if not data.get("vc_pointcollections_json_version") == "1":
            print(f"Error: Unsupported JSON version in {filename}")
            return None

        collections = {}

        # Load collections
        collections_data = data.get("collections", {})
        for id_str, collection_data in collections_data.items():
            collection_id = int(id_str)
            collection = {
                "id": collection_id,
                "name": collection_data["name"],
                "points": {},
                "metadata": collection_data.get("metadata", {}),
                "color": collection_data.get("color", [0.0, 0.0, 0.0])
            }

            # Load points
            points_data = collection_data.get("points", {})
            for point_id_str, point_data in points_data.items():
                point_id = int(point_id_str)
                point = {
                    "id": point_id,
                    "collectionId": collection_id,
                    "p": point_data["p"],
                    "winding_annotation": point_data.get("wind_a") if point_data.get("wind_a") is not None else float('nan'),
                    "creation_time": point_data.get("creation_time", 0)
                }
                collection["points"][point_id] = point

            collections[collection_id] = collection

        total_points = sum(len(col["points"]) for col in collections.values())
        print(f"Loaded point collection with {len(collections)} collections ({total_points} points)")
        return collections

    except Exception as e:
        print(f"Error loading point collection from {filename}: {e}")
        return None


def _load_surface_index_backend():
    try:
        from vc import surface_index
        return surface_index
    except ImportError:
        pass

    try:
        import vc_surface_index
        return vc_surface_index
    except ImportError:
        return None


def can_use_surface_index_backend(patches: Dict[str, Patch]) -> bool:
    return _load_surface_index_backend() is not None


def _record_point_patch_link(
    links: Dict[str, List[PointPatchLink]],
    collection: Dict[str, Any],
    collection_id: int,
    point_id: int,
    point: Dict[str, Any],
    patch_id: str,
    distance: float,
    ij_coords: List[float],
) -> None:
    point_zyx = np.asarray(point.get('zyx', point['p'][::-1]), dtype=np.float32).tolist()
    point['on_patch'] = {'id': patch_id, 'distance': float(distance), 'ij': list(ij_coords)}
    links.setdefault(patch_id, []).append(
        PointPatchLink(
            point_id=point_id,
            collection_id=collection_id,
            collection_name=collection['name'],
            point_zyx=point_zyx,
            ij_coords=list(ij_coords),
            distance=float(distance),
            winding_annotation=float(point['winding_annotation']),
        )
    )


def _process_point_collections_with_surface_index(
    patches: Dict[str, Patch],
    point_collections: Dict[int, Dict[str, Any]],
    tolerance: float,
    distance_scale: float,
) -> Optional[Dict[str, List[PointPatchLink]]]:
    surface_index = _load_surface_index_backend()
    if surface_index is None:
        return None

    surface_defs = []
    for patch_id, patch in patches.items():
        zyx = patch.zyxs.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        scale = patch.scale.detach().cpu().numpy() if hasattr(patch.scale, 'detach') else patch.scale
        surface_defs.append(surface_index.QuadSurface(patch_id, zyx, float(scale[0]), float(scale[1])))

    links: Dict[str, List[PointPatchLink]] = {}
    if not surface_defs or not point_collections:
        return links

    print(f'linking points to patches with vc.surface_index ({len(surface_defs)} patches)')
    index = surface_index.SurfacePatchIndex()
    index.rebuild(surface_defs, bbox_padding=tolerance, sampling_stride=1)

    for collection_id, collection in tqdm(point_collections.items(), 'linking points to patches'):
        for point_id, point in collection['points'].items():
            point_zyx = np.asarray(point.get('zyx', point['p'][::-1]), dtype=np.float32)
            hit = index.locate_xyz(point_zyx[::-1].copy(), tolerance)
            if hit is None:
                continue
            _record_point_patch_link(
                links,
                collection,
                collection_id,
                point_id,
                point,
                hit['id'],
                float(hit['distance']) / distance_scale,
                hit['ij'],
            )

    return links


def _process_point_collections(
    patches: Dict[str, Patch],
    point_collections: Dict[int, Dict[str, Any]],
    tolerance: float = 10.0,
    surface_index_tolerance: Optional[float] = None,
    distance_scale: float = 1.0,
) -> Dict[str, List[PointPatchLink]]:
    """Process point collections and link them to patches."""
    if surface_index_tolerance is not None:
        links = _process_point_collections_with_surface_index(
            patches, point_collections, surface_index_tolerance, distance_scale
        )
        if links is not None:
            return links

    links: Dict[str, List[PointPatchLink]] = {}
    device = torch.device('cpu')
    tolerance_t = torch.tensor(tolerance, device=device, dtype=torch.float32)

    for collection_id, collection in tqdm(point_collections.items(), 'linking points to patches'):
        for point_id, point in collection['points'].items():
            point_zyx = torch.as_tensor(point.get('zyx', point['p'][::-1]), dtype=torch.float32, device=device)

            nearest_patch_id = None
            nearest_distance = torch.tensor(float('inf'), device=device)
            nearest_ij = None

            for patch_id, patch in patches.items():
                ij_coord, distance = patch.project(point_zyx)
                if distance < nearest_distance and distance <= tolerance_t:
                    nearest_distance = distance
                    nearest_patch_id = patch_id
                    nearest_ij = ij_coord

            if nearest_patch_id:
                _record_point_patch_link(
                    links,
                    collection,
                    collection_id,
                    point_id,
                    point,
                    nearest_patch_id,
                    float(nearest_distance.cpu().item()),
                    nearest_ij.tolist(),
                )

    return links


def normalise_pcl_winding_annotations(point_collections):
    # Per-pcl: if every point has a winding annotation, leave alone; if none has one, set them all to 0;
    # if mixed, print a warning and strip the unannotated points.
    for pcl_id, pcl in point_collections.items():
        points = pcl['points']
        annotated = [pid for pid, p in points.items() if np.isfinite(p['winding_annotation'])]
        unannotated = [pid for pid, p in points.items() if not np.isfinite(p['winding_annotation'])]
        if not unannotated:
            continue
        if not annotated:
            for p in points.values():
                p['winding_annotation'] = 0.0
            continue
        print(
            f'WARNING: pcl {pcl_id} ({pcl.get("name", "?")}) has mixed winding-number annotations: '
            f'{len(annotated)} annotated, {len(unannotated)} missing — stripping unannotated points'
        )
        for pid in unannotated:
            del points[pid]


