"""Threshold-connectivity diagnostic for one frozen pair of thick fibers."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from numcodecs.registry import get_codec
from scipy import ndimage
from skimage.morphology import skeletonize

from lasagna.inference_provenance import atomic_write
from .embedding import nearest_polyline_points


_CONN6 = ndimage.generate_binary_structure(3, 1)
_CONN26 = ndimage.generate_binary_structure(3, 3)


class LocalZarrV2:
    """Small, bounded reader for the immutable local V2 arrays used here.

    Reading chunks directly avoids making this diagnostic depend on Zarr's
    asynchronous store machinery.  The accepted metadata is deliberately
    narrow and validated rather than silently supporting a second data path.
    """

    def __init__(self, path):
        self.path = Path(path)
        meta = json.loads((self.path / ".zarray").read_text())
        if (meta.get("zarr_format") != 2 or meta.get("dimension_separator") != "/"
                or meta.get("order") != "C" or meta.get("filters") not in (None, [])):
            raise ValueError("pair connectivity requires a C-order, slash-separated local Zarr V2 array")
        self.shape = np.asarray(meta["shape"], dtype=np.int64)
        self.chunks = np.asarray(meta["chunks"], dtype=np.int64)
        self.dtype = np.dtype(meta["dtype"])
        self.fill = np.asarray(meta.get("fill_value", 0), dtype=self.dtype).item()
        self.codec = get_codec(meta["compressor"]) if meta.get("compressor") else None

    def read(self, begin_zyx, end_zyx):
        begin = np.asarray(begin_zyx, dtype=np.int64)
        end = np.asarray(end_zyx, dtype=np.int64)
        if begin.shape != (3,) or end.shape != (3,) or np.any(begin < 0) or np.any(end > self.shape) or np.any(end <= begin):
            raise ValueError("invalid local Zarr crop")
        result = np.full(tuple(end - begin), self.fill, dtype=self.dtype)
        first, last = begin // self.chunks, (end - 1) // self.chunks
        for key in itertools.product(*(range(int(a), int(b) + 1) for a, b in zip(first, last))):
            chunk_begin = np.asarray(key) * self.chunks
            chunk_end = np.minimum(chunk_begin + self.chunks, self.shape)
            path = self.path.joinpath(*(str(v) for v in key))
            if not path.is_file():
                continue
            payload = path.read_bytes()
            if self.codec is not None:
                payload = self.codec.decode(payload)
            chunk = np.frombuffer(payload, dtype=self.dtype).reshape(tuple(chunk_end - chunk_begin))
            source_begin = np.maximum(begin, chunk_begin)
            source_end = np.minimum(end, chunk_end)
            src = tuple(slice(int(a - c), int(b - c)) for a, b, c in zip(source_begin, source_end, chunk_begin))
            dst = tuple(slice(int(a - c), int(b - c)) for a, b, c in zip(source_begin, source_end, begin))
            result[dst] = chunk[src]
        return result


def _manifest_group(manifest_path, name):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    group = manifest["groups"][name]
    return manifest_path.parent / group["zarr"], float(1 << int(group["scaledown"]))


def _curve_arcs(points):
    points = np.asarray(points, dtype=np.float64)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def _resample_curve(points, spacing):
    points = np.asarray(points, dtype=np.float64)
    arcs = _curve_arcs(points)
    sample_arcs = np.arange(0.0, arcs[-1], spacing)
    if not len(sample_arcs) or sample_arcs[-1] != arcs[-1]:
        sample_arcs = np.r_[sample_arcs, arcs[-1]]
    result = np.column_stack([np.interp(sample_arcs, arcs, points[:, axis]) for axis in range(3)])
    tangent = np.gradient(result, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    return sample_arcs, result, tangent


def _compact_normal(nx, ny):
    x = (np.asarray(nx, dtype=np.float64) - 128.0) / 127.0
    y = (np.asarray(ny, dtype=np.float64) - 128.0) / 127.0
    z = np.sqrt(np.maximum(0.0, 1.0 - x*x - y*y))
    value = np.stack([x, y, z], axis=-1)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12)


def _sample_scalar(volume, points_base_xyz, origin_zyx, scale, order):
    xyz = np.asarray(points_base_xyz, dtype=np.float64) / scale
    coordinates = np.vstack([xyz[:, 2] - origin_zyx[0], xyz[:, 1] - origin_zyx[1], xyz[:, 0] - origin_zyx[2]])
    return ndimage.map_coordinates(volume, coordinates, order=order, mode="constant", cval=0.0,
                                    prefilter=order > 1)


def _sample_normals(nx, ny, points, origin_zyx, scale):
    # Compact normals have a canonical +z sign, so interpolating their encoded
    # components is stable in this local sheet neighborhood.
    return _compact_normal(_sample_scalar(nx, points, origin_zyx, scale, 1),
                           _sample_scalar(ny, points, origin_zyx, scale, 1))


def endpoint_ellipsoids(shape, origin_zyx, scale, endpoints_xyz, normals_xyz,
                        normal_radius_base, tangential_factor):
    """Return exact lattice masks for the replay metric's endpoint ellipsoids."""
    masks = []
    tangent_radius = normal_radius_base * tangential_factor
    for point, normal in zip(np.asarray(endpoints_xyz), np.asarray(normals_xyz)):
        center_zyx = point[::-1] / scale - origin_zyx
        radius_voxels = tangent_radius / scale
        low = np.maximum(0, np.floor(center_zyx - radius_voxels).astype(int))
        high = np.minimum(shape, np.ceil(center_zyx + radius_voxels).astype(int) + 1)
        local = np.indices(tuple(high-low), dtype=np.float64)
        lattice_zyx = np.moveaxis(local, 0, -1) + low
        delta_xyz = (lattice_zyx[..., ::-1] - center_zyx[::-1]) * scale
        along_normal = np.sum(delta_xyz * normal, axis=-1)
        tangent2 = np.maximum(0.0, np.sum(delta_xyz*delta_xyz, axis=-1) - along_normal*along_normal)
        inside = (along_normal / normal_radius_base)**2 + tangent2 / tangent_radius**2 <= 1.0
        mask = np.zeros(shape, dtype=bool)
        mask[tuple(slice(int(a), int(b)) for a, b in zip(low, high))] = inside
        masks.append(mask)
    return masks


def _ball(radius):
    if radius == 0:
        return np.ones((1, 1, 1), dtype=bool)
    grid = np.indices((2*radius+1,)*3) - radius
    return np.sum(grid*grid, axis=0) <= radius*radius


def connecting_component(mask, endpoint_masks, connectivity=_CONN26):
    labels, count = ndimage.label(mask, structure=connectivity)
    common = None
    hits = []
    for endpoint in endpoint_masks:
        values = np.unique(labels[endpoint & (labels != 0)])
        hits.append(values.tolist())
        common = set(values) if common is None else common.intersection(values)
    if not common:
        return None, hits
    sizes = np.bincount(labels.ravel())
    selected = min(common, key=lambda value: (sizes[value], value))
    return labels == selected, hits


def _closed_mask(mask, settings):
    iterations = int(settings.get("closing_iterations", 0))
    if iterations < 0:
        raise ValueError("closing_iterations must be non-negative")
    if not iterations:
        return mask
    # One iteration uses the full 3x3x3 neighborhood. Repeating it grows and
    # shrinks by one Chebyshev voxel per iteration.
    return ndimage.binary_closing(mask, structure=_CONN26, iterations=iterations)


def _transforms(presence, settings):
    yield {"kind": "raw", "sigma": 0.0, "amount": 0.0}, presence
    for sigma in settings["gaussian_sigmas_prediction"]:
        blurred = ndimage.gaussian_filter(presence, float(sigma), mode="nearest")
        yield {"kind": "gaussian", "sigma": float(sigma), "amount": 0.0}, blurred
    for sigma in settings["unsharp_sigmas_prediction"]:
        blurred = ndimage.gaussian_filter(presence, float(sigma), mode="nearest")
        for amount in settings["unsharp_amounts"]:
            sharpened = np.clip(presence + float(amount)*(presence-blurred), 0.0, 1.0)
            yield {"kind": "unsharp", "sigma": float(sigma), "amount": float(amount)}, sharpened


def search_connectivity(presence, endpoint_masks, settings, domain=None):
    """Grid search, ranked by final target-component volume."""
    exact = settings.get("threshold_search") == "exact_u8"
    if exact:
        jobs = [(transform, values, int(radius))
                for transform, values in _transforms(presence, settings)
                for radius in settings["dilation_radii_prediction"]]
    else:
        jobs = [(transform, values, int(threshold_u8))
                for transform, values in _transforms(presence, settings)
                for threshold_u8 in settings["thresholds_u8"]]

    def evaluate(job):
        transform, values, parameter = job
        rows = []
        radii = [parameter] if exact else [int(v) for v in settings["dilation_radii_prediction"]]

        def candidate(threshold_u8, radius):
            raw = values >= float(threshold_u8)/255.0
            if domain is not None:
                raw &= domain
            distances = ndimage.distance_transform_edt(~raw) if radius else None
            dilated = raw if radius == 0 else distances <= radius
            if domain is not None:
                dilated &= domain
            base_component, _ = connecting_component(dilated, endpoint_masks)
            if base_component is None:
                return raw, dilated, None, None, None, []
            # Closing is a refinement of the selected two-fiber support. If it
            # were applied before component selection, nearby threshold islands
            # could be annexed merely because they are within the closing span.
            closed = _closed_mask(base_component, settings)
            if domain is not None:
                closed &= domain
            component, endpoint_hits = connecting_component(closed, endpoint_masks)
            return raw, dilated, base_component, closed, component, endpoint_hits

        for radius in radii:
            if exact:
                # Feasibility is monotone as the uint8 threshold falls.  The
                # highest feasible threshold also minimizes this component for
                # the fixed transform/radius family.
                low, high, threshold_u8 = 1, 255, None
                while low <= high:
                    middle = (low + high)//2
                    _, _, _, _, component, _ = candidate(middle, radius)
                    if component is None:
                        high = middle - 1
                    else:
                        threshold_u8 = middle
                        low = middle + 1
                if threshold_u8 is None:
                    continue
            else:
                threshold_u8 = parameter
            raw, dilated, base_component, closed, component, endpoint_hits = candidate(threshold_u8, radius)
            if component is None:
                continue
            raw_component = raw & component
            dilation_added = dilated & ~raw & component
            closing_added = closed & ~base_component & component
            closing_removed = base_component & ~closed
            original_values = presence[component]
            boundary = bool(component[0].any() or component[-1].any() or component[:, 0].any()
                            or component[:, -1].any() or component[:, :, 0].any()
                            or component[:, :, -1].any())
            region_counts = [int(np.count_nonzero(component & m)) for m in endpoint_masks]
            rows.append(dict(transform=transform, threshold_u8=threshold_u8,
                dilation_radius_prediction=radius, component_voxels=int(component.sum()),
                closing_iterations=int(settings.get("closing_iterations", 0)),
                raw_component_voxels=int(raw_component.sum()),
                dilation_added_voxels=int(dilation_added.sum()),
                closing_added_voxels=int(closing_added.sum()),
                closing_removed_voxels=int(closing_removed.sum()),
                total_threshold_voxels=int(raw.sum()),
                total_dilated_voxels=int(dilated.sum()),
                total_closed_voxels=int(closed.sum()), boundary_contact=boundary,
                required_region_component_voxels=region_counts,
                required_region_label_choices=[len(h) for h in endpoint_hits],
                original_presence_mean=float(original_values.mean()),
                original_presence_p05=float(np.quantile(original_values, .05)),
                original_presence_min=float(original_values.min())))
        return rows

    with ThreadPoolExecutor(max_workers=int(settings.get("search_workers", 8))) as pool:
        feasible = [row for rows in pool.map(evaluate, jobs) for row in rows]
    if not feasible:
        raise ValueError("no searched threshold/sharpen/dilation setting connects all endpoint ellipsoids")
    feasible.sort(key=lambda r: (r["boundary_contact"], r["component_voxels"],
                                 r["dilation_added_voxels"], -r["original_presence_mean"],
                                 r["transform"]["kind"], r["transform"]["sigma"],
                                 r["transform"]["amount"], -r["threshold_u8"],
                                 r["dilation_radius_prediction"]))
    return feasible


def _materialize_choice(presence, endpoint_masks, choice, domain=None):
    kind, sigma, amount = (choice["transform"][k] for k in ("kind", "sigma", "amount"))
    if kind == "raw":
        values = presence
    else:
        blurred = ndimage.gaussian_filter(presence, sigma, mode="nearest")
        values = blurred if kind == "gaussian" else np.clip(presence + amount*(presence-blurred), 0, 1)
    raw = values >= choice["threshold_u8"]/255.0
    if domain is not None:
        raw &= domain
    radius = choice["dilation_radius_prediction"]
    dilated = ndimage.binary_dilation(raw, structure=_ball(radius)) if radius else raw
    if domain is not None:
        dilated &= domain
    base_component, _ = connecting_component(dilated, endpoint_masks)
    if base_component is None:
        raise RuntimeError("selected pre-closing connectivity result is not reproducible")
    closed = _closed_mask(base_component, choice)
    if domain is not None:
        closed &= domain
    component, _ = connecting_component(closed, endpoint_masks)
    if component is None:
        raise RuntimeError("selected connectivity result is not reproducible")
    return (raw & component, dilated & ~raw & component,
            closed & ~base_component & component, component)


def skeleton_topology(component, origin_zyx, scale, required_masks, long_arm_min_base=80.0,
                      junction_merge_max_base=64.0):
    """Skeletonize a 3D component and contract degree-two voxel chains.

    Lee's 3D thinning comes from scikit-image. The resulting voxel graph uses
    the same 26-neighbor adjacency as the connectivity experiment. Contiguous
    non-degree-two voxels are contracted into graph nodes; intervening chains
    become measured graph edges.
    """
    skeleton = skeletonize(component, method="lee")
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    degree = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode="constant", cval=0)
    critical = skeleton & (degree != 2)
    node_labels, node_count = ndimage.label(critical, structure=_CONN26)
    shape = np.asarray(skeleton.shape, dtype=np.int64)
    offsets = [np.asarray(v, dtype=np.int64) for v in itertools.product((-1, 0, 1), repeat=3)
               if v != (0, 0, 0)]

    def neighbors(coord):
        for offset in offsets:
            other = coord + offset
            if np.all(other >= 0) and np.all(other < shape) and skeleton[tuple(other)]:
                yield other

    def flat(coord):
        return int(np.ravel_multi_index(tuple(coord), tuple(shape)))

    nodes = []
    for label in range(1, node_count + 1):
        coordinates = np.argwhere(node_labels == label)
        centroid_zyx = coordinates.mean(axis=0)
        region_hits = [int(np.count_nonzero(mask[tuple(coordinates.T)])) for mask in required_masks]
        nodes.append(dict(id=label-1, voxel_count=int(len(coordinates)),
            centroid_base_xyz=((centroid_zyx + origin_zyx)[::-1]*scale).tolist(),
            raw_endpoint_voxels=int(np.count_nonzero(degree[tuple(coordinates.T)] <= 1)),
            raw_junction_voxels=int(np.count_nonzero(degree[tuple(coordinates.T)] >= 3)),
            required_region_skeleton_voxels=region_hits))

    visited = set()
    edges = []
    for label in range(1, node_count + 1):
        for start in np.argwhere(node_labels == label):
            for following in neighbors(start):
                if node_labels[tuple(following)] == label:
                    continue
                segment = tuple(sorted((flat(start), flat(following))))
                if segment in visited:
                    continue
                visited.add(segment)
                path = [start.copy(), following.copy()]
                previous, current = start, following
                while node_labels[tuple(current)] == 0:
                    choices = [candidate for candidate in neighbors(current)
                               if not np.array_equal(candidate, previous)]
                    if len(choices) != 1:
                        raise RuntimeError("degree-two skeleton chain has inconsistent adjacency")
                    subsequent = choices[0]
                    visited.add(tuple(sorted((flat(current), flat(subsequent)))))
                    path.append(subsequent.copy())
                    previous, current = current, subsequent
                end_label = int(node_labels[tuple(current)])
                path_array = np.asarray(path)
                length = float(np.linalg.norm(np.diff(path_array, axis=0), axis=1).sum()*scale)
                hits = [int(np.count_nonzero(mask[tuple(path_array.T)])) for mask in required_masks]
                edges.append(dict(id=len(edges), node_ids=[label-1, end_label-1],
                    length_base=length, voxel_count=int(len(path_array)),
                    start_base_xyz=((path_array[0] + origin_zyx)[::-1]*scale).tolist(),
                    end_base_xyz=((path_array[-1] + origin_zyx)[::-1]*scale).tolist(),
                    required_region_skeleton_voxels=hits))

    graph_degrees = np.zeros(node_count, dtype=np.int64)
    for edge in edges:
        a, b = edge["node_ids"]
        graph_degrees[a] += 1
        graph_degrees[b] += 1
    for node, graph_degree in zip(nodes, graph_degrees):
        node["graph_degree"] = int(graph_degree)
        node["kind"] = "terminal" if graph_degree == 1 else ("junction" if graph_degree >= 3 else "other")
    for edge in edges:
        kinds = [nodes[node]["kind"] for node in edge["node_ids"]]
        if "terminal" in kinds and "junction" in kinds:
            edge["kind"] = "terminal_arm"
        elif kinds == ["junction", "junction"]:
            edge["kind"] = "junction_bridge"
        else:
            edge["kind"] = "other"

    component_labels, component_count = ndimage.label(skeleton, structure=_CONN26)
    node_free_cycles = 0
    for label in range(1, component_count + 1):
        if not np.any(critical & (component_labels == label)):
            node_free_cycles += 1
    graph_component_count = max(0, component_count-node_free_cycles)
    cycle_rank = max(0, len(edges)-node_count+graph_component_count) + node_free_cycles
    terminals = [node for node in nodes if node["kind"] == "terminal"]
    junctions = [node for node in nodes if node["kind"] == "junction"]
    terminal_arms = [edge for edge in edges if edge["kind"] == "terminal_arm"]
    long_arms = [edge for edge in terminal_arms if edge["length_base"] >= long_arm_min_base]
    bridges = [edge for edge in edges if edge["kind"] == "junction_bridge"]
    two_y = (component_count == 1 and cycle_rank == 0 and len(terminals) == 4
             and len(junctions) == 2 and all(node["graph_degree"] == 3 for node in junctions)
             and len(terminal_arms) == 4 and len(bridges) == 1)

    # Digital 26-neighbor skeletons often contain tiny ladders of adjacent
    # junctions and parallel paths. Contract only short junction-to-junction
    # edges to expose the topology at a scale larger than those voxel effects.
    parent = list(range(node_count))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for edge in edges:
        a, b = edge["node_ids"]
        if (graph_degrees[a] >= 3 and graph_degrees[b] >= 3
                and edge["length_base"] <= junction_merge_max_base):
            union(a, b)
    groups = {}
    for node_id in range(node_count):
        groups.setdefault(find(node_id), []).append(node_id)
    group_ids = {root: index for index, root in enumerate(sorted(groups))}
    coarse_nodes = []
    for root in sorted(groups):
        members = groups[root]
        weights = np.asarray([nodes[node]["voxel_count"] for node in members], dtype=np.float64)
        centers = np.asarray([nodes[node]["centroid_base_xyz"] for node in members])
        coarse_nodes.append(dict(id=group_ids[root], original_node_ids=members,
            voxel_count=int(weights.sum()),
            centroid_base_xyz=np.average(centers, axis=0, weights=weights).tolist()))
    coarse_edges = []
    for edge in edges:
        a = group_ids[find(edge["node_ids"][0])]
        b = group_ids[find(edge["node_ids"][1])]
        if a == b:
            continue
        coarse_edges.append(dict(id=len(coarse_edges), node_ids=[a, b],
            source_edge_id=edge["id"], length_base=edge["length_base"],
            required_region_skeleton_voxels=edge["required_region_skeleton_voxels"]))
    coarse_degrees = np.zeros(len(coarse_nodes), dtype=np.int64)
    coarse_parent = list(range(len(coarse_nodes)))

    def coarse_find(value):
        while coarse_parent[value] != value:
            coarse_parent[value] = coarse_parent[coarse_parent[value]]
            value = coarse_parent[value]
        return value

    for edge in coarse_edges:
        a, b = edge["node_ids"]
        coarse_degrees[a] += 1; coarse_degrees[b] += 1
        ra, rb = coarse_find(a), coarse_find(b)
        if ra != rb:
            coarse_parent[max(ra, rb)] = min(ra, rb)
    for node, graph_degree in zip(coarse_nodes, coarse_degrees):
        node["graph_degree"] = int(graph_degree)
        node["kind"] = "terminal" if graph_degree == 1 else ("junction" if graph_degree >= 3 else "other")
    for edge in coarse_edges:
        kinds = [coarse_nodes[node]["kind"] for node in edge["node_ids"]]
        if "terminal" in kinds and "junction" in kinds:
            edge["kind"] = "terminal_arm"
        elif kinds == ["junction", "junction"]:
            edge["kind"] = "junction_bridge"
        else:
            edge["kind"] = "other"
    coarse_components = len({coarse_find(i) for i in range(len(coarse_nodes))}) if coarse_nodes else 0
    coarse_cycle_rank = max(0, len(coarse_edges)-len(coarse_nodes)+coarse_components)
    coarse_terminals = [node for node in coarse_nodes if node["kind"] == "terminal"]
    coarse_junctions = [node for node in coarse_nodes if node["kind"] == "junction"]
    coarse_terminal_arms = [edge for edge in coarse_edges if edge["kind"] == "terminal_arm"]
    coarse_bridges = [edge for edge in coarse_edges if edge["kind"] == "junction_bridge"]
    coarse_two_y = (coarse_components == 1 and coarse_cycle_rank == 0 and len(coarse_terminals) == 4
                    and len(coarse_junctions) == 2
                    and all(node["graph_degree"] == 3 for node in coarse_junctions)
                    and len(coarse_terminal_arms) == 4 and len(coarse_bridges) == 1)
    coarse = dict(junction_merge_max_base=float(junction_merge_max_base),
        graph_nodes=len(coarse_nodes), graph_edges=len(coarse_edges),
        graph_components=int(coarse_components), cycle_rank=int(coarse_cycle_rank),
        terminal_nodes=len(coarse_terminals), junction_nodes=len(coarse_junctions),
        terminal_arms=len(coarse_terminal_arms), junction_bridges=len(coarse_bridges),
        matches_two_y_topology=bool(coarse_two_y), nodes=coarse_nodes,
        edges=sorted(coarse_edges, key=lambda edge: edge["length_base"], reverse=True))

    # Suppress degree-two nodes as a final purely topological reduction. Edge
    # lengths are summed, retaining the physical extent of the resulting arm.
    reduced_nodes = {node["id"]: dict(node) for node in coarse_nodes}
    reduced_edges = [dict(edge, source_edge_ids=[edge["source_edge_id"]]) for edge in coarse_edges]
    while True:
        reduced_degrees = {node: 0 for node in reduced_nodes}
        for edge in reduced_edges:
            a, b = edge["node_ids"]
            reduced_degrees[a] += 1; reduced_degrees[b] += 1
        candidate = next((node for node in sorted(reduced_nodes)
                          if reduced_degrees[node] == 2), None)
        if candidate is None:
            break
        incident = [edge for edge in reduced_edges if candidate in edge["node_ids"]]
        if len(incident) != 2:
            break
        first, second = incident
        other_a = first["node_ids"][1] if first["node_ids"][0] == candidate else first["node_ids"][0]
        other_b = second["node_ids"][1] if second["node_ids"][0] == candidate else second["node_ids"][0]
        combined_hits = (np.asarray(first["required_region_skeleton_voxels"], dtype=np.int64)
                         + np.asarray(second["required_region_skeleton_voxels"], dtype=np.int64))
        reduced_edges = [edge for edge in reduced_edges if edge is not first and edge is not second]
        reduced_edges.append(dict(node_ids=[other_a, other_b],
            source_edge_ids=first["source_edge_ids"]+second["source_edge_ids"],
            length_base=first["length_base"]+second["length_base"],
            required_region_skeleton_voxels=combined_hits.tolist()))
        del reduced_nodes[candidate]
    old_ids = sorted(reduced_nodes)
    reduced_id = {old: new for new, old in enumerate(old_ids)}
    reduced_node_list = []
    for old in old_ids:
        node = reduced_nodes[old]
        node["id"] = reduced_id[old]
        reduced_node_list.append(node)
    for edge_id, edge in enumerate(reduced_edges):
        edge["id"] = edge_id
        edge["node_ids"] = [reduced_id[node] for node in edge["node_ids"]]
    reduced_degrees = np.zeros(len(reduced_node_list), dtype=np.int64)
    reduced_parent = list(range(len(reduced_node_list)))

    def reduced_find(value):
        while reduced_parent[value] != value:
            reduced_parent[value] = reduced_parent[reduced_parent[value]]
            value = reduced_parent[value]
        return value

    for edge in reduced_edges:
        a, b = edge["node_ids"]
        reduced_degrees[a] += 1; reduced_degrees[b] += 1
        ra, rb = reduced_find(a), reduced_find(b)
        if ra != rb:
            reduced_parent[max(ra, rb)] = min(ra, rb)
    for node, graph_degree in zip(reduced_node_list, reduced_degrees):
        node["graph_degree"] = int(graph_degree)
        node["kind"] = "terminal" if graph_degree == 1 else ("junction" if graph_degree >= 3 else "other")
    for edge in reduced_edges:
        kinds = [reduced_node_list[node]["kind"] for node in edge["node_ids"]]
        if "terminal" in kinds and "junction" in kinds:
            edge["kind"] = "terminal_arm"
        elif kinds == ["junction", "junction"]:
            edge["kind"] = "junction_bridge"
        else:
            edge["kind"] = "other"
    reduced_components = (len({reduced_find(i) for i in range(len(reduced_node_list))})
                          if reduced_node_list else 0)
    reduced_cycle_rank = max(0, len(reduced_edges)-len(reduced_node_list)+reduced_components)
    reduced_terminals = [node for node in reduced_node_list if node["kind"] == "terminal"]
    reduced_junctions = [node for node in reduced_node_list if node["kind"] == "junction"]
    reduced_arms = [edge for edge in reduced_edges if edge["kind"] == "terminal_arm"]
    reduced_bridges = [edge for edge in reduced_edges if edge["kind"] == "junction_bridge"]
    reduced_two_y = (reduced_components == 1 and reduced_cycle_rank == 0
        and len(reduced_terminals) == 4 and len(reduced_junctions) == 2
        and all(node["graph_degree"] == 3 for node in reduced_junctions)
        and len(reduced_arms) == 4 and len(reduced_bridges) == 1)
    reduced_single_y = (reduced_components == 1 and reduced_cycle_rank == 0
        and len(reduced_terminals) == 3 and len(reduced_junctions) == 1
        and reduced_junctions[0]["graph_degree"] == 3 and len(reduced_arms) == 3)
    reduced = dict(graph_nodes=len(reduced_node_list), graph_edges=len(reduced_edges),
        graph_components=int(reduced_components), cycle_rank=int(reduced_cycle_rank),
        terminal_nodes=len(reduced_terminals), junction_nodes=len(reduced_junctions),
        terminal_arms=len(reduced_arms), junction_bridges=len(reduced_bridges),
        terminal_arm_lengths_base=sorted([edge["length_base"] for edge in reduced_arms], reverse=True),
        matches_single_y_topology=bool(reduced_single_y),
        matches_two_y_topology=bool(reduced_two_y), nodes=reduced_node_list,
        edges=sorted(reduced_edges, key=lambda edge: edge["length_base"], reverse=True))
    report = dict(method="skimage.morphology.skeletonize(method='lee')",
        adjacency=26, skeleton_voxels=int(skeleton.sum()),
        skeleton_components=int(component_count), graph_nodes=len(nodes), graph_edges=len(edges),
        cycle_rank=int(cycle_rank), node_free_cycle_components=int(node_free_cycles),
        terminal_nodes=len(terminals), junction_nodes=len(junctions),
        terminal_arms=len(terminal_arms), long_terminal_arms=len(long_arms),
        long_arm_min_base=float(long_arm_min_base), junction_bridges=len(bridges),
        matches_two_y_topology=bool(two_y),
        required_region_skeleton_voxels=[int(np.count_nonzero(skeleton & mask)) for mask in required_masks],
        terminal_arm_lengths_base=sorted([edge["length_base"] for edge in terminal_arms], reverse=True),
        nodes=nodes, edges=sorted(edges, key=lambda edge: edge["length_base"], reverse=True),
        coarse=coarse, reduced=reduced)
    return skeleton, report


def write_skeleton_graph_obj(path, skeleton, origin_zyx, scale, topology):
    """Write separate, viewer-safe OBJ files for exact and reduced graphs."""
    path = Path(path)
    reduced_path = path.with_name(path.stem + "_reduced.obj")
    marker_path = path.with_name(path.stem + "_markers.obj")
    coordinates = np.argwhere(skeleton)
    coordinate_to_vertex = {tuple(coord): i+1 for i, coord in enumerate(coordinates)}
    lines = ["# Exact 26-neighbor skeleton; base-volume XYZ coordinates.",
             "o skeleton_26_neighbor_graph"]
    for coord in coordinates:
        xyz = (coord + origin_zyx)[::-1]*scale
        lines.append(f"v {xyz[0]:.9g} {xyz[1]:.9g} {xyz[2]:.9g}")
    lines.append("g skeleton_edges")
    edge_count = 0
    for coord in coordinates:
        source = coordinate_to_vertex[tuple(coord)]
        for offset in itertools.product((-1, 0, 1), repeat=3):
            if offset == (0, 0, 0):
                continue
            neighbor = tuple(coord + np.asarray(offset))
            target = coordinate_to_vertex.get(neighbor)
            if target is not None and source < target:
                lines.append(f"l {source} {target}")
                edge_count += 1
    # Explicit point elements let MeshLab display every graph sample when
    # vertex rendering is enabled, without introducing any triangle geometry.
    lines.append("g skeleton_points")
    lines.extend(f"p {index}" for index in range(1, len(coordinates)+1))
    path.write_text("\n".join(lines)+"\n")

    reduced = topology["reduced"]
    reduced_lines = ["# Degree-two-suppressed graph; base-volume XYZ coordinates.",
                     "o reduced_skeleton_graph"]
    reduced_vertex = {}
    for node in reduced["nodes"]:
        reduced_vertex[node["id"]] = len(reduced_vertex)+1
        x, y, z = node["centroid_base_xyz"]
        reduced_lines.append(f"v {x:.9g} {y:.9g} {z:.9g}")
    reduced_lines.append("g reduced_edges")
    for edge in reduced["edges"]:
        a, b = (reduced_vertex[node] for node in edge["node_ids"])
        reduced_lines.append(f"l {a} {b}")
    reduced_lines.append("g reduced_points")
    reduced_lines.extend(f"p {index}" for index in range(1, len(reduced_vertex)+1))
    reduced_path.write_text("\n".join(reduced_lines)+"\n")

    marker_lines = ["# Optional octahedral markers only; no graph lines.",
                    "o reduced_graph_markers"]

    def octahedron(center, radius, name):
        start = sum(line.startswith("v ") for line in marker_lines)+1
        x, y, z = center
        vertices = [(x+radius,y,z),(x-radius,y,z),(x,y+radius,z),
                    (x,y-radius,z),(x,y,z+radius),(x,y,z-radius)]
        marker_lines.append(f"g {name}")
        marker_lines.extend(f"v {a:.9g} {b:.9g} {c:.9g}" for a,b,c in vertices)
        for a,b,c in ((0,2,4),(2,1,4),(1,3,4),(3,0,4),
                      (2,0,5),(1,2,5),(3,1,5),(0,3,5)):
            marker_lines.append(f"f {start+a} {start+b} {start+c}")

    for node in reduced["nodes"]:
        radius = 16.0 if node["kind"] == "junction" else 12.0
        octahedron(node["centroid_base_xyz"], radius,
                   f"reduced_{node['kind']}_{node['id']}")
    marker_path.write_text("\n".join(marker_lines)+"\n")
    return dict(path=str(path), reduced_path=str(reduced_path), marker_path=str(marker_path),
                skeleton_vertices=int(len(coordinates)), skeleton_edges=edge_count,
                reduced_nodes=len(reduced["nodes"]), reduced_edges=len(reduced["edges"]))


def write_reference_fiber_objs(output, fibers):
    """Write each CP-bounded reference fiber as a point-and-line-only OBJ."""
    reports = []
    for index, points in enumerate(fibers):
        path = Path(output) / f"reference_fiber_{index}.obj"
        lines = [f"# CP-bounded corrected reference fiber {index}; base-volume XYZ coordinates.",
                 f"o reference_fiber_{index}"]
        lines.extend(f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in points)
        lines.append("g reference_edges")
        lines.extend(f"l {i} {i+1}" for i in range(1, len(points)))
        lines.append("g reference_points")
        lines.extend(f"p {i}" for i in range(1, len(points)+1))
        path.write_text("\n".join(lines)+"\n")
        reports.append(dict(path=str(path), vertices=len(points), edges=len(points)-1,
                            start_base_xyz=points[0].tolist(), end_base_xyz=points[-1].tolist(),
                            length_base=float(_curve_arcs(points)[-1])))
    return reports


def fiber_corridor(shape, origin_zyx, scale, fibers, radius_base):
    """Finite union of Euclidean tubes around the two CP-bounded fibers."""
    seeds = np.zeros(shape, dtype=bool)
    for fiber in fibers:
        _, sampled, _ = _resample_curve(fiber, scale/2)
        index_xyz = (np.rint(sampled/scale).astype(np.int64)
                     - np.asarray(origin_zyx[::-1], dtype=np.int64))
        valid = np.all((index_xyz >= 0) & (index_xyz < np.asarray(shape)[::-1]), axis=1)
        index_zyx = index_xyz[valid, ::-1]
        seeds[index_zyx[:, 0], index_zyx[:, 1], index_zyx[:, 2]] = True
    return ndimage.distance_transform_edt(~seeds) <= radius_base/scale


def _frames(points, tangents, normals):
    horizontal = normals - tangents*np.sum(normals*tangents, axis=1, keepdims=True)
    length = np.linalg.norm(horizontal, axis=1)
    for i in range(len(horizontal)):
        if length[i] < 1e-8:
            seed = np.array([1.0, 0.0, 0.0]) if abs(tangents[i, 0]) < .9 else np.array([0.0, 1.0, 0.0])
            horizontal[i] = seed - tangents[i]*np.dot(seed, tangents[i])
        horizontal[i] /= np.linalg.norm(horizontal[i])
        if i and np.dot(horizontal[i], horizontal[i-1]) < 0:
            horizontal[i] *= -1
    vertical = np.cross(tangents, horizontal)
    vertical /= np.maximum(np.linalg.norm(vertical, axis=1, keepdims=True), 1e-12)
    return horizontal, vertical


def _strip(points, axis, offsets, volume, origin, scale, order):
    coordinates = points[None, :, :] + offsets[:, None, None]*axis[None, :, :]
    values = _sample_scalar(volume, coordinates.reshape(-1, 3), origin, scale, order)
    return values.reshape(len(offsets), len(points))


def _component_strip(points, axis, offsets, raw, dilation_added, closing_added, skeleton,
                     origin, scale):
    a = _strip(points, axis, offsets, raw.astype(np.uint8), origin, scale, 0) > .5
    b = _strip(points, axis, offsets, dilation_added.astype(np.uint8), origin, scale, 0) > .5
    c = _strip(points, axis, offsets, closing_added.astype(np.uint8), origin, scale, 0) > .5
    s = _strip(points, axis, offsets, skeleton.astype(np.uint8), origin, scale, 0) > .5
    image = np.zeros((*a.shape, 3), dtype=np.uint8)
    image[a] = (235, 235, 235)
    image[b] = (255, 170, 0)  # BGR: cyan dilation-only support.
    image[c] = (60, 200, 60)  # BGR: green closing-only support.
    image[s] = (255, 0, 255)  # BGR: magenta skeleton.
    return image


def _gray(values):
    image = np.clip(np.rint(values*255), 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def _decorate_center(image, merge_column=None):
    result = image.copy()
    cv2.line(result, (0, result.shape[0]//2), (result.shape[1]-1, result.shape[0]//2),
             (0, 220, 255), 1, cv2.LINE_AA)
    if merge_column is not None:
        cv2.line(result, (int(merge_column), 0), (int(merge_column), result.shape[0]-1),
                 (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(result, "M", (max(0, int(merge_column)-4), 10),
                    cv2.FONT_HERSHEY_SIMPLEX, .3, (255,0,255), 1)
    return result


def _section_mosaic(points, horizontal, vertical, presence, raw, dilation_added, closing_added,
                    skeleton, origin, scale,
                    normal_radius, tangent_radius, merge_index, count=12):
    indices = np.unique(np.r_[np.linspace(0, len(points)-1, count).round().astype(int), merge_index])
    xs = np.arange(-normal_radius, normal_radius+1e-9, 2.0)
    ys = np.arange(-tangent_radius, tangent_radius+1e-9, 2.0)
    presence_tiles, component_tiles = [], []
    for number, index in enumerate(indices):
        coords = (points[index][None, None, :] + xs[None, :, None]*horizontal[index]
                  + ys[:, None, None]*vertical[index])
        flat = coords.reshape(-1, 3)
        values = _sample_scalar(presence, flat, origin, scale, 1).reshape(len(ys), len(xs))
        r = _sample_scalar(raw.astype(np.uint8), flat, origin, scale, 0).reshape(len(ys), len(xs)) > .5
        d = _sample_scalar(dilation_added.astype(np.uint8), flat, origin, scale, 0).reshape(len(ys), len(xs)) > .5
        c = _sample_scalar(closing_added.astype(np.uint8), flat, origin, scale, 0).reshape(len(ys), len(xs)) > .5
        s = _sample_scalar(skeleton.astype(np.uint8), flat, origin, scale, 0).reshape(len(ys), len(xs)) > .5
        p = _gray(values)
        support = np.zeros((*r.shape, 3), dtype=np.uint8)
        support[r] = (235,235,235); support[d] = (255,170,0)
        support[c] = (60,200,60); support[s] = (255,0,255)
        for tile in (p, support):
            cv2.ellipse(tile, (len(xs)//2, len(ys)//2), (len(xs)//2, len(ys)//2),
                        0, 0, 360, (0,220,255), 1, cv2.LINE_AA)
            label = "M" if index == merge_index else str(number)
            cv2.putText(tile, label, (2, 10), cv2.FONT_HERSHEY_SIMPLEX, .3,
                        (255,0,255) if label == "M" else (0,220,255), 1)
        presence_tiles.append(p); component_tiles.append(support)
    return np.hstack(presence_tiles), np.hstack(component_tiles), len(indices)


def _labeled_rows(rows):
    width = max(image.shape[1] for _, image in rows)
    rendered = []
    for label, image in rows:
        row = np.zeros((image.shape[0]+22, width, 3), dtype=np.uint8)
        row[22:22+image.shape[0], :image.shape[1]] = image
        cv2.putText(row, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (220,220,220), 1, cv2.LINE_AA)
        rendered.append(row)
    return np.vstack(rendered)


def render_fiber(path, fiber_index, points, merge_point, presence, raw, dilation_added,
                 closing_added, skeleton, presence_origin, presence_scale,
                 normal_x, normal_y, normal_origin, normal_scale, settings):
    arcs, sampled, tangents = _resample_curve(points, settings["render_spacing_base"])
    normals = _sample_normals(normal_x, normal_y, sampled, normal_origin, normal_scale)
    horizontal, vertical = _frames(sampled, tangents, normals)
    merge_index = int(np.argmin(np.linalg.norm(sampled-merge_point, axis=1)))
    offsets = np.arange(-settings["render_half_width_base"], settings["render_half_width_base"]+1e-9,
                        settings["render_spacing_base"])
    p_top = _decorate_center(_gray(_strip(sampled, horizontal, offsets, presence,
                                          presence_origin, presence_scale, 1)), merge_index)
    p_side = _decorate_center(_gray(_strip(sampled, vertical, offsets, presence,
                                           presence_origin, presence_scale, 1)), merge_index)
    c_top = _decorate_center(_component_strip(sampled, horizontal, offsets, raw, dilation_added,
                                              closing_added, skeleton,
                                              presence_origin, presence_scale), merge_index)
    c_side = _decorate_center(_component_strip(sampled, vertical, offsets, raw, dilation_added,
                                               closing_added, skeleton,
                                               presence_origin, presence_scale), merge_index)
    p_sections, c_sections, section_count = _section_mosaic(sampled, horizontal, vertical, presence,
        raw, dilation_added, closing_added, skeleton, presence_origin, presence_scale,
        settings["normal_radius_base"],
        settings["normal_radius_base"]*settings["tangential_factor"], merge_index)
    labels = [
        (f"Presence / fiber {fiber_index} / top / 2 base voxels per pixel", p_top),
        ("Closed component + skeleton / top / white=raw, cyan=dilation, green=closing, magenta=skeleton", c_top),
        (f"Presence / fiber {fiber_index} / side / 2 base voxels per pixel", p_side),
        ("Closed component + skeleton / side / white=raw, cyan=dilation, green=closing, magenta=skeleton", c_side),
        ("Presence / aligned transverse slices", p_sections),
        ("Closed component + skeleton / aligned transverse slices", c_sections),
    ]
    image = _labeled_rows(labels)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"failed to write {path}")
    return dict(path=str(path), shape_yx=list(image.shape[:2]), arc_length_base=float(arcs[-1]),
                rows=[label for label, _ in labels], transverse_slice_count=section_count,
                merge_column=merge_index, merge_point_base_xyz=np.asarray(merge_point).tolist())


def run_pair_connectivity(workload):
    root = Path(workload["output_root"])
    pair = root / "pair"
    settings = dict(workload["pair_connectivity"])
    output = pair / settings.get("output_name", "connectivity_v1")
    output.mkdir(parents=False, exist_ok=False)
    evaluation = json.loads((root / "evaluation.json").read_text())
    if settings.get("fiber_paths"):
        from .fiber_pair_benchmark import build_reference_plan
        plan = build_reference_plan(evaluation, settings["fiber_paths"], workload["native_binary"],
            output / "reference_config.json", output / "reference_plan.json",
            settings.get("fiber_sha256"))
    else:
        plan = json.loads((pair / "reference_plan.json").read_text())
    fibers = plan["sources"][0]["fibers"]
    points = [np.asarray(f["runs"][0]["points_base_xyz"], dtype=np.float64) for f in fibers]
    if len(points) != 2 or any(len(f["runs"]) != 1 for f in fibers):
        raise ValueError("connectivity diagnostic requires exactly two single-span fibers")
    source = evaluation["sources"][0]
    presence_path, presence_scale = _manifest_group(source["fiber_manifest"], "presence")
    presence_array = LocalZarrV2(presence_path)
    all_points = np.concatenate(points)
    halo = float(settings["roi_halo_base"])
    lo_xyz = np.maximum(0, np.floor((all_points.min(axis=0)-halo)/presence_scale).astype(int))
    hi_xyz = np.minimum(presence_array.shape[::-1],
                        np.ceil((all_points.max(axis=0)+halo)/presence_scale).astype(int)+1)
    origin_zyx = lo_xyz[::-1]
    presence_u8 = presence_array.read(origin_zyx, hi_xyz[::-1])
    presence = presence_u8.astype(np.float32)/255.0

    nx_path, normal_scale = _manifest_group(source["normal_manifest"], "nx")
    ny_path, ny_scale = _manifest_group(source["normal_manifest"], "ny")
    if ny_scale != normal_scale:
        raise ValueError("normal channels have inconsistent scales")
    nx_array, ny_array = LocalZarrV2(nx_path), LocalZarrV2(ny_path)
    normal_halo = float(settings["render_half_width_base"])+2*normal_scale
    normal_lo_xyz = np.maximum(0, np.floor((all_points.min(axis=0)-normal_halo)/normal_scale).astype(int))
    normal_hi_xyz = np.minimum(nx_array.shape[::-1],
        np.ceil((all_points.max(axis=0)+normal_halo)/normal_scale).astype(int)+1)
    normal_origin = normal_lo_xyz[::-1]
    normal_x = nx_array.read(normal_origin, normal_hi_xyz[::-1])
    normal_y = ny_array.read(normal_origin, normal_hi_xyz[::-1])

    endpoints = np.asarray([points[0][0], points[0][-1], points[1][0], points[1][-1]])
    nearest, squared = nearest_polyline_points(points[0], points[1])
    closest = int(np.argmin(squared))
    merge_point = (points[0][closest] + nearest[closest])/2
    required_points = np.vstack([endpoints, merge_point])
    required_normals = _sample_normals(normal_x, normal_y, required_points, normal_origin, normal_scale)
    endpoint_masks = endpoint_ellipsoids(presence.shape, origin_zyx, presence_scale, endpoints,
        required_normals[:4], settings["normal_radius_base"], settings["tangential_factor"])
    merge_mask, = endpoint_ellipsoids(presence.shape, origin_zyx, presence_scale, [merge_point],
        required_normals[4:], settings["merge_radius_base"], settings["tangential_factor"])
    required_masks = endpoint_masks + [merge_mask]
    domain = fiber_corridor(presence.shape, origin_zyx, presence_scale, points,
                            settings["analysis_corridor_radius_base"])
    feasible = search_connectivity(presence, required_masks, settings, domain)
    selected = feasible[0]
    raw, dilation_added, closing_added, component = _materialize_choice(
        presence, required_masks, selected, domain)
    six_component, _ = connecting_component(component, required_masks, _CONN6)
    selected["connected_with_6_neighbors"] = six_component is not None
    overlap = np.zeros(presence.shape, dtype=np.uint8)
    for mask in endpoint_masks:
        overlap += mask
    selected.update(endpoint_region_voxels=[int(m.sum()) for m in endpoint_masks],
                    merge_region_voxels=int(merge_mask.sum()),
                    endpoint_region_overlap_voxels=int(np.count_nonzero(overlap > 1)))
    skeleton, topology = skeleton_topology(component, origin_zyx, presence_scale, required_masks,
        settings.get("topology_long_arm_min_base", 80.0),
        settings.get("topology_junction_merge_max_base", 64.0))
    renderings = [render_fiber(output / f"fiber_{i}_{'primary' if i == 0 else 'auxiliary'}.jpg",
        i, fiber, merge_point, presence, raw, dilation_added, closing_added, skeleton,
        origin_zyx, presence_scale, normal_x, normal_y, normal_origin, normal_scale, settings)
        for i, fiber in enumerate(points)]
    np.savez_compressed(output / "selected_component.npz", raw_threshold_component=raw,
                        dilation_added_component=dilation_added,
                        closing_added_component=closing_added, component=component,
                        skeleton=skeleton, analysis_corridor=domain,
                        presence_origin_zyx=origin_zyx, presence_scale_base=presence_scale)
    atomic_write(output / "skeleton_topology.json", topology)
    graph_obj = write_skeleton_graph_obj(output / "skeleton_graph.obj", skeleton,
                                         origin_zyx, presence_scale, topology)
    reference_objs = write_reference_fiber_objs(output, points)
    report = dict(format="fiber_pair_threshold_connectivity", version=2,
        interpretation="closed-support connectivity and skeleton-topology diagnostic; not a fiber assignment",
        source=dict(fiber_manifest=source["fiber_manifest"], normal_manifest=source["normal_manifest"],
                    presence_zarr=str(presence_path), presence_scale_base=presence_scale),
        roi=dict(origin_prediction_zyx=origin_zyx.tolist(), shape_zyx=list(presence.shape),
                 halo_base=halo, total_voxels=int(presence.size),
                 analysis_corridor_radius_base=settings["analysis_corridor_radius_base"],
                 analysis_corridor_voxels=int(domain.sum())),
        required_region_order=["primary_start", "primary_end", "auxiliary_start", "auxiliary_end", "merge"],
        endpoints_base_xyz=endpoints.tolist(), endpoint_normals_xyz=required_normals[:4].tolist(),
        merge_point_base_xyz=merge_point.tolist(), merge_normal_xyz=required_normals[4].tolist(),
        closest_fiber_distance_base=float(math.sqrt(squared[closest])),
        search=settings, feasible_count=len(feasible), selected=selected,
        pareto_preview=feasible[:min(30, len(feasible))], topology=topology,
        graph_obj=graph_obj, reference_objs=reference_objs, renderings=renderings,
        limitations=["The selected component proves support connectivity only, not the correct pairing.",
                     "Skeleton topology is resolution- and morphology-dependent.",
                     "The two frozen curves are incomplete labels; outside support is not called false-positive."])
    atomic_write(output / "result.json", report)
    lines = ["# Two-fiber threshold connectivity", "",
        "The same selected 26-connected component intersects all four native replay acceptance ellipsoids.",
        "This is a support-connectivity diagnostic, not a resolved fiber assignment.", "",
        f"- Selected transform: `{selected['transform']}`",
        f"- Threshold: `{selected['threshold_u8']}/255`",
        f"- Ball dilation: `{selected['dilation_radius_prediction']}` prediction voxels "
        f"(`{selected['dilation_radius_prediction']*presence_scale:g}` base voxels)",
        f"- 3D closing: `{selected['closing_iterations']}` iterations with a 3x3x3 neighborhood",
        f"- Final component: `{selected['component_voxels']}` voxels",
        f"- Raw threshold support in component: `{selected['raw_component_voxels']}` voxels",
        f"- Dilation-only support: `{selected['dilation_added_voxels']}` voxels",
        f"- Closing-only support: `{selected['closing_added_voxels']}` voxels",
        f"- 6-neighbor connected too: `{selected['connected_with_6_neighbors']}`",
        f"- ROI boundary contact: `{selected['boundary_contact']}`", "",
        "## Skeleton topology", "",
        f"- Skeleton voxels/components: `{topology['skeleton_voxels']}` / `{topology['skeleton_components']}`",
        f"- Graph nodes/edges/cycle rank: `{topology['graph_nodes']}` / `{topology['graph_edges']}` / `{topology['cycle_rank']}`",
        f"- Terminal nodes/junction nodes: `{topology['terminal_nodes']}` / `{topology['junction_nodes']}`",
        f"- Terminal arms/long terminal arms: `{topology['terminal_arms']}` / `{topology['long_terminal_arms']}`",
        f"- Junction bridges: `{topology['junction_bridges']}`",
        f"- Exact two-Y topology: `{topology['matches_two_y_topology']}`", "",
        f"After contracting junction-to-junction paths up to "
        f"`{topology['coarse']['junction_merge_max_base']:g}` base voxels:", "",
        f"- Coarse graph nodes/edges/cycle rank: `{topology['coarse']['graph_nodes']}` / "
        f"`{topology['coarse']['graph_edges']}` / `{topology['coarse']['cycle_rank']}`",
        f"- Coarse terminal/junction nodes: `{topology['coarse']['terminal_nodes']}` / "
        f"`{topology['coarse']['junction_nodes']}`",
        f"- Coarse exact two-Y topology: `{topology['coarse']['matches_two_y_topology']}`", "",
        "After also suppressing degree-two nodes:", "",
        f"- Reduced nodes/edges/cycle rank: `{topology['reduced']['graph_nodes']}` / "
        f"`{topology['reduced']['graph_edges']}` / `{topology['reduced']['cycle_rank']}`",
        f"- Reduced terminal/junction nodes: `{topology['reduced']['terminal_nodes']}` / "
        f"`{topology['reduced']['junction_nodes']}`",
        f"- Single-Y / two-Y topology: `{topology['reduced']['matches_single_y_topology']}` / "
        f"`{topology['reduced']['matches_two_y_topology']}`", "",
        f"- [Primary visualization]({Path(renderings[0]['path']).name})",
        f"- [Auxiliary visualization]({Path(renderings[1]['path']).name})",
        "- [Exact 26-neighbor skeleton OBJ (points + lines only)](skeleton_graph.obj)",
        "- [Reduced graph OBJ (points + lines only)](skeleton_graph_reduced.obj)",
        "- [Optional reduced-node marker mesh](skeleton_graph_markers.obj)",
        "- [Corrected reference fiber 0 OBJ](reference_fiber_0.obj)",
        "- [Corrected reference fiber 1 OBJ](reference_fiber_1.obj)",
        "- [Skeleton topology graph](skeleton_topology.json)",
        "- [Full result and top alternatives](result.json)", "",
        "White is raw threshold support, cyan is dilation-only support, green is closing-only support, "
        "and magenta is the Lee 3D skeleton."]
    (output / "README.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"output": str(output), "selected": selected,
                      "feasible_count": len(feasible), "renderings": renderings}, indent=2))
    return report
