import numpy as np


def bbox_wireframe_segments(bbox):
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    corners = np.asarray(
        [
            [z_min, y_min, x_min],
            [z_min, y_min, x_max],
            [z_min, y_max, x_min],
            [z_min, y_max, x_max],
            [z_max, y_min, x_min],
            [z_max, y_min, x_max],
            [z_max, y_max, x_min],
            [z_max, y_max, x_max],
        ],
        dtype=np.float32,
    )
    edge_pairs = (
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7),
    )
    return [corners[[i0, i1]] for i0, i1 in edge_pairs]


def show_cond_edge_bboxes_napari(
    points_zyx,
    edge_zyx,
    bboxes,
    voxelized_bboxes=None,
    point_downsample=8,
    point_size=1.0,
    edge_point_size=4.0,
    voxel_point_size=1.0,
):
    show_streamline_geometry_napari(
        points_zyx,
        edge_zyx,
        bboxes,
        voxelized_bboxes=voxelized_bboxes,
        point_downsample=point_downsample,
        point_size=point_size,
        edge_point_size=edge_point_size,
        voxel_point_size=voxel_point_size,
    )


def show_streamline_geometry_napari(
    points_zyx,
    edge_zyx,
    bboxes,
    voxelized_bboxes=None,
    integration_group=None,
    point_downsample=2,
    point_size=2.0,
    edge_point_size=4.0,
    voxel_point_size=1.0,
):
    try:
        import napari
    except Exception as exc:
        raise RuntimeError("napari is not available.") from exc

    viewer = napari.Viewer(ndisplay=3)

    points = np.asarray(points_zyx, dtype=np.float32)
    downsample = max(1, int(point_downsample))
    if downsample > 1:
        points = points[::downsample]
    if points.shape[0] > 0:
        viewer.add_points(
            points,
            name="tifxyz_points",
            size=float(point_size),
            face_color=[0.0, 0.8, 0.2, 0.8],
        )

    edge = np.asarray(edge_zyx, dtype=np.float32)
    edge_valid = np.isfinite(edge).all(axis=1) & ~(edge == -1).all(axis=1)
    edge = edge[edge_valid]
    if edge.shape[0] > 0:
        viewer.add_points(
            edge,
            name="conditioning_edge",
            size=float(edge_point_size),
            face_color=[1.0, 0.2, 0.0, 1.0],
        )

    voxel_points = []
    for item in voxelized_bboxes or ():
        voxels = np.asarray(item.get("voxels"))
        if voxels.ndim != 3 or not voxels.any():
            continue
        z_min, _, y_min, _, x_min, _ = item["bbox"]
        coords = np.argwhere(voxels > 0).astype(np.float32, copy=False)
        coords += np.asarray([z_min, y_min, x_min], dtype=np.float32)
        voxel_points.append(coords)
    if voxel_points:
        merged_voxel_points = np.concatenate(voxel_points, axis=0)
        merged_voxel_points = np.unique(merged_voxel_points, axis=0)
        viewer.add_points(
            merged_voxel_points,
            name="upsampled_voxelized_surface",
            size=float(voxel_point_size),
            face_color=[1.0, 0.85, 0.05, 0.65],
        )

    segments = []
    for bbox in bboxes:
        segments.extend(bbox_wireframe_segments(bbox))
    if segments:
        viewer.add_shapes(
            segments,
            shape_type="path",
            edge_color=[0.0, 0.45, 1.0, 0.9],
            edge_width=1,
            face_color="transparent",
            name="cond_edge_bboxes",
            opacity=0.9,
        )

    if integration_group is not None and "points_zyx" in integration_group:
        traces = np.asarray(integration_group["points_zyx"], dtype=np.float32)
        active_mask = np.asarray(integration_group["active_mask"], dtype=bool)
        path_segments = []
        if traces.ndim == 3 and traces.shape[0] > 1:
            for point_idx in range(traces.shape[1]):
                valid_steps = np.flatnonzero(active_mask[:, point_idx])
                if valid_steps.size < 2:
                    continue
                path = traces[valid_steps, point_idx, :]
                if np.isfinite(path).all():
                    path_segments.append(path)
        if path_segments:
            viewer.add_shapes(
                path_segments,
                shape_type="path",
                edge_color=[1.0, 0.0, 0.8, 0.9],
                edge_width=1,
                face_color="transparent",
                name="integrated_streamlines",
                opacity=0.9,
            )

    napari.run()
