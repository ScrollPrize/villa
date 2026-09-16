import numpy as np

from vesuvius.neural_tracing.fiber_trace_3d.fiber_pair_connectivity import (
    _closed_mask, connecting_component, endpoint_ellipsoids, fiber_corridor,
    search_connectivity, skeleton_topology, write_reference_fiber_objs,
    write_skeleton_graph_obj,
)


def test_endpoint_ellipsoid_has_declared_normal_and_tangential_radii():
    mask, = endpoint_ellipsoids((25, 25, 25), np.zeros(3), 1.0,
        [[12, 12, 12]], [[0, 0, 1]], 2.0, 2.0)
    # Array axes are z,y,x.  A z displacement is normal; x/y are tangential.
    assert mask[14, 12, 12]
    assert not mask[15, 12, 12]
    assert mask[12, 12, 16]
    assert not mask[12, 12, 17]


def test_corner_contact_is_26_connected_but_not_6_connected():
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1, 1, 1] = True
    mask[2, 2, 2] = True
    endpoints = [np.zeros_like(mask), np.zeros_like(mask)]
    endpoints[0][1, 1, 1] = True
    endpoints[1][2, 2, 2] = True
    component, _ = connecting_component(mask, endpoints)
    assert component is not None and component.sum() == 2
    from scipy import ndimage
    component6, _ = connecting_component(mask, endpoints, ndimage.generate_binary_structure(3, 1))
    assert component6 is None


def test_search_minimizes_final_component_not_just_raw_threshold_support():
    presence = np.zeros((5, 5, 9), dtype=np.float32)
    presence[2, 2, 1:8] = 0.5
    endpoints = [np.zeros_like(presence, dtype=bool), np.zeros_like(presence, dtype=bool)]
    endpoints[0][2, 2, 1] = True
    endpoints[1][2, 2, 7] = True
    settings = dict(gaussian_sigmas_prediction=[], unsharp_sigmas_prediction=[],
                    unsharp_amounts=[], thresholds_u8=[120, 200],
                    dilation_radii_prediction=[0, 3])
    feasible = search_connectivity(presence, endpoints, settings)
    assert feasible[0]["threshold_u8"] == 120
    assert feasible[0]["dilation_radius_prediction"] == 0
    assert feasible[0]["component_voxels"] == 7


def test_exact_threshold_search_finds_highest_connected_uint8_cutoff():
    presence = np.zeros((3, 3, 6), dtype=np.float32)
    presence[1, 1, :] = np.array([.9, .7, .51, .6, .8, .95])
    endpoints = [np.zeros_like(presence, dtype=bool), np.zeros_like(presence, dtype=bool)]
    endpoints[0][1, 1, 0] = True
    endpoints[1][1, 1, 5] = True
    settings = dict(gaussian_sigmas_prediction=[], unsharp_sigmas_prediction=[],
                    unsharp_amounts=[], thresholds_u8=[], threshold_search="exact_u8",
                    dilation_radii_prediction=[0])
    feasible = search_connectivity(presence, endpoints, settings)
    assert feasible[0]["threshold_u8"] == 130
    assert feasible[0]["component_voxels"] == 6


def test_search_reports_infeasible_grid():
    presence = np.zeros((3, 3, 7), dtype=np.float32)
    endpoints = [np.zeros_like(presence, dtype=bool), np.zeros_like(presence, dtype=bool)]
    endpoints[0][1, 1, 0] = True
    endpoints[1][1, 1, 6] = True
    settings = dict(gaussian_sigmas_prediction=[], unsharp_sigmas_prediction=[],
                    unsharp_amounts=[], thresholds_u8=[255], dilation_radii_prediction=[0])
    try:
        search_connectivity(presence, endpoints, settings)
    except ValueError as error:
        assert "no searched" in str(error)
    else:
        raise AssertionError("infeasible search unexpectedly succeeded")


def test_fiber_corridor_is_finite_and_bounded():
    line = np.array([[3., 3., 3.], [3., 3., 7.]])
    corridor = fiber_corridor((12, 12, 12), np.zeros(3), 1.0, [line], 2.0)
    assert corridor[3, 3, 3]
    assert corridor[7, 3, 3]
    assert corridor[5, 5, 3]
    assert not corridor[3, 3, 0]
    assert not corridor[3, 3, 10]


def test_four_iteration_3d_closing_fills_internal_gap():
    mask = np.zeros((17, 17, 17), dtype=bool)
    mask[4:13, 4:13, 4:13] = True
    mask[8, 8, 8] = False
    closed = _closed_mask(mask, {"closing_iterations": 4})
    assert closed[8, 8, 8]
    assert np.all(closed[4:13, 4:13, 4:13])


def test_closing_refines_selected_component_without_annexing_nearby_island():
    presence = np.zeros((15, 15, 15), dtype=np.float32)
    presence[7, 7, 3:12] = .8
    presence[7, 10, 3:12] = .8
    endpoints = [np.zeros_like(presence, dtype=bool), np.zeros_like(presence, dtype=bool)]
    endpoints[0][7, 7, 3] = True
    endpoints[1][7, 7, 11] = True
    settings = dict(gaussian_sigmas_prediction=[], unsharp_sigmas_prediction=[],
        unsharp_amounts=[], thresholds_u8=[], threshold_search="exact_u8",
        dilation_radii_prediction=[0], closing_iterations=2)
    selected = search_connectivity(presence, endpoints, settings)[0]
    assert selected["total_threshold_voxels"] == 18
    assert selected["component_voxels"] == 9
    assert selected["total_closed_voxels"] == 9


def test_skeleton_topology_recognizes_two_y_graph():
    component = np.zeros((9, 9, 15), dtype=bool)
    component[4, 4, 4:11] = True
    component[4, 1:8, 4] = True
    component[4, 1:8, 10] = True
    _, report = skeleton_topology(component, np.zeros(3), 1.0,
        [np.zeros_like(component)]*5, long_arm_min_base=2.0,
        junction_merge_max_base=2.0)
    assert report["skeleton_components"] == 1
    assert report["terminal_nodes"] == 4
    assert report["junction_nodes"] == 2
    assert report["terminal_arms"] == 4
    assert report["long_terminal_arms"] == 4
    assert report["junction_bridges"] == 1
    assert report["cycle_rank"] == 0
    assert report["matches_two_y_topology"]
    assert report["coarse"]["matches_two_y_topology"]
    assert report["reduced"]["matches_two_y_topology"]


def test_skeleton_graph_obj_contains_full_and_reduced_graph(tmp_path):
    component = np.zeros((9, 9, 15), dtype=bool)
    component[4, 4, 4:11] = True
    component[4, 1:8, 4] = True
    component[4, 1:8, 10] = True
    skeleton, topology = skeleton_topology(component, np.zeros(3), 2.0,
        [np.zeros_like(component)]*5, junction_merge_max_base=3.0)
    report = write_skeleton_graph_obj(tmp_path / "graph.obj", skeleton,
                                      np.zeros(3), 2.0, topology)
    text = (tmp_path / "graph.obj").read_text()
    assert report["skeleton_vertices"] == int(skeleton.sum())
    assert report["reduced_nodes"] == 6
    assert "g skeleton_edges" in text
    assert "g skeleton_points" in text
    assert "f " not in text
    assert "g reduced_edges" in (tmp_path / "graph_reduced.obj").read_text()
    assert "g reduced_junction_" in (tmp_path / "graph_markers.obj").read_text()


def test_reference_fiber_objs_are_point_and_line_only(tmp_path):
    fibers = [np.array([[1., 2., 3.], [4., 6., 3.], [5., 6., 3.]]),
              np.array([[7., 8., 9.], [8., 8., 9.]])]
    reports = write_reference_fiber_objs(tmp_path, fibers)
    text = (tmp_path / "reference_fiber_0.obj").read_text()
    assert len(reports) == 2
    assert reports[0]["vertices"] == 3
    assert reports[0]["edges"] == 2
    assert reports[0]["length_base"] == 6.0
    assert text.count("\nv ") == 3
    assert text.count("\nl ") == 2
    assert "\nf " not in text
