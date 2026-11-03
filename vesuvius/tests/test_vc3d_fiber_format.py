"""Current object-store wire documents must be readable by Python consumers."""
import copy
import pytest
from vc3d_fiber_format import parse_vc3d_fiber_format


def document():
    return {
        "type": "vc3d_fiber", "version": 4,
        "uuid": "12345678-1234-1234-1234-123456789abc",
        "coordinates": {"volume_uuid": None, "scale_factor": 4},
        "control_points": [
            {"position": [1, 2, 3], "direction": [1, 0, 0],
             "display_normal": [0, 0, 1], "display_normal_source": "manual"},
            {"position": [4, 5, 6]},
        ],
        "line_points": [],
    }


def test_current_geometry_does_not_invent_tracing_provenance():
    source = document()
    original = copy.deepcopy(source)
    parsed = parse_vc3d_fiber_format(source)
    assert parsed.optimization_mode == "lasagna"
    assert parsed.control_points_xyz == ((1., 2., 3.), (4., 5., 6.))
    assert parsed.control_point_segments == (None, None)
    assert parsed.line_points_xyz == ()
    assert parsed.metadata["coordinates"] == source["coordinates"]
    assert source == original


@pytest.mark.parametrize("field,value", [
    ("direction", [0, 0, 0]), ("direction", [1, 2]),
    ("display_normal", [0, float("nan"), 1]),
    ("display_normal_source", "invalid"), ("obsolete", True),
])
def test_current_control_annotations_are_validated(field, value):
    source = document()
    source["control_points"][0][field] = value
    with pytest.raises(ValueError):
        parse_vc3d_fiber_format(source)


def test_explicit_bad_mode_and_final_span_still_fail():
    source = document()
    source["optimization_mode"] = "unknown"
    with pytest.raises(ValueError):
        parse_vc3d_fiber_format(source)
    source.pop("optimization_mode")
    source["control_points"][-1]["segment_to_next"] = {}
    with pytest.raises(ValueError, match="final control point"):
        parse_vc3d_fiber_format(source)
