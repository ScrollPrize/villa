import logging

import numpy as np
from build123d import *  # type: ignore

logger = logging.getLogger(__name__)


def _divider_points(lining_outer_radius, wall_thickness_mm):
    """Ends and middles of the divider curve's two three-point arcs."""
    return [
        (-lining_outer_radius + wall_thickness_mm, 0),
        (-lining_outer_radius / 2, lining_outer_radius / 5),
        (0, 0),
        (lining_outer_radius / 2, -lining_outer_radius / 5),
        (lining_outer_radius - wall_thickness_mm, 0),
    ]


def divider_curve(lining_outer_radius, wall_thickness_mm):
    """The curve that, once extruded, splits the two case halves."""
    pts = _divider_points(lining_outer_radius, wall_thickness_mm)
    ln1 = ThreePointArc(pts[0], pts[1], pts[2])
    ln2 = ThreePointArc(pts[2], pts[3], pts[4])
    return ln1 + ln2


def divider_y(x, lining_outer_radius, wall_thickness_mm):
    """y of the divider curve at each x (an array), 0 beyond its ends as for the
    divider solid."""
    pts = _divider_points(lining_outer_radius, wall_thickness_mm)
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    for (x0, y0), (x1, y1), (x2, y2) in (pts[0:3], pts[2:5]):
        # Circle through the three points, then the branch containing the middle one.
        a = np.array([[x1 - x0, y1 - y0], [x2 - x0, y2 - y0]])
        b = 0.5 * np.array(
            [x1**2 - x0**2 + y1**2 - y0**2, x2**2 - x0**2 + y2**2 - y0**2]
        )
        cx, cy = np.linalg.solve(a, b)
        r2 = (x0 - cx) ** 2 + (y0 - cy) ** 2
        on = (x >= x0) & (x <= x2)
        y[on] = cy + np.sign(y1 - cy) * np.sqrt(np.maximum(r2 - (x[on] - cx) ** 2, 0))
    return y


def build_divider_solid(
    lining_outer_radius,
    case_max_radius,
    wall_thickness_mm,
    rotation_deg=0,
    cap_side_deg=0,
    cap_half_width=None,
):
    """A solid that can be intersected or subtracted from a case to produce the right or left halves, respectively.

    With rotation_deg, the divider curve is turned that far (anticlockwise) about Z.
    Past the lining, the split then runs radially to a wall thickness beyond it and on,
    straight, to the middle of the caps' side at cap_side_deg (a multiple of 90 within
    45 degrees of rotation_deg) cap_half_width from the axis, and out along that side's
    normal, where the caps' bolt protrusions straddle it. Along the divider's turned
    direction it only ever advances, so the halves still part along its normal.
    """
    if rotation_deg != cap_side_deg:
        return _build_turned_divider_solid(
            lining_outer_radius,
            case_max_radius,
            wall_thickness_mm,
            rotation_deg,
            cap_side_deg,
            cap_half_width,
        )
    with BuildPart() as divider_solid:
        with BuildSketch():
            with BuildLine() as divider_ln:
                ln1 = Line(
                    (-case_max_radius * 2, 0),
                    (-lining_outer_radius + wall_thickness_mm, 0),
                )
                ln2 = divider_curve(lining_outer_radius, wall_thickness_mm)
                ln3 = Line(
                    (lining_outer_radius - wall_thickness_mm, 0),
                    (case_max_radius * 2, 0),
                )

                ln4 = Line(
                    (case_max_radius * 2, 0),
                    (case_max_radius * 2, case_max_radius * 2),
                )
                ln5 = Line(
                    (case_max_radius * 2, case_max_radius * 2),
                    (-case_max_radius * 2, case_max_radius * 2),
                )
                ln6 = Line(
                    (-case_max_radius * 2, case_max_radius * 2),
                    (-case_max_radius * 2, 0),
                )
            make_face()
        extrude(amount=1000)

    if rotation_deg:
        divider_solid.part = divider_solid.part.rotate(Axis.Z, rotation_deg)
    return divider_solid


def split_reaches_cap_side(lining_outer_radius, wall_thickness_mm, rotation_deg, cap_side_deg, cap_half_width):
    """Whether build_divider_solid's split, from a wall thickness beyond the lining on a
    divider turned by rotation_deg, can reach the middle of the caps' side at
    cap_side_deg while advancing along the divider (so the halves part along its normal)."""
    turn = np.radians(cap_side_deg - rotation_deg)
    return cap_half_width * np.cos(turn) > lining_outer_radius + wall_thickness_mm


def _build_turned_divider_solid(
    lining_outer_radius,
    case_max_radius,
    wall_thickness_mm,
    rotation_deg,
    cap_side_deg,
    cap_half_width,
):
    """build_divider_solid for a divider turned off the caps' sides."""
    if abs(cap_side_deg - rotation_deg) > 45:
        raise ValueError(f"Cap side at {cap_side_deg} deg is more than 45 deg from the divider at {rotation_deg} deg")
    if not split_reaches_cap_side(lining_outer_radius, wall_thickness_mm, rotation_deg, cap_side_deg, cap_half_width):
        raise ValueError(
            f"A divider at {rotation_deg:.1f} deg cannot reach the middle of the caps' side at "
            f"{cap_side_deg} deg, {cap_half_width:.1f} mm from the axis, without turning back"
        )
    turn = np.radians(cap_side_deg - rotation_deg)
    curve_end = lining_outer_radius - wall_thickness_mm
    elbow = lining_outer_radius + wall_thickness_mm
    far = case_max_radius * 2
    # In the divider's frame (its curve along X, its positive side +Y): the caps' side
    # middles and far points along their normals, either end.
    side = (cap_half_width * np.cos(turn), cap_half_width * np.sin(turn))
    out = (far * np.cos(turn), far * np.sin(turn))
    reach = max(far, abs(out[1]) + far)
    with BuildPart() as divider_solid:
        with BuildSketch():
            with BuildLine():
                points = [(-out[0], -out[1]), (-side[0], -side[1]), (-elbow, 0), (-curve_end, 0)]
                for a, b in zip(points, points[1:]):
                    Line(a, b)
                divider_curve(lining_outer_radius, wall_thickness_mm)
                points = [(curve_end, 0), (elbow, 0), side, out, (out[0], reach), (-out[0], reach), points[0]]
                for a, b in zip(points, points[1:]):
                    Line(a, b)
            make_face()
        extrude(amount=1000)

    divider_solid.part = divider_solid.part.rotate(Axis.Z, rotation_deg)
    return divider_solid


def divider_wall_and_solid(
    lining_outer_radius,
    wall_thickness_mm,
    height,
    case_max_radius,
    rotation_deg=0,
    cap_side_deg=0,
    cap_half_width=None,
):
    """The divider wall and build_divider_solid, both turned by rotation_deg about Z."""
    with BuildPart() as divider_wall:
        with BuildLine():
            ln = divider_curve(
                lining_outer_radius,
                wall_thickness_mm,
            )
            assert isinstance(ln, Wire)
            tangent = ln % 0.5
            orthogonal_plane = Plane(
                origin=(0, 0, 0),
                z_dir=tangent,
            )
        with BuildSketch(orthogonal_plane) as spline_sk:
            Rectangle(
                wall_thickness_mm * 2,
                height,
                align=(Align.CENTER, Align.MAX),
            )
        sweep()

        with BuildSketch(
            Plane(
                origin=(0, 0, 0),
                z_dir=(0, 0, 1),
            )
        ):
            with Locations(
                (-lining_outer_radius + wall_thickness_mm, 0),
                (lining_outer_radius - wall_thickness_mm, 0),
            ):
                Circle(wall_thickness_mm)
        extrude(amount=height)

    if rotation_deg:
        divider_wall.part = divider_wall.part.rotate(Axis.Z, rotation_deg)

    divider_solid = build_divider_solid(
        lining_outer_radius,
        case_max_radius,
        wall_thickness_mm,
        rotation_deg,
        cap_side_deg,
        cap_half_width,
    )

    return divider_wall, divider_solid
