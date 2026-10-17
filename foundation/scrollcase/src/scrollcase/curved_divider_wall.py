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


def build_divider_solid(lining_outer_radius, case_max_radius, wall_thickness_mm):
    """A solid that can be intersected or subtracted from a case to produce the right or left halves, respectively."""
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

    return divider_solid


def divider_wall_and_solid(
    lining_outer_radius, wall_thickness_mm, height, case_max_radius
):
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

    divider_solid = build_divider_solid(
        lining_outer_radius,
        case_max_radius,
        wall_thickness_mm,
    )

    return divider_wall, divider_solid
