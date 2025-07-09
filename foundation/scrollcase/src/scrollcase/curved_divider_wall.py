import logging

from build123d import *  # type: ignore

logger = logging.getLogger(__name__)


def divider_curve(lining_outer_radius, wall_thickness_mm):
    """The curve that, once extruded, splits the two case halves."""
    pts = [
        (-lining_outer_radius + wall_thickness_mm, 0),
        (-lining_outer_radius / 2, lining_outer_radius / 5),
        (0, 0),
        (lining_outer_radius / 2, -lining_outer_radius / 5),
        (lining_outer_radius - wall_thickness_mm, 0),
    ]
    ln1 = ThreePointArc(pts[0], pts[1], pts[2])
    ln2 = ThreePointArc(pts[2], pts[3], pts[4])
    return ln1 + ln2


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
